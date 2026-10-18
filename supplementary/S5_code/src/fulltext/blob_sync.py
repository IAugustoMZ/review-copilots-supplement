"""Syncs the local verified-OA/CC full-text PDF cache (data/fulltext_pdfs/) to
Azure Blob Storage, replacing the earlier "commit the PDFs to git" approach --
548MB of PDFs was bloating .git for every clone/fetch. See CLAUDE.md
"Verified-OA/CC full-text cache" for the policy this encodes.

Idempotent: run it as many times as you like. Each local PDF is uploaded only
if the blob is missing or its size differs from the local file; everything
else is skipped. Blobs are written at the Cool access tier -- the cheapest
tier that stays online with millisecond access (unlike Archive, which needs a
multi-hour rehydration to read back).

Auth is a connection string, not az CLI login, so this runs from any machine
without extra setup: copy .env.example to .env (gitignored -- never commit
real credentials) and fill in AZURE_STORAGE_CONNECTION_STRING /
AZURE_STORAGE_CONTAINER, then:

    poetry run review-cli fulltext blob-sync
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from azure.storage.blob import BlobServiceClient, StandardBlobTier
from dotenv import load_dotenv

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.fulltext.download import PDF_DIR

DEFAULT_COVERAGE_REPORT_PATH = OUTPUTS_DIR / "fulltext_blob_sync.md"

SyncStatus = str  # "uploaded" | "updated" | "skipped" | "failed"


@dataclass
class SyncOutcome:
    filename: str
    status: SyncStatus
    size_bytes: int = 0
    error: str | None = None


def connection_string_from_env(env: str = "AZURE_STORAGE_CONNECTION_STRING") -> str | None:
    """Loads .env (if present, searching upward from the CWD) and returns the
    connection string. Values already set in the environment take precedence
    over .env, matching python-dotenv's default override=False."""
    load_dotenv()
    return os.environ.get(env)


def container_name_from_env(env: str = "AZURE_STORAGE_CONTAINER") -> str | None:
    load_dotenv()
    return os.environ.get(env)


def sync_to_blob(
    connection_string: str,
    container_name: str,
    pdf_dir: Path = PDF_DIR,
    tier: StandardBlobTier = StandardBlobTier.COOL,
) -> list[SyncOutcome]:
    """Uploads every *.pdf in pdf_dir to container_name, skipping blobs that
    already exist with a matching size. Creates the container if it does not
    exist yet (idempotent, so a second run is a no-op)."""
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()

    outcomes: list[SyncOutcome] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        local_size = pdf_path.stat().st_size
        blob_client = container_client.get_blob_client(pdf_path.name)
        try:
            existing = blob_client.get_blob_properties() if blob_client.exists() else None
            if existing is not None and existing.size == local_size:
                outcomes.append(SyncOutcome(pdf_path.name, "skipped", local_size))
                continue
            status = "updated" if existing is not None else "uploaded"
            with pdf_path.open("rb") as fh:
                blob_client.upload_blob(
                    fh, overwrite=True, standard_blob_tier=tier
                )
            outcomes.append(SyncOutcome(pdf_path.name, status, local_size))
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the batch
            outcomes.append(SyncOutcome(pdf_path.name, "failed", local_size, error=str(exc)))

    return outcomes


def write_sync_report(
    outcomes: list[SyncOutcome], container_name: str, tier: StandardBlobTier, out_path: Path = DEFAULT_COVERAGE_REPORT_PATH
) -> None:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    total_bytes = sum(o.size_bytes for o in outcomes if o.status in ("uploaded", "updated"))

    lines = [
        "# Full-text PDF -> Azure Blob Storage sync report",
        "",
        f"- Container: `{container_name}`",
        f"- Access tier: `{tier.value if hasattr(tier, 'value') else tier}`",
        f"- Files considered: {len(outcomes)}",
        f"- Uploaded: {counts.get('uploaded', 0)}",
        f"- Updated (local size differed from blob): {counts.get('updated', 0)}",
        f"- Skipped (already up to date): {counts.get('skipped', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        f"- Bytes transferred this run: {total_bytes:,}",
        "",
        "| File | Status | Size (bytes) | Error |",
        "|---|---|---|---|",
    ]
    for o in outcomes:
        lines.append(f"| {o.filename} | {o.status} | {o.size_bytes:,} | {o.error or ''} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def push_file_to_blob(
    connection_string: str,
    container_name: str,
    local_path: Path,
    blob_name: str | None = None,
    tier: StandardBlobTier = StandardBlobTier.COOL,
) -> SyncOutcome:
    """Uploads one file (not a directory) to the same container the PDF cache
    uses, skipping the transfer when the blob already matches byte-for-byte in
    size. Used for the compressed DB snapshot, which outgrew what git should
    carry -- see `review-cli db blob-push`."""
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()

    name = blob_name or local_path.name
    local_size = local_path.stat().st_size
    blob_client = container_client.get_blob_client(name)
    try:
        existing = blob_client.get_blob_properties() if blob_client.exists() else None
        if existing is not None and existing.size == local_size:
            return SyncOutcome(name, "skipped", local_size)
        status = "updated" if existing is not None else "uploaded"
        with local_path.open("rb") as fh:
            blob_client.upload_blob(fh, overwrite=True, standard_blob_tier=tier)
        return SyncOutcome(name, status, local_size)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised, like sync_to_blob
        return SyncOutcome(name, "failed", local_size, error=str(exc))
