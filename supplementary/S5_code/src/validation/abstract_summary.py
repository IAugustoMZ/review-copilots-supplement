"""Phase 11 support: the short summary shown by default on the blinded-review
screen, with [f] revealing the full abstract.

docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 16 originally specced an
AI-paraphrased summary with a banned-phrase lint. No AI backend (API key) is
available in this environment, so this is extractive instead: the first
max_sentences of the record's own abstract, verbatim. This is a deliberate,
documented simplification -- an LLM-generated summary would need its own
banned-phrase lint to avoid smuggling in evaluative language ("clearly
include this", capability-level claims); showing the authors' own sentences
verbatim cannot introduce evaluative bias that wasn't already in the source
text, and BANNED_PHRASES / lint_summary() are still applied as a defense-in-
depth check on the resulting slice, and are ready to reuse for a future
AI-generated version.
"""

from __future__ import annotations

import re

NOT_AVAILABLE = "not available — press [f] for full abstract"

# Drawn from the brief's own forbidden examples: evaluative/inclusion or
# capability-level language has no place in a supposedly neutral summary.
BANNED_PHRASES = [
    "include this",
    "exclude this",
    "clearly",
    "obviously",
    "should be included",
    "should be excluded",
    "prototype",
    "deployment-ready",
    "level 1",
    "level 2",
    "level 3",
    "level 4",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def lint_summary(text: str) -> list[str]:
    """Returns the banned phrases found in text (case-insensitive), empty if clean."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def generate_abstract_summary(abstract: str | None, max_sentences: int = 2) -> str:
    if not abstract or not abstract.strip():
        return NOT_AVAILABLE

    sentences = _SENTENCE_SPLIT_RE.split(abstract.strip())
    summary = " ".join(sentences[:max_sentences]).strip()
    if not summary or lint_summary(summary):
        return NOT_AVAILABLE
    return summary
