"""Shared types every search connector produces/consumes.

A connector's only job is: given a query string and filters, return a
ConnectorResponse -- status is always one of ok/zero_results/error, never an
uncaught exception, so a caller (src/search/executor.py) can always write a
complete search_runs row. See docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 3
and the "never silently discard a failure" requirement in section 17/23.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

ConnectorStatus = Literal["ok", "zero_results", "error"]


class AuthorStub(BaseModel):
    full_name: str
    surname: str
    position: int


class RawRecord(BaseModel):
    """One candidate record as reported by a single source, pre-enrichment/dedup."""

    doi: str | None = None
    title: str
    year: int | None = None
    venue: str | None = None
    publication_type: str | None = None
    authors: list[AuthorStub] = Field(default_factory=list)
    abstract: str | None = None
    abstract_source: str | None = None
    keywords: list[str] = Field(default_factory=list)
    publisher: str | None = None
    issn: str | None = None
    citation_count: int | None = None
    oa_status: str | None = None
    oa_url: str | None = None
    raw_source_id: str
    raw_metadata: dict = Field(default_factory=dict)


@dataclass
class QueryFilters:
    from_year: int | None = None
    to_year: int | None = None
    rows: int = 50
    extra: dict = field(default_factory=dict)


@dataclass
class ConnectorResponse:
    status: ConnectorStatus
    records: list[RawRecord] = field(default_factory=list)
    page_count: int = 0
    api_version: str | None = None
    error_detail: str | None = None
    rate_limit_events: int = 0


class SearchConnector(ABC):
    """Protocol every source connector implements. See plan section 3."""

    name: str

    @abstractmethod
    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse: ...
