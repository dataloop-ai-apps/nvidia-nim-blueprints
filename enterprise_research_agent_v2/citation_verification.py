"""
Citation verification and report sanitization for AIQ v2.

Ported from the NVIDIA AI-Q Blueprint v2.0.0 citation_verification.py.
Provides:
  - SourceRegistry: tracks all URLs returned by search tools
  - verify_citations(): validates report citations against the registry
  - sanitize_report(): strips hallucinated/unsafe URLs from report body
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Regex patterns for citation parsing
_CITATION_INLINE_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_SECTION_RE = re.compile(
    r"^(?:#{1,3}\s*|\*\*)"
    r"(Sources|References|Reference List|Bibliography)"
    r"(?:\*?\*?\s*:?\s*)$",
    re.MULTILINE | re.IGNORECASE,
)
_CITATION_LINE_RE = re.compile(
    r"^\s*[-*]?\s*\[(\d+)\]\s*(.*?)$", re.MULTILINE
)
_URL_IN_LINE_RE = re.compile(r"https?://[^\s\),\"'>]+")
_BODY_URL_RE = re.compile(r"(?<!\[)\bhttps?://[^\s\),\"'>]+")
_SHORTENED_URL_RE = re.compile(
    r"https?://(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|is\.gd)/\S+"
)

MIN_URL_LENGTH = 15
MAX_URL_LENGTH = 500


@dataclass
class Source:
    """A single source captured from tool results."""

    url: Optional[str] = None
    title: Optional[str] = None
    citation_key: Optional[str] = None

    def __hash__(self):
        return hash(self.url or self.citation_key or "")

    def __eq__(self, other):
        if not isinstance(other, Source):
            return False
        if self.url and other.url:
            return _normalize_url(self.url) == _normalize_url(other.url)
        return self.citation_key == other.citation_key


class SourceRegistry:
    """Registry tracking all URLs/sources returned by search tools during a research session."""

    def __init__(self):
        self._sources: list[Source] = []
        self._url_index: dict[str, Source] = {}

    def add(self, source: Source) -> None:
        if source.url:
            norm = _normalize_url(source.url)
            if norm in self._url_index:
                return
            self._url_index[norm] = source
        self._sources.append(source)

    def all_sources(self) -> list[Source]:
        return list(self._sources)

    def resolve_url(self, url: str) -> Optional[Source]:
        """5-level URL matching against the registry."""
        norm = _normalize_url(url)
        if norm in self._url_index:
            return self._url_index[norm]

        # Level 2: strip trailing slash
        stripped = norm.rstrip("/")
        if stripped in self._url_index:
            return self._url_index[stripped]

        # Level 3: domain + path prefix match
        parsed = urlparse(norm)
        for registered_norm, source in self._url_index.items():
            reg_parsed = urlparse(registered_norm)
            if parsed.netloc == reg_parsed.netloc and (
                reg_parsed.path.startswith(parsed.path)
                or parsed.path.startswith(reg_parsed.path)
            ):
                return source

        # Level 4: domain-only match (last resort)
        for registered_norm, source in self._url_index.items():
            reg_parsed = urlparse(registered_norm)
            if parsed.netloc == reg_parsed.netloc:
                return source

        return None

    def clear(self) -> None:
        self._sources.clear()
        self._url_index.clear()

    def to_dict(self) -> list[dict]:
        return [
            {"url": s.url, "title": s.title, "citation_key": s.citation_key}
            for s in self._sources
        ]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "SourceRegistry":
        registry = cls()
        for item in data:
            registry.add(Source(
                url=item.get("url"),
                title=item.get("title"),
                citation_key=item.get("citation_key"),
            ))
        return registry

    def get_source_list_text(self) -> str:
        """Format all sources as a numbered list for the get_verified_sources tool."""
        if not self._sources:
            return ""
        lines = ["## Verified Sources (use ONLY these in your report)"]
        for i, source in enumerate(self._sources, 1):
            title = source.title or "Untitled"
            url = source.url or source.citation_key or "N/A"
            lines.append(f"[{i}] {title}: {url}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Result of citation verification."""

    verified_report: str
    valid_citations: list[dict] = field(default_factory=list)
    removed_citations: list[dict] = field(default_factory=list)


@dataclass
class SanitizationResult:
    """Result of report sanitization."""

    sanitized_report: str
    removed_urls: list[str] = field(default_factory=list)


class EmptySourceRegistryError(Exception):
    """Raised when research produces no verifiable sources."""

    def __init__(self, research_type: str = "research"):
        self.research_type = research_type
        super().__init__(
            f"The search tools did not return any results for this question. "
            f"This may be due to a temporary issue or the question may need to be rephrased. "
            f"Please try again."
        )


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison: lowercase scheme+host, strip fragments."""
    try:
        parsed = urlparse(url.strip().rstrip(".,;)"))
        normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception:
        return url.strip().lower()


def _is_knowledge_citation(ref_text: str, registry: SourceRegistry) -> tuple[bool, Optional[str]]:
    """Check if a reference line is a knowledge-layer citation (e.g. 'filename.pdf, p.X')."""
    if re.search(r"\.(pdf|docx?|xlsx?|pptx?|txt|csv|md)\b", ref_text, re.IGNORECASE):
        return True, ref_text.strip()
    return False, None


def extract_sources_from_tool_result(tool_name: str, content: str) -> list[Source]:
    """Extract Source objects from a tool call result string."""
    sources = []
    urls_found = _URL_IN_LINE_RE.findall(content)
    for url in urls_found:
        url = url.rstrip(".,;)")
        if len(url) < MIN_URL_LENGTH or len(url) > MAX_URL_LENGTH:
            continue
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            continue
        title = _extract_title_near_url(content, url)
        sources.append(Source(url=url, title=title or tool_name))
    return sources


def _extract_title_near_url(content: str, url: str) -> Optional[str]:
    """Try to extract a title near a URL in the content."""
    idx = content.find(url)
    if idx < 0:
        return None
    before = content[max(0, idx - 200):idx]
    title_match = re.search(r'["\']([^"\']{10,100})["\']', before)
    if title_match:
        return title_match.group(1)
    line_start = before.rfind("\n")
    if line_start >= 0:
        line_text = before[line_start + 1:].strip()
        line_text = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line_text)
        if 5 < len(line_text) < 150:
            return line_text
    return None


def verify_citations(report: str, registry: SourceRegistry) -> VerificationResult:
    """
    Verify all citations in a report against the source registry.

    - Parses the References/Sources section
    - Checks each citation URL against the registry
    - Removes invalid citations and renumbers the remaining ones
    - Updates inline [N] references to match new numbering
    """
    ref_match = _REFERENCE_SECTION_RE.search(report)
    if not ref_match:
        return VerificationResult(verified_report=report)

    body = report[:ref_match.start()]
    ref_section = report[ref_match.start():]

    valid = []
    removed = []
    citation_map = {}

    for line_match in _CITATION_LINE_RE.finditer(ref_section):
        number = int(line_match.group(1))
        ref_text = line_match.group(2).strip()

        url_match = _URL_IN_LINE_RE.search(ref_text)
        if url_match:
            url = url_match.group(0).rstrip(".,;)")
            source = registry.resolve_url(url)
            if source:
                valid.append({"number": number, "line": line_match.group(0), "url": url})
            else:
                removed.append({
                    "number": number,
                    "line": line_match.group(0),
                    "reason": "url_not_in_registry",
                })
            continue

        is_kl, citation_key = _is_knowledge_citation(ref_text, registry)
        if is_kl and citation_key:
            valid.append({"number": number, "line": line_match.group(0), "citation_key": citation_key})
        else:
            removed.append({
                "number": number,
                "line": line_match.group(0),
                "reason": "unverifiable_citation",
            })

    if not removed:
        return VerificationResult(
            verified_report=report,
            valid_citations=valid,
            removed_citations=[],
        )

    # Build renumbering map: old_number -> new_number
    new_num = 1
    for v in valid:
        citation_map[v["number"]] = new_num
        new_num += 1

    removed_numbers = {r["number"] for r in removed}

    # Renumber inline citations in body
    def _replace_inline(match):
        old_num = int(match.group(1))
        if old_num in removed_numbers:
            return ""
        new = citation_map.get(old_num)
        return f"[{new}]" if new else match.group(0)

    new_body = _CITATION_INLINE_RE.sub(_replace_inline, body)
    new_body = re.sub(r"\s{2,}", " ", new_body)

    # Rebuild references section
    ref_header_line = ref_section[:ref_section.find("\n") + 1] if "\n" in ref_section else ref_section
    new_ref_lines = [ref_header_line.rstrip()]
    for v in valid:
        old_line = v["line"]
        old_num_str = str(v["number"])
        new_num = citation_map[v["number"]]
        new_line = old_line.replace(f"[{old_num_str}]", f"[{new_num}]", 1)
        new_ref_lines.append(new_line)

    verified_report = new_body.rstrip() + "\n\n" + "\n".join(new_ref_lines) + "\n"

    return VerificationResult(
        verified_report=verified_report,
        valid_citations=valid,
        removed_citations=removed,
    )


def sanitize_report(report: str) -> SanitizationResult:
    """
    Sanitize a report by removing:
    - Bare URLs in body text (outside References section)
    - Shortened URLs anywhere
    - URLs that look unsafe
    """
    ref_match = _REFERENCE_SECTION_RE.search(report)
    if ref_match:
        body = report[:ref_match.start()]
        ref_section = report[ref_match.start():]
    else:
        body = report
        ref_section = ""

    removed = []

    # Remove bare URLs from body (not in citation brackets)
    def _remove_body_url(match):
        url = match.group(0)
        removed.append(url)
        return ""

    sanitized_body = _BODY_URL_RE.sub(_remove_body_url, body)

    # Remove shortened URLs from everywhere
    def _remove_shortened(match):
        url = match.group(0)
        if url not in removed:
            removed.append(url)
        return ""

    sanitized_body = _SHORTENED_URL_RE.sub(_remove_shortened, sanitized_body)
    sanitized_ref = _SHORTENED_URL_RE.sub(_remove_shortened, ref_section)

    # Clean up extra whitespace from removals
    sanitized_body = re.sub(r" {2,}", " ", sanitized_body)
    sanitized_body = re.sub(r"\n{3,}", "\n\n", sanitized_body)

    sanitized_report = sanitized_body + sanitized_ref

    return SanitizationResult(
        sanitized_report=sanitized_report,
        removed_urls=removed,
    )
