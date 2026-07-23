from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from common import HttpClient, compact_query, local_filter, normalize_record, reset_source_outputs, save_raw, write_source_csv


ENDPOINT = "http://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}totalResults"


def text(entry: ET.Element, path: str) -> str:
    found = entry.find(path, ATOM)
    return "".join(found.itertext()).strip() if found is not None else ""


def entry_year(entry: ET.Element) -> str:
    published = text(entry, "atom:published")
    match = re.match(r"(\d{4})", published)
    return match.group(1) if match else ""


def entry_authors(entry: ET.Element) -> str:
    names = []
    for author in entry.findall("atom:author", ATOM):
        name = author.find("atom:name", ATOM)
        if name is not None and name.text:
            names.append(name.text.strip())
    return "; ".join(names)


def entry_url(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", ATOM):
        href = link.attrib.get("href", "")
        if href:
            return href
    return text(entry, "atom:id")


def entry_doi(entry: ET.Element) -> str:
    doi = text(entry, "arxiv:doi")
    if doi:
        return doi
    return ""


def parse_feed(body: bytes) -> tuple[int, list[dict[str, str]]]:
    root = ET.fromstring(body)
    total_text = root.findtext(OPENSEARCH) or "0"
    rows = []
    for entry in root.findall("atom:entry", ATOM):
        rows.append(
            normalize_record(
                source_db="arXiv",
                title=text(entry, "atom:title"),
                authors=entry_authors(entry),
                year=entry_year(entry),
                venue="arXiv",
                doi=entry_doi(entry),
                url=entry_url(entry),
                abstract=text(entry, "atom:summary"),
                publisher="arXiv",
                retrieved_via="arxiv_api",
            )
        )
    return int(total_text), rows


def harvest(config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    source = "arxiv"
    reset_source_outputs(source)
    limits = config.get("limits", {})
    client = HttpClient(
        config["contact_email"],
        min_interval_seconds=float(limits.get("arxiv_min_interval_seconds", 3.2)),
        timeout_seconds=45,
    )
    raw_count = 0
    kept: list[dict[str, str]] = []
    queries: list[str] = []
    max_pages = int(limits.get("max_pages_per_query", 3))
    page_size = int(limits.get("arxiv_max_results", 100))

    for phrase in config["query_core"]["block1"]:
        query = f'all:"{phrase}"'
        queries.append(query)
        for page in range(max_pages):
            start = page * page_size
            params = {"search_query": query, "start": start, "max_results": page_size, "sortBy": "submittedDate", "sortOrder": "descending"}
            body, final_url = client.get(ENDPOINT, params=params)
            total, rows = parse_feed(body)
            raw_count += len(rows)
            save_raw(source, {"endpoint": ENDPOINT, "url": final_url, "query": params, "total_results": total, "payload_xml": body.decode("utf-8", errors="replace")})
            kept.extend(local_filter(rows, config))
            if start + page_size >= total or not rows:
                break

    csv_path = write_source_csv(source, kept)
    return {
        "endpoint": ENDPOINT,
        "queries": queries,
        "status": "ok",
        "raw_count": raw_count,
        "kept_count": len(kept),
        "csv": str(csv_path),
    }
