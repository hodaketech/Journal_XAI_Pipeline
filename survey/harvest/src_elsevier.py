from __future__ import annotations

from typing import Any

from common import HttpClient, compact_query, local_filter, normalize_record, reset_source_outputs, save_raw, write_source_csv


ENDPOINT = "https://api.elsevier.com/content/search/sciencedirect"


def normalize_item(item: dict[str, Any]) -> dict[str, str]:
    return normalize_record(
        source_db="ScienceDirect",
        title=item.get("dc:title", ""),
        authors=item.get("dc:creator", ""),
        year=str(item.get("prism:coverDate", ""))[:4],
        venue=item.get("prism:publicationName", ""),
        doi=item.get("prism:doi", ""),
        url=item.get("prism:url", ""),
        abstract=item.get("dc:description", ""),
        publisher="Elsevier",
        retrieved_via="publisher_api",
    )


def harvest(config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    source = "elsevier"
    reset_source_outputs(source)
    api_key = secrets.get("ELSEVIER_API_KEY", "")
    if not api_key:
        return {"endpoint": ENDPOINT, "queries": [], "status": "skipped_no_key", "raw_count": 0, "kept_count": 0, "warning": "ELSEVIER_API_KEY missing in survey/.secrets.env"}
    limits = config.get("limits", {})
    client = HttpClient(config["contact_email"], min_interval_seconds=float(limits.get("default_min_interval_seconds", 1.0)))
    raw_count = 0
    kept: list[dict[str, str]] = []
    queries: list[str] = []
    max_pages = int(limits.get("max_pages_per_query", 3))
    for phrase1 in config["query_core"]["block1"]:
        for phrase2 in config["query_core"]["block2"]:
            query = compact_query(config, phrase1, phrase2)
            queries.append(query)
            for page in range(max_pages):
                params = {"query": query, "date": f"{config['year_from']}-{config['year_to']}", "start": page * 100, "count": 100}
                payload, final_url = client.get_json(ENDPOINT, params=params, headers={"X-ELS-APIKey": api_key, "Accept": "application/json"})
                entries = ((payload.get("search-results") or {}).get("entry") or [])
                raw_count += len(entries)
                save_raw(source, {"endpoint": ENDPOINT, "url": final_url, "query": params, "payload": payload})
                kept.extend(local_filter([normalize_item(item) for item in entries], config))
                if not entries:
                    break
    csv_path = write_source_csv(source, kept)
    return {"endpoint": ENDPOINT, "queries": queries, "status": "ok", "raw_count": raw_count, "kept_count": len(kept), "csv": str(csv_path)}
