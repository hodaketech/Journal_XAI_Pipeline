from __future__ import annotations

from typing import Any

from common import HttpClient, compact_query, local_filter, normalize_record, reset_source_outputs, save_raw, write_source_csv


ENDPOINT = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


def normalize_item(item: dict[str, Any]) -> dict[str, str]:
    return normalize_record(
        source_db="IEEE Xplore",
        title=item.get("title", ""),
        authors=[author.get("full_name", "") for author in item.get("authors", {}).get("authors", [])] if isinstance(item.get("authors"), dict) else "",
        year=item.get("publication_year", ""),
        venue=item.get("publication_title", ""),
        doi=item.get("doi", ""),
        url=item.get("html_url", ""),
        abstract=item.get("abstract", ""),
        publisher="IEEE",
        retrieved_via="publisher_api",
    )


def harvest(config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    source = "ieee"
    reset_source_outputs(source)
    api_key = secrets.get("IEEE_API_KEY", "")
    if not api_key:
        return {"endpoint": ENDPOINT, "queries": [], "status": "skipped_no_key", "raw_count": 0, "kept_count": 0, "warning": "IEEE_API_KEY missing in survey/.secrets.env"}
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
                params = {
                    "apikey": api_key,
                    "querytext": query,
                    "start_year": config["year_from"],
                    "end_year": config["year_to"],
                    "start_record": page * 100 + 1,
                    "max_records": 100,
                }
                payload, final_url = client.get_json(ENDPOINT, params=params)
                articles = payload.get("articles", []) or []
                raw_count += len(articles)
                safe_query = {k: ("<redacted>" if k == "apikey" else v) for k, v in params.items()}
                save_raw(source, {"endpoint": ENDPOINT, "url": final_url.replace(api_key, "<redacted>"), "query": safe_query, "payload": payload})
                kept.extend(local_filter([normalize_item(item) for item in articles], config))
                if not articles:
                    break
    csv_path = write_source_csv(source, kept)
    return {"endpoint": ENDPOINT, "queries": queries, "status": "ok", "raw_count": raw_count, "kept_count": len(kept), "csv": str(csv_path)}
