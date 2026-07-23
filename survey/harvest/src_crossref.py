from __future__ import annotations

from typing import Any

from common import HttpClient, compact_query, infer_source_db, local_filter, normalize_record, reset_source_outputs, save_raw, to_text, write_source_csv


ENDPOINT = "https://api.crossref.org/works"


def first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def year_from_item(item: dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "published", "issued", "created"]:
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def authors(item: dict[str, Any]) -> str:
    names = []
    for author in item.get("author", []) or []:
        name = " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part)
        if name:
            names.append(name)
    return "; ".join(names)


def normalize_item(item: dict[str, Any]) -> dict[str, str]:
    venue = to_text(first(item.get("container-title", "")))
    publisher = to_text(item.get("publisher", ""))
    return normalize_record(
        source_db=infer_source_db(publisher, venue, "Crossref"),
        title=first(item.get("title", "")),
        authors=authors(item),
        year=year_from_item(item),
        venue=venue,
        doi=item.get("DOI", ""),
        url=item.get("URL", ""),
        abstract=item.get("abstract", ""),
        publisher=publisher,
        retrieved_via="crossref_api",
    )


def harvest(config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    source = "crossref"
    reset_source_outputs(source)
    limits = config.get("limits", {})
    client = HttpClient(config["contact_email"], min_interval_seconds=float(limits.get("default_min_interval_seconds", 1.0)))
    raw_count = 0
    kept: list[dict[str, str]] = []
    queries: list[str] = []
    max_pages = int(limits.get("max_pages_per_query", 3))
    rows = int(limits.get("crossref_rows", 100))
    date_filter = f"from-pub-date:{config['year_from']}-01-01,until-pub-date:{config['year_to']}-12-31"

    for phrase1 in config["query_core"]["block1"]:
        for phrase2 in config["query_core"]["block2"]:
            query = compact_query(config, phrase1, phrase2)
            queries.append(query)
            cursor = "*"
            for _page in range(max_pages):
                params = {
                    "query.bibliographic": query,
                    "filter": date_filter,
                    "rows": rows,
                    "cursor": cursor,
                    "mailto": config["contact_email"],
                }
                payload, final_url = client.get_json(ENDPOINT, params=params)
                message = payload.get("message", {})
                items = message.get("items", []) or []
                raw_count += len(items)
                save_raw(source, {"endpoint": ENDPOINT, "url": final_url, "query": params, "payload": payload})
                kept.extend(local_filter([normalize_item(item) for item in items], config))
                next_cursor = message.get("next-cursor")
                if not items or not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

    csv_path = write_source_csv(source, kept)
    return {
        "endpoint": ENDPOINT,
        "queries": queries,
        "status": "ok",
        "raw_count": raw_count,
        "kept_count": len(kept),
        "csv": str(csv_path),
    }
