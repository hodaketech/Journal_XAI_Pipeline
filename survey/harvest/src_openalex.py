from __future__ import annotations

from typing import Any

from common import HttpClient, compact_query, infer_source_db, local_filter, normalize_record, reset_source_outputs, save_raw, to_text, write_source_csv


ENDPOINT = "https://api.openalex.org/works"


def inverted_abstract(value: dict[str, list[int]] | None) -> str:
    if not value:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in value.items():
        for index in indexes:
            positions[int(index)] = word
    return " ".join(positions[index] for index in sorted(positions))


def authors(item: dict[str, Any]) -> str:
    names = []
    for authorship in item.get("authorships", []) or []:
        author = authorship.get("author", {}) or {}
        if author.get("display_name"):
            names.append(author["display_name"])
    return "; ".join(names)


def normalize_item(item: dict[str, Any]) -> dict[str, str]:
    source = (((item.get("primary_location") or {}).get("source")) or {})
    venue = to_text(source.get("display_name", ""))
    publisher = to_text(source.get("host_organization_name", "") or source.get("host_organization", ""))
    return normalize_record(
        source_db=infer_source_db(publisher, venue, "OpenAlex"),
        title=item.get("display_name") or item.get("title", ""),
        authors=authors(item),
        year=item.get("publication_year", ""),
        venue=venue,
        doi=item.get("doi", ""),
        url=item.get("id", ""),
        abstract=inverted_abstract(item.get("abstract_inverted_index")),
        publisher=publisher,
        retrieved_via="openalex_api",
    )


def harvest(config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    source = "openalex"
    reset_source_outputs(source)
    api_key = secrets.get("OPENALEX_API_KEY", "")
    if not api_key:
        return {
            "endpoint": ENDPOINT,
            "queries": [],
            "status": "skipped_no_key",
            "raw_count": 0,
            "kept_count": 0,
            "warning": "OPENALEX_API_KEY missing in survey/.secrets.env",
        }
    limits = config.get("limits", {})
    client = HttpClient(config["contact_email"], min_interval_seconds=float(limits.get("default_min_interval_seconds", 1.0)))
    raw_count = 0
    kept: list[dict[str, str]] = []
    queries: list[str] = []
    max_pages = int(limits.get("max_pages_per_query", 3))
    per_page = min(int(limits.get("openalex_per_page", 100)), 100)
    date_filter = f"from_publication_date:{config['year_from']}-01-01,to_publication_date:{config['year_to']}-12-31"

    for phrase1 in config["query_core"]["block1"]:
        for phrase2 in config["query_core"]["block2"]:
            search = compact_query(config, phrase1, phrase2)
            queries.append(search)
            cursor = "*"
            for _page in range(max_pages):
                params = {
                    "api_key": api_key,
                    "search": search,
                    "filter": date_filter,
                    "per_page": per_page,
                    "cursor": cursor,
                    "select": "id,doi,display_name,title,publication_year,primary_location,authorships,abstract_inverted_index",
                }
                payload, final_url = client.get_json(ENDPOINT, params=params)
                items = payload.get("results", []) or []
                raw_count += len(items)
                safe_query = {k: ("<redacted>" if k == "api_key" else v) for k, v in params.items()}
                save_raw(source, {"endpoint": ENDPOINT, "url": final_url.replace(api_key, "<redacted>"), "query": safe_query, "payload": payload})
                kept.extend(local_filter([normalize_item(item) for item in items], config))
                next_cursor = (payload.get("meta") or {}).get("next_cursor")
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
