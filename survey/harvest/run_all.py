from __future__ import annotations

import importlib
import traceback
from typing import Any

from common import load_config, load_secrets, source_enabled, utc_now, write_manifest


SOURCES = ["arxiv", "crossref", "openalex", "springer", "elsevier", "ieee"]


def run_source(source: str, config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    if not source_enabled(config, source):
        return {"status": "disabled", "endpoint": "", "queries": [], "raw_count": 0, "kept_count": 0}
    module = importlib.import_module(f"src_{source}")
    try:
        return module.harvest(config, secrets)
    except Exception as exc:
        return {
            "status": "error",
            "endpoint": getattr(module, "ENDPOINT", ""),
            "queries": [],
            "raw_count": 0,
            "kept_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }


def main() -> None:
    config = load_config()
    secrets = load_secrets()
    manifest: dict[str, Any] = {
        "run_started_at_utc": utc_now(),
        "query_core": config["query_core"],
        "year_from": config["year_from"],
        "year_to": config["year_to"],
        "contact_email": config["contact_email"],
        "sources": {},
        "total_raw_before_dedup": 0,
    }
    for source in SOURCES:
        print(f"=== {source} ===")
        result = run_source(source, config, secrets)
        manifest["sources"][source] = {**result, "run_at_utc": utc_now()}
        manifest["total_raw_before_dedup"] += int(result.get("raw_count", 0) or 0)
        print(f"{source}: {result.get('status')} raw={result.get('raw_count', 0)} kept={result.get('kept_count', 0)}")
    manifest["run_finished_at_utc"] = utc_now()
    write_manifest(manifest)
    print("Wrote survey/raw_exports/run_manifest.json")


if __name__ == "__main__":
    main()
