from __future__ import annotations

import csv
import html
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw_exports"
CONFIG_PATH = Path(__file__).with_name("config.yaml")
SECRETS_PATH = ROOT / ".secrets.env"

SCHEMA = ["source_db", "title", "authors", "year", "venue", "doi", "url", "abstract", "publisher", "retrieved_via"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def normalize_doi(value: str) -> str:
    doi = (value or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(".")


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(to_text(item) for item in value if to_text(item))
    if isinstance(value, dict):
        return "; ".join(to_text(item) for item in value.values() if to_text(item))
    return strip_html(str(value))


def normalize_record(
    *,
    source_db: str,
    title: Any,
    authors: Any = "",
    year: Any = "",
    venue: Any = "",
    doi: Any = "",
    url: Any = "",
    abstract: Any = "",
    publisher: Any = "",
    retrieved_via: str,
) -> dict[str, str]:
    return {
        "source_db": to_text(source_db),
        "title": to_text(title),
        "authors": to_text(authors),
        "year": to_text(year),
        "venue": to_text(venue),
        "doi": normalize_doi(to_text(doi)),
        "url": to_text(url),
        "abstract": to_text(abstract),
        "publisher": to_text(publisher),
        "retrieved_via": retrieved_via,
    }


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return load_config_minimal(path)


def load_config_minimal(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_top: str | None = None
    current_mid: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_top = line[:-1]
            current_mid = None
            config[current_top] = {}
        elif indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            config[key.strip()] = parse_scalar(value)
        elif indent == 2 and line.endswith(":") and current_top:
            current_mid = line[:-1]
            config[current_top][current_mid] = []
        elif indent == 2 and ":" in line and current_top:
            key, value = line.split(":", 1)
            config[current_top][key.strip()] = parse_scalar(value)
            current_mid = key.strip() if isinstance(config[current_top][key.strip()], dict) else current_mid
        elif indent == 4 and line.startswith("- ") and current_top and current_mid:
            config[current_top][current_mid].append(parse_scalar(line[2:]))
        elif indent == 4 and ":" in line and current_top and current_mid:
            if not isinstance(config[current_top].get(current_mid), dict):
                config[current_top][current_mid] = {}
            key, value = line.split(":", 1)
            config[current_top][current_mid][key.strip()] = parse_scalar(value)
    return config


def load_secrets(path: Path = SECRETS_PATH) -> dict[str, str]:
    secrets: dict[str, str] = {}
    if not path.exists():
        return secrets
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        secrets[key.strip()] = value.strip().strip('"').strip("'")
    return secrets


class HttpClient:
    def __init__(self, contact_email: str, min_interval_seconds: float = 1.0, timeout_seconds: int = 30) -> None:
        self.contact_email = contact_email
        self.min_interval_seconds = min_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.last_request_at = 0.0

    @property
    def user_agent(self) -> str:
        return f"JournalXAI/1.0 (mailto:{self.contact_email})"

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, retries: int = 4) -> tuple[bytes, str]:
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{query}"
        request_headers = {"User-Agent": self.user_agent}
        if headers:
            request_headers.update(headers)
        for attempt in range(retries + 1):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            request = urllib.request.Request(url, headers=request_headers)
            self.last_request_at = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read(), response.geturl()
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (2**attempt)
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                    return self.get_with_powershell(url, request_headers)
                if attempt == retries:
                    raise
                time.sleep(2.0 * (2**attempt))
        raise RuntimeError("unreachable retry state")

    def get_with_powershell(self, url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            out_path = tmp.name
        env = os.environ.copy()
        env["CODEX_SURVEY_URL"] = url
        env["CODEX_SURVEY_UA"] = headers.get("User-Agent", self.user_agent)
        env["CODEX_SURVEY_OUT"] = out_path
        ps = (
            "$ProgressPreference='SilentlyContinue'; "
            "$headers=@{'User-Agent'=$env:CODEX_SURVEY_UA}; "
            "Invoke-WebRequest -UseBasicParsing -Uri $env:CODEX_SURVEY_URL "
            "-Headers $headers -TimeoutSec 60 -OutFile $env:CODEX_SURVEY_OUT"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env, capture_output=True, text=True)
            return Path(out_path).read_bytes(), url
        finally:
            try:
                Path(out_path).unlink()
            except FileNotFoundError:
                pass

    def get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> tuple[dict[str, Any], str]:
        body, final_url = self.get(url, params=params, headers=headers)
        return json.loads(body.decode("utf-8-sig")), final_url


def save_raw(source: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    path = RAW_DIR / f"{source}_raw.json"
    pages: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as fh:
            pages = json.load(fh)
    pages.append({"retrieved_at_utc": utc_now(), **payload})
    path.write_text(json.dumps(pages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reset_source_outputs(source: str) -> None:
    for suffix in ["_raw.json", ".csv"]:
        path = RAW_DIR / f"{source}{suffix}"
        if path.exists():
            path.unlink()


def write_source_csv(source: str, rows: list[dict[str, str]]) -> Path:
    ensure_dirs()
    path = RAW_DIR / f"{source}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCHEMA, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SCHEMA})
    return path


def write_manifest(payload: dict[str, Any]) -> None:
    ensure_dirs()
    (RAW_DIR / "run_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def phrase_match(record: dict[str, str], config: dict[str, Any]) -> bool:
    text = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
    block1 = [phrase.lower() for phrase in config["query_core"]["block1"]]
    block2 = [phrase.lower() for phrase in config["query_core"]["block2"]]
    return any(phrase in text for phrase in block1) and any(phrase in text for phrase in block2)


def in_year_range(record: dict[str, str], config: dict[str, Any]) -> bool:
    try:
        year = int(str(record.get("year", ""))[:4])
    except ValueError:
        return False
    return int(config["year_from"]) <= year <= int(config["year_to"])


def local_filter(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in rows if in_year_range(row, config) and phrase_match(row, config)]


def source_enabled(config: dict[str, Any], source: str) -> bool:
    return bool(config.get("sources", {}).get(source, {}).get("enabled", False))


def infer_source_db(publisher: str, venue: str, via: str) -> str:
    haystack = f"{publisher} {venue}".lower()
    if "ieee" in haystack:
        return f"IEEE Xplore (via {via})"
    if "association for computing machinery" in haystack or "acm" in haystack:
        return f"ACM Digital Library (via {via})"
    if "springer" in haystack or "nature" in haystack:
        return f"SpringerLink (via {via})"
    if "elsevier" in haystack or "sciencedirect" in haystack:
        return f"ScienceDirect (via {via})"
    if "arxiv" in haystack:
        return "arXiv"
    return f"Other (via {via})"


def compact_query(config: dict[str, Any], block1_phrase: str, block2_phrase: str | None = None) -> str:
    if block2_phrase:
        return f'"{block1_phrase}" "{block2_phrase}"'
    return f'"{block1_phrase}"'
