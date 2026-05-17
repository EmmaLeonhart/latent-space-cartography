"""
Provenance resolver: given a DATE, return the Ollama runtime version and the
mxbai-embed-large model revision that were current on/before that date.

Why this exists
---------------
The repo previously hard-coded "Ollama 0.6.5 = the 2026-04-06 discovery
baseline". That pin is fabricated: v0.6.5 shipped in early 2025, roughly a
year before the claimed discovery date (Ollama was ~v0.20.2 on 2026-04-06).
Hard-coded version guesses are how the collision CI ended up asserting
against a configuration the defect never lived in.

This script makes provenance *derived from the date* instead of guessed:

  - Ollama:  the latest STABLE ollama/ollama GitHub release whose
             publish time is <= the query date (GitHub Releases API).
  - Model:   the mixedbread-ai/mxbai-embed-large-v1 HuggingFace commit
             that was repo HEAD at the query date (HF commits API). This
             is the upstream provenance; see the Ollama-registry caveat
             printed in the notes.

Stdlib only. Set GITHUB_TOKEN (or GH_TOKEN) in the environment to lift the
unauthenticated GitHub API rate limit (CI sets GITHUB_TOKEN automatically).

Usage
-----
  python scripts/resolve_versions_for_date.py --date 2026-04-06
  python scripts/resolve_versions_for_date.py --date 2026-04-06 --json
  python scripts/resolve_versions_for_date.py --date 2026-04-06 \
      --field ollama_version          # bare value, for shell capture
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OLLAMA_RELEASES = "https://api.github.com/repos/ollama/ollama/releases"
HF_REPO = "mixedbread-ai/mxbai-embed-large-v1"
HF_COMMITS = f"https://huggingface.co/api/models/{HF_REPO}/commits/main"


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 timestamp (with trailing 'Z') to aware UTC."""
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()), resp.headers


def _gh_headers() -> dict:
    h = {"User-Agent": "lsc-provenance/1.0",
         "Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def resolve_ollama(cutoff: datetime) -> dict:
    """Latest STABLE ollama release published at or before `cutoff`."""
    best = None
    page = 1
    while True:
        try:
            rels, _ = _get(f"{OLLAMA_RELEASES}?per_page=100&page={page}",
                           _gh_headers())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"GitHub API error {e.code}: {e.read().decode()[:200]}")
        if not rels:
            break
        for r in rels:
            tag = r.get("tag_name", "")
            if r.get("prerelease") or "-rc" in tag or "-beta" in tag:
                continue
            pub = r.get("published_at")
            if not pub:
                continue
            pub_dt = _parse_iso(pub)
            if pub_dt <= cutoff and (best is None or pub_dt > best["_dt"]):
                best = {"tag": tag,
                        "version": tag.lstrip("v"),
                        "published_at": pub,
                        "_dt": pub_dt}
        page += 1
        if page > 20:  # safety: ollama has < 2000 releases
            break
    if best is None:
        raise SystemExit(f"No stable Ollama release found on/before {cutoff.date()}")
    best.pop("_dt")
    return best


def resolve_model(cutoff: datetime) -> dict:
    """HF commit that was HEAD of the model repo at/before `cutoff`."""
    try:
        commits, _ = _get(HF_COMMITS, {"User-Agent": "lsc-provenance/1.0"})
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HF API error {e.code}: {e.read().decode()[:200]}")
    chosen = None
    for c in commits:  # HF returns newest-first
        cd = c.get("date")
        if cd and _parse_iso(cd) <= cutoff:
            chosen = c
            break
    if chosen is None:
        # Query date predates the repo's first commit on `main`.
        oldest = commits[-1] if commits else {}
        return {"repo": HF_REPO, "revision": None,
                "commit_date": None,
                "note": "query date predates the model repo history; "
                        f"oldest known commit is {oldest.get('id','?')[:12]} "
                        f"({oldest.get('date','?')})"}
    return {"repo": HF_REPO,
            "revision": chosen.get("id"),
            "commit_date": chosen.get("date"),
            "title": (chosen.get("title") or "").strip()}


def resolve(date_str: str) -> dict:
    try:
        cutoff = _parse_iso(date_str if "T" in date_str else f"{date_str}T23:59:59Z")
    except ValueError:
        raise SystemExit(f"Could not parse --date {date_str!r}; use YYYY-MM-DD")
    ollama = resolve_ollama(cutoff)
    model = resolve_model(cutoff)
    return {
        "query_date": date_str,
        "ollama": ollama,
        "hf_model": model,
        "notes": [
            "ollama.version is the string to pass as OLLAMA_VERSION to "
            "ollama.com/install.sh (no leading 'v').",
            "hf_model.revision is the UPSTREAM source provenance. Ollama's "
            "library model `mxbai-embed-large` is a separately-built GGUF; "
            "its registry digest is not historically queryable by date, so "
            "byte-exact model reproduction before 'now' requires pinning "
            "this HF revision and rebuilding the GGUF, not `ollama pull`.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve Ollama + model versions for a date.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (or full ISO-8601)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--field", help="print only one value: "
                    "ollama_version | ollama_tag | hf_revision")
    args = ap.parse_args()

    result = resolve(args.date)

    if args.field:
        flat = {
            "ollama_version": result["ollama"]["version"],
            "ollama_tag": result["ollama"]["tag"],
            "hf_revision": result["hf_model"].get("revision") or "",
        }
        if args.field not in flat:
            raise SystemExit(f"--field must be one of {sorted(flat)}")
        print(flat[args.field])
        return

    if args.json:
        print(json.dumps(result, indent=2))
        return

    o, m = result["ollama"], result["hf_model"]
    print(f"Provenance for {result['query_date']}")
    print(f"  Ollama : {o['version']}  (tag {o['tag']}, released {o['published_at']})")
    print(f"  Model  : {m['repo']}")
    print(f"           revision {m.get('revision')}  ({m.get('commit_date')})")
    if m.get("title"):
        print(f"           commit: {m['title']}")
    if m.get("note"):
        print(f"           note: {m['note']}")


if __name__ == "__main__":
    main()
