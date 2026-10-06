"""
fetch_fields.py - replace the guessed data fields in operators.json with the exact fields
BRAIN offers your account, for the same settings alpha_loop.py simulates with.

Run from the alpha-miner folder (it uses the same .env and saved session as alpha_loop.py):
    python fetch_fields.py               # the 80 most-used fields with coverage >= 0.8
    python fetch_fields.py --top 120     # give the generator more fields

It writes three files:
    data_fields_cache.json   every field BRAIN returned, so you can look things up later
    operators.json           "fields" replaced by the selection; the operators are untouched
    operators.json.bak       your previous operators.json, in case you want it back

This only lists fields. It runs no simulations, submits nothing, and uses no simulation quota.

Where the API details come from: BRAIN publishes no official API docs, so the endpoint
(GET /data-fields, paged 50 at a time, replying {count, results}) follows community clients
that recorded BRAIN's real responses. If BRAIN rejects a request, the script stops and prints
BRAIN's own error message so it can be fixed quickly.
"""
import argparse
import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import alpha_loop as al
import brain_engine as be

logger = logging.getLogger("fetch_fields")

DATA_FIELDS_URL = "https://api.worldquantbrain.com/data-fields"
PAGE_SIZE = 50
CACHE_FILE = "data_fields_cache.json"
# Fields differ by region, universe and delay - keep this in step with the settings in alpha_loop.py's main().
SCOPE = {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1}
ALWAYS_INCLUDE = ("close", "volume", "returns")  # the basics stay in, even if usage ranks them lower
MAX_GROUP_FIELDS = 10  # grouping fields (sector, industry...) for group_* operators


def retry_after_seconds(resp, default=5.0):
    """BRAIN sends Retry-After as a float string such as "5.0"."""
    try:
        return min(max(float(resp.headers.get("Retry-After", default)), 1.0), 120.0)
    except (TypeError, ValueError):
        return default


def fetch_page(session, field_type, offset):
    params = {**SCOPE, "type": field_type, "limit": PAGE_SIZE, "offset": offset}
    for _ in range(6):
        resp = session.get(DATA_FIELDS_URL, params=params, timeout=30)
        if resp.status_code in (429, 503):
            wait = retry_after_seconds(resp)
            logger.warning("BRAIN asked us to slow down (%s) - waiting %.1fs.", resp.status_code, wait)
            time.sleep(wait)
            continue
        if be.looks_like_expired_session(resp):
            raise PermissionError("BRAIN session expired")
        if resp.status_code != 200:
            raise RuntimeError(f"BRAIN refused the field list ({resp.status_code}): {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"BRAIN's reply wasn't JSON ({resp.status_code}): {resp.text[:300]}")
        if not isinstance(data, dict) or "results" not in data:
            raise RuntimeError(f"Unexpected reply from BRAIN: {str(data)[:300]}")
        return data
    raise RuntimeError("BRAIN kept rate-limiting the field list. Try again in a few minutes.")


def fetch_all(session, field_type):
    first = fetch_page(session, field_type, 0)
    total = int(first.get("count") or len(first["results"]))
    fields = list(first["results"])
    logger.info("%s fields available to your account: %d (%d pages).",
                field_type, total, math.ceil(total / PAGE_SIZE) if total else 0)
    for page, offset in enumerate(range(PAGE_SIZE, total, PAGE_SIZE), start=2):
        time.sleep(0.3)  # be gentle - this is a few hundred requests at most
        fields.extend(fetch_page(session, field_type, offset)["results"])
        if page % 10 == 0:
            logger.info("  ... %d of %d", len(fields), total)
    # Keep only the type we asked for, in case BRAIN ever ignores the filter.
    return [f for f in fields if f.get("id") and f.get("type", field_type) == field_type]


def popularity(field):
    return int(field.get("userCount") or 0), int(field.get("alphaCount") or 0)


def describe(field):
    dataset = field.get("dataset") or {}
    source = dataset.get("name") or dataset.get("id") or ""
    text = (field.get("description") or "").strip()
    return f"{text} [{source}]" if source else text


def select_fields(matrix, groups, top, min_coverage):
    """Most-used fields first: popular fields are well understood and well covered."""
    usable = [f for f in matrix if (f.get("coverage") or 0) >= min_coverage]
    chosen = sorted(usable, key=popularity, reverse=True)[:top]
    names = {f["id"] for f in chosen}
    for basic in ALWAYS_INCLUDE:
        match = next((f for f in matrix if f["id"] == basic), None)
        if match and basic not in names:
            chosen.append(match)
            names.add(basic)
    entries = [{"name": f["id"], "description": describe(f)} for f in chosen]
    for g in sorted(groups, key=popularity, reverse=True)[:MAX_GROUP_FIELDS]:
        entries.append({"name": g["id"],
                        "description": f"group field - use as the group argument of group_* operators. {describe(g)}".strip()})
    return entries


def write_operators(entries, top, min_coverage, path=None):
    path = Path(path or al.OPERATORS_FILE)
    ref = json.loads(path.read_text(encoding="utf-8"))
    shutil.copyfile(path, path.with_name(path.name + ".bak"))
    ref["_fields_note"] = (
        f"Fetched from BRAIN on {datetime.now():%Y-%m-%d %H:%M} by fetch_fields.py for "
        f"{SCOPE['region']} / {SCOPE['universe']} / delay {SCOPE['delay']}: the {top} most-used fields with "
        f"coverage >= {min_coverage}, plus the basics and up to {MAX_GROUP_FIELDS} group fields. "
        f"Every name here is one BRAIN itself listed. Full list in {CACHE_FILE}.")
    ref["fields"] = entries
    path.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fetch_everything(session):
    matrix = fetch_all(session, "MATRIX")
    try:
        groups = fetch_all(session, "GROUP")
    except RuntimeError as exc:  # nice to have, not essential
        logger.warning("Couldn't fetch group fields (%s) - continuing without them.", exc)
        groups = []
    return matrix, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fill operators.json with the exact data fields BRAIN offers you.")
    parser.add_argument("--top", type=int, default=80, help="how many fields to give the generator (default 80)")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="skip fields with data for less than this share of stocks (default 0.8)")
    args = parser.parse_args(argv)

    email, password = be.get_credentials()
    session = al.get_authenticated_session(email, password)
    try:
        matrix, groups = fetch_everything(session)
    except PermissionError:
        logger.warning("BRAIN session expired partway through - logging in again and restarting the list.")
        session.cookies.clear()
        if not be.authenticate(session, email, password):
            logger.error("Couldn't log back in. operators.json is unchanged.")
            sys.exit(1)
        al.save_session(session)
        matrix, groups = fetch_everything(session)

    Path(CACHE_FILE).write_text(json.dumps(
        {"fetched_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "scope": SCOPE,
         "matrix_fields": matrix, "group_fields": groups}, ensure_ascii=False), encoding="utf-8")

    entries = select_fields(matrix, groups, args.top, args.min_coverage)
    if not entries:
        logger.error("BRAIN returned no usable fields. operators.json is unchanged.")
        sys.exit(1)
    write_operators(entries, args.top, args.min_coverage)
    logger.info("Done. operators.json now has %d fields BRAIN confirmed (old file saved as operators.json.bak).",
                len(entries))
    logger.info("First few: %s", ", ".join(e["name"] for e in entries[:12]))


if __name__ == "__main__":
    main()