"""
fetch_fields.py - replace the guessed data fields in operators.json with the exact fields
BRAIN offers your account, for the same settings alpha_loop.py simulates with.

Run from the alpha-miner folder (it uses the same .env and saved session as alpha_loop.py):
    python fetch_fields.py               # the 80 most-used fields with coverage >= 0.8
    python fetch_fields.py --least-used  # the 80 LEAST-used fields with coverage >= 0.8 instead: data few
                                         # other people have built alphas on, so yours look less like theirs
                                         # (the leaderboard's Uniqueness). At most 8 per dataset, so the
                                         # generator gets many kinds of data, and only fields at least 10
                                         # people have used - the very rarest are obscure analyst-guidance items
    python fetch_fields.py --top 120     # give the generator more fields
    python fetch_fields.py --skip 80     # the next 80 instead (ranks 81-160): for a second person mining
                                         # next to someone on the default, so you work on different data
                                         # and your alphas don't come out as near-copies of each other
    --per-dataset N and --min-users N change those two limits (0 turns either off), with or without --least-used.
    To go back, run it again without --least-used, or copy operators.json.bak over operators.json.

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
# --least-used: without a per-dataset limit, the rarest fields all come from one big dataset (on a USA TOP3000
# account, all 80 were analyst estimates); and the very rarest - used by a handful of people - are obscure
# items such as one kind of guidance figure, so fields fewer than MIN_USERS people have used are left out.
MAX_PER_DATASET = 8
MIN_USERS = 10


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


def usable_fields(matrix, min_coverage, min_users=0):
    return [f for f in matrix if (f.get("coverage") or 0) >= min_coverage and popularity(f)[0] >= min_users]


def least_used_first(fields):
    """Fewest users first, then fewest alphas; of two equally rare fields, the better-covered one first."""
    return sorted(fields, key=lambda f: (*popularity(f), -(f.get("coverage") or 0), str(f.get("id") or "")))


def dataset_of(field) -> str:
    dataset = field.get("dataset") or {}
    if isinstance(dataset, dict):
        return str(dataset.get("id") or dataset.get("name") or "")
    return str(dataset)


def spread_over_datasets(ordered, top, per_dataset):
    """The first `top` fields of `ordered`, at most `per_dataset` from any one dataset (0: no limit). If there
    are too few datasets to fill the selection that way, the fields held back fill the rest, in order."""
    if per_dataset <= 0:
        return ordered[:top]
    taken, held, count = [], [], {}
    for f in ordered:
        if len(taken) == top:
            break
        dataset = dataset_of(f)
        if count.get(dataset, 0) < per_dataset:
            taken.append(f)
            count[dataset] = count.get(dataset, 0) + 1
        else:
            held.append(f)
    return taken + held[:top - len(taken)]


def select_fields(matrix, groups, top, min_coverage, skip=0, least_used=False, per_dataset=0, min_users=0):
    """Most-used fields first: popular fields are well understood and well covered. skip leaves out that
    many of the most-used ones first, so the selection is the fields ranked skip+1 to skip+top.
    least_used turns the order round: the fields fewest people use first (skip then leaves out that many of
    the least-used). per_dataset caps how many come from one dataset; min_users leaves out fields fewer
    people than that have used."""
    usable = usable_fields(matrix, min_coverage, min_users)
    ordered = least_used_first(usable) if least_used else sorted(usable, key=popularity, reverse=True)
    chosen = spread_over_datasets(ordered[skip:], top, per_dataset)
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


def selection_note(top, skip=0, least_used=False, per_dataset=0, min_users=0) -> str:
    """How the fields were picked, in words, for operators.json's _fields_note."""
    if least_used:
        which = f"the {top} least-used fields" + (f" after the {skip} least-used" if skip else "")
    elif skip:
        which = (f"the fields ranked {skip + 1}-{skip + top} by use (the {skip} most-used left out, so this miner "
                 f"works on different data from one using the default)")
    else:
        which = f"the {top} most-used fields"
    if per_dataset > 0:
        which += f", at most {per_dataset} from any one dataset"
    if min_users > 0:
        which += f", each used by at least {min_users} people"
    return which


def write_operators(entries, top, min_coverage, path=None, skip=0, which=None):
    path = Path(path or al.OPERATORS_FILE)
    ref = json.loads(path.read_text(encoding="utf-8"))
    shutil.copyfile(path, path.with_name(path.name + ".bak"))
    which = which or selection_note(top, skip)
    ref["_fields_note"] = (
        f"Fetched from BRAIN on {datetime.now():%Y-%m-%d %H:%M} by fetch_fields.py for "
        f"{SCOPE['region']} / {SCOPE['universe']} / delay {SCOPE['delay']}: {which}, with "
        f"coverage >= {min_coverage}, plus the basics and up to {MAX_GROUP_FIELDS} group fields. "
        f"Every name here is one BRAIN itself listed. Full list in {CACHE_FILE}.")
    ref["fields"] = entries
    path.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def current_field_names(path=None) -> list:
    """The field names operators.json lists now - [] if it can't be read."""
    try:
        ref = json.loads(Path(path or al.OPERATORS_FILE).read_text(encoding="utf-8"))
        return [f.get("name") for f in ref.get("fields") or [] if isinstance(f, dict)]
    except (OSError, ValueError, AttributeError):
        return []


def usage_summary(names, matrix):
    """(fewest users, most users, datasets) over these fields, leaving out the basics - None if BRAIN listed none."""
    by_id = {f.get("id"): f for f in matrix}
    found = [by_id[n] for n in names if n in by_id and n not in ALWAYS_INCLUDE]
    if not found:
        return None
    users = [popularity(f)[0] for f in found]
    return min(users), max(users), len({dataset_of(f) for f in found})


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
    parser.add_argument("--skip", type=int, default=0, metavar="N",
                        help="leave out the N most-used fields and take the next ones - e.g. --skip 80 for a "
                             "second person mining next to someone on the default 80, so you work on different data")
    parser.add_argument("--least-used", action="store_true",
                        help="take the fields fewest people use instead of the most-used, so your alphas look less "
                             "like everyone else's (with --skip N, the N least-used are left out)")
    parser.add_argument("--per-dataset", type=int, default=None, metavar="N",
                        help=f"at most N fields from any one dataset (default {MAX_PER_DATASET} with --least-used, "
                             f"no limit otherwise; 0 = no limit)")
    parser.add_argument("--min-users", type=int, default=None, metavar="N",
                        help=f"leave out fields fewer than N people have used (default {MIN_USERS} with --least-used, "
                             f"0 otherwise)")
    args = parser.parse_args(argv)
    per_dataset = args.per_dataset if args.per_dataset is not None else (MAX_PER_DATASET if args.least_used else 0)
    min_users = args.min_users if args.min_users is not None else (MIN_USERS if args.least_used else 0)

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

    usable = len(usable_fields(matrix, args.min_coverage, min_users))
    if args.skip > 0 and usable <= args.skip:
        logger.error("Only %d fields have coverage >= %s%s on this account, so --skip %d leaves none. "
                     "operators.json is unchanged - try a smaller --skip.", usable, args.min_coverage,
                     f" and {min_users}+ users" if min_users > 0 else "", args.skip)
        sys.exit(1)
    before = usage_summary(current_field_names(), matrix)
    entries = select_fields(matrix, groups, args.top, args.min_coverage, args.skip,
                            least_used=args.least_used, per_dataset=per_dataset, min_users=min_users)
    if not entries:
        logger.error("BRAIN returned no usable fields. operators.json is unchanged.")
        sys.exit(1)
    write_operators(entries, args.top, args.min_coverage, skip=args.skip,
                    which=selection_note(args.top, args.skip, args.least_used, per_dataset, min_users))
    if args.skip > 0 and args.least_used:
        logger.info("Left out the %d least-used fields.", args.skip)
    elif args.skip > 0:
        logger.info("Left out the %d most-used fields: these are the ones ranked %d-%d.", args.skip,
                    args.skip + 1, min(usable, args.skip + args.top))
    logger.info("Done. operators.json now has %d fields BRAIN confirmed (old file saved as operators.json.bak).",
                len(entries))
    after = usage_summary([e["name"] for e in entries if not e["description"].startswith("group field")], matrix)
    if after:
        logger.info("The new fields come from %d dataset(s), and %s-%s people have used each of them%s.",
                    after[2], f"{after[0]:,}", f"{after[1]:,}",
                    f" (the previous list: {before[0]:,}-{before[1]:,})" if before else "")
    logger.info("First few: %s", ", ".join(e["name"] for e in entries[:12]))


if __name__ == "__main__":
    main()
