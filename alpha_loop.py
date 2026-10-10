"""
alpha_loop.py — autonomous generate -> test -> learn -> repeat loop for WorldQuant BRAIN,
now with optional autonomous submission of alphas that clear a quality bar.

Setup (once), same folder as brain_engine.py:
    pip install requests python-dotenv google-genai pydantic
    .env needs three lines:
        WQ_EMAIL=you@example.com
        WQ_PASSWORD=your-password
        GEMINI_API_KEY=...

Run:
    python alpha_loop.py                      # Gemini writes the ideas
    python alpha_loop.py --try "<expression>" "<another>"
                                              # test your own expressions through the same
                                              # pipeline: simulate, check, submit if they qualify
    python alpha_loop.py --log-dir logs       # also keep the log in logs/alpha_loop_YYYY-MM-DD.log
                                              # (what the daily scheduled run uses)

Only one run at a time: a second run started while one is going exits straight away (alpha_loop.lock),
so a scheduled run and a manual one can never double-submit.

NOTE on the LLM library: this uses google-genai (import as `from google import genai`),
NOT google-generativeai (`import google.generativeai as genai`) - that package was
deprecated in August 2025 and is now archived/legacy. If you see import errors or silent
generation failures with the old package, that's why. Model defaults to gemini-3.8-flash
(current as of Sept 2026) - Google ships new Flash versions every few weeks, so this is
overridable via GEMINI_MODEL without touching code.

When Gemini is busy (503) or rate-limited (429), idea generation retries with backoff
(10s, 20s, 40s, 80s), then tries GEMINI_FALLBACK_MODEL once (default gemini-3.7-flash,
which Google's own 3.8 Flash docs name as the fully supported alternative). A failed
generation does NOT use up one of the MAX_ROUNDS - rounds only count when ideas actually
get tested. After MAX_GENERATION_FAILURES failed attempts in a row, the run stops cleanly;
nothing was simulated in that case, so no BRAIN quota was spent.

READ BEFORE YOU RUN THIS:
- It spends real BRAIN simulation quota and real Gemini API cost, unattended, up to the
  caps below (defaults: 5 rounds x 8 ideas = 40 simulations max per run). Raise
  IDEAS_PER_ROUND / MAX_ROUNDS via env vars once you trust it, not before.
- This version can AUTONOMOUSLY SUBMIT alphas to BRAIN, at most MAX_SUBMISSIONS_PER_DAY
  (default 1) per BRAIN day, which starts at 3 AM US Eastern, not local midnight. An alpha
  qualifies when Sharpe >= 1.25, fitness >= 1.0 and every check passes, except SELF_CORRELATION
  while it reads PENDING: BRAIN only works that one out during submission, so it shows PENDING
  on every unsubmitted alpha. Before submitting, the loop asks BRAIN's free pre-submit
  self-correlation check and skips anything at 0.7 or above, so the daily slot isn't wasted.
  After submitting, it waits for BRAIN's verdict and only logs "YES" once BRAIN lists the
  alpha as ACTIVE. Anything unclear is logged as VERDICT_PENDING - confirm those with the
  dashboard's Check with BRAIN button. The submit and correlation endpoints follow community
  clients that recorded BRAIN's live traffic; BRAIN publishes no official API docs.
- Held alphas: one that clears the bar after today's submission is used is first run through BRAIN's
  free self-correlation check - too similar to your submitted alphas, it's logged TOO_CORRELATED;
  otherwise it's held as QUALIFIED_NOT_SUBMITTED. Every run (--try included) starts by re-checking
  every held alpha, slot free or not, since similarity grows as you submit more: one BRAIN already
  lists as submitted is marked YES, one where a check now FAILs is marked NOT_SUBMITTED, one that's
  become too similar is marked TOO_CORRELATED. If today's slot is free, the best remaining one
  (highest fitness) is submitted. If BRAIN can't be reached or the submit itself fails, the alpha stays
  queued for the next run. Re-checking stops after HELD_CHECK_BUDGET_SECONDS (default 300) per run.
- Auto-flip: when an alpha scores Sharpe -1.0 or worse, the loop also tests reverse(<expression>)
  right after it (up to MAX_FLIPS_PER_ROUND per round, default 3), since a strongly negative signal
  usually turns positive when flipped. Flips use extra simulations.
- Auto-smooth: when an alpha has Sharpe >= 1.25 but fitness under 1.0 and turnover above
  SMOOTH_MIN_TURNOVER (default 0.2) - or turnover over BRAIN's 70% limit with Sharpe >= 1.0
  (SMOOTH_MIN_SHARPE_OVER_LIMIT) - the loop also tests
  ts_decay_linear(<expression>, 10) right after it (up to MAX_SMOOTHS_PER_ROUND per round, default 3).
  Fitness divides by turnover, so trading less can lift it over the bar.
- Auto-repair: an alpha with Sharpe and fitness over the bar that FAILs one of BRAIN's fixable checks
  gets up to MAX_REPAIRS_PER_ALPHA (default 2) repaired versions tested right after it, at most
  MAX_REPAIRS_PER_ROUND (default 4) per round: CONCENTRATED_WEIGHT -> rank(...) or ts_backfill on its data
  fields; LOW_SUB_UNIVERSE_SHARPE -> rank within subindustry, or neutralize against market-cap buckets;
  LOW_TURNOVER -> drop the outer ts_decay_linear. Repairs follow community write-ups of BRAIN's checks,
  and BRAIN's own result decides whether each worked. If the original is already too similar to your
  submitted alphas, it isn't repaired (the repair would be too). A near-miss failing a check with no
  automatic fix gets a log line naming that check.
- Auto-sharpen: an alpha with Sharpe >= 1.25 but fitness under 1.0 whose turnover is already under 12.5%
  - where BRAIN's fitness formula stops giving credit for trading less - is re-tested without its outer
  ts_decay_linear/hump (up to MAX_SHARPENS_PER_ROUND, default 3), after the same free similarity check.
- Each normal run starts by re-examining up to MAX_PAST_REPAIRS (default 8) earlier near-misses whose
  repairs or sharpening haven't been tried.
- Learning what doesn't work: any operator or field BRAIN rejects ('unknown variable/operator') is
  read back from the results file, removed from what the generator sees, and ideas still using it are
  dropped before they cost a simulation. Fields tried DEAD_END_MIN_TRIES (default 8) times without
  fitness reaching DEAD_END_MAX_FITNESS (default 0.5) are left out the same way; basic price/volume
  fields (close, volume, returns...) are exempt.
- Variety: alphas built on the same kind of data usually come back too correlated to submit. Each
  field is labelled with its data type - BRAIN's own category from data_fields_cache.json (written by
  fetch_fields.py), or the dataset name in operators.json without it. The generator is told which types
  your submitted or queued alphas already use, and asked for VARIETY_EXTRA_IDEAS (default 4) spare
  ideas; the loop then keeps at most MAX_IDEAS_PER_TYPE (default 2) per type per round, and at most
  MAX_IDEAS_PER_CROWDED_TYPE (default 1) for types you've already submitted.
- Formula shape: each expression is reduced to its skeleton (smoothing wrapped around the outside,
  fields, numbers and groups stripped, any ts_*/group_* operator treated alike). Shapes your submitted, queued or too-similar alphas use - or that
  make up 30%+ of your last 60 simulations - count as overused: the generator is shown them with a list of
  other structures from your operator list, and the loop keeps at most MAX_IDEAS_PER_CROWDED_SHAPE
  (default 2) ideas per overused shape per round.
- Parallel simulations: up to PARALLEL_SIMULATIONS (default 3) run at once. If BRAIN answers that too
  many are running, the loop carries on with however many it already has going. Results are still
  handled one at a time, so submissions never overlap. Set PARALLEL_SIMULATIONS=1 for the old pace.
- Network resilience: every request to BRAIN has a 30-second timeout and is retried after a dropped
  connection (waiting 5s, 15s, 45s). If an idea still can't get through, the loop logs it as
  "Couldn't reach BRAIN" and moves on to the next one; after MAX_NETWORK_FAILURES (default 3) in a
  row, it stops cleanly, keeping everything so far. Gemini connection drops are retried the same way.
- Memory across runs: every round, the generator sees your best MEMORY_TOP_N (default 10) results
  from all runs so far (read from alpha_loop_results.csv), not just the last round.
- operators.json holds the full base-tier operator list (66 operators) transcribed from
  your BRAIN account. Its "fields" list is still the small 8-field starter set - add the
  data fields your account can use to widen what the generator can propose.
- "Unattended" has a real ceiling: BRAIN sessions expire (roughly 4 hours) and can
  require a one-time identity check to renew. This script saves its session to disk
  so a re-run doesn't need to log in again unnecessarily, but if BRAIN asks for
  verification mid-run with nobody there to click the link, it STOPS CLEANLY and
  logs why. It does not hang waiting for input that will never come.
"""
import argparse
import csv
import json
import logging
import math
import os
import re
import time
from collections import Counter, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx  # the HTTP library google-genai uses; its network errors are worth retrying
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

import brain_engine as be  # reuses: get_credentials, authenticate, submit_simulation,
                            # poll_simulation, fetch_alpha_metrics, looks_like_expired_session

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Guardrails: real quota and real money per round. Keep these low until you trust it. ----
IDEAS_PER_ROUND = int(os.environ.get("IDEAS_PER_ROUND", 8))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 5))
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 4 * 60 * 60))  # ~ BRAIN's own session lifetime
MAX_SUBMISSIONS_PER_DAY = int(os.environ.get("MAX_SUBMISSIONS_PER_DAY", 1))  # BRAIN's own submitter tools cap at 1/day

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Google's own Gemini 3.8 Flash docs name 3.7 Flash as the fully supported alternative.
# Set GEMINI_FALLBACK_MODEL to an empty value in .env to turn the fallback off.
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")
RETRY_DELAYS_SECONDS = (10, 20, 40, 80)  # backoff when the main model is busy or rate-limited
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}  # worth waiting out; anything else won't fix itself
MAX_GENERATION_FAILURES = int(os.environ.get("MAX_GENERATION_FAILURES", 3))  # in a row, then stop the run
GENERATION_COOLDOWN_SECONDS = int(os.environ.get("GENERATION_COOLDOWN_SECONDS", 60))  # pause between failed attempts

# Submission bar - unchanged values, just named once so dashboard.py reads the same numbers.
SUBMIT_MIN_SHARPE = 1.25
SUBMIT_MIN_FITNESS = 1.0
# BRAIN's accepted turnover range: 1% to 70% of the book traded per day (reported as 0.01 to 0.70).
TURNOVER_MIN, TURNOVER_MAX = 0.01, 0.70
MAX_SELF_CORRELATION = 0.7  # BRAIN rejects a submission this correlated with your already-submitted alphas
CORR_POLL_MAX_SECONDS = int(os.environ.get("CORR_POLL_MAX_SECONDS", 300))
SUBMIT_POLL_MAX_SECONDS = int(os.environ.get("SUBMIT_POLL_MAX_SECONDS", 600))
CORRELATION_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/self"
SUBMIT_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}/submit"

# Auto-flip: a strongly negative Sharpe is a real signal pointing the wrong way, so test its reverse.
FLIP_SHARPE_AT_OR_BELOW = float(os.environ.get("FLIP_SHARPE_AT_OR_BELOW", -1.0))
MAX_FLIPS_PER_ROUND = int(os.environ.get("MAX_FLIPS_PER_ROUND", 3))  # caps the extra simulations flips add

# Auto-smooth: a good Sharpe held back by trading too much gets re-tested with linear decay.
SMOOTH_DECAY_DAYS = int(os.environ.get("SMOOTH_DECAY_DAYS", 10))
SMOOTH_MIN_TURNOVER = float(os.environ.get("SMOOTH_MIN_TURNOVER", 0.2))  # below this, decay can barely help fitness
SMOOTH_MIN_SHARPE_OVER_LIMIT = float(os.environ.get("SMOOTH_MIN_SHARPE_OVER_LIMIT", 1.0))  # turnover over 70%
MAX_SMOOTHS_PER_ROUND = int(os.environ.get("MAX_SMOOTHS_PER_ROUND", 3))  # caps the extra simulations smoothing adds

# Network resilience: one dropped connection skips that idea; this many in a row stops the run cleanly.
MAX_NETWORK_FAILURES = int(os.environ.get("MAX_NETWORK_FAILURES", 3))

# Auto-repair: a strong alpha (Sharpe and fitness over the bar) failing a fixable check gets
# re-tested with a known fix for that check. Each repair is one extra simulation.
MAX_REPAIRS_PER_ALPHA = int(os.environ.get("MAX_REPAIRS_PER_ALPHA", 2))
MAX_REPAIRS_PER_ROUND = int(os.environ.get("MAX_REPAIRS_PER_ROUND", 4))
MAX_PAST_REPAIRS = int(os.environ.get("MAX_PAST_REPAIRS", 8))  # earlier near-misses re-examined per run
# Auto-sharpen: below this turnover BRAIN's fitness formula gets no extra credit, so smoothing an alpha
# that's already this slow only dampens its returns.
FITNESS_TURNOVER_FLOOR = 0.125
MAX_SHARPENS_PER_ROUND = int(os.environ.get("MAX_SHARPENS_PER_ROUND", 3))
BACKFILL_DAYS = 120

# Learning what doesn't work: names BRAIN rejected, and fields that never got close.
DEAD_END_MIN_TRIES = int(os.environ.get("DEAD_END_MIN_TRIES", 8))
DEAD_END_MAX_FITNESS = float(os.environ.get("DEAD_END_MAX_FITNESS", 0.5))
BASIC_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap")  # never dropped

# Variety: alphas built on the same kind of data tend to be too correlated to submit together.
FIELDS_CACHE_FILE = "data_fields_cache.json"  # written by fetch_fields.py - holds BRAIN's category per field
MAX_IDEAS_PER_TYPE = int(os.environ.get("MAX_IDEAS_PER_TYPE", 2))  # per data type, per round
MAX_IDEAS_PER_CROWDED_TYPE = int(os.environ.get("MAX_IDEAS_PER_CROWDED_TYPE", 1))  # types you've already submitted
VARIETY_EXTRA_IDEAS = int(os.environ.get("VARIETY_EXTRA_IDEAS", 6))  # ask for spares, keep the most varied
# Formula shape: the operators nested the same way, whatever the fields, windows or groups.
MAX_IDEAS_PER_CROWDED_SHAPE = int(os.environ.get("MAX_IDEAS_PER_CROWDED_SHAPE", 2))  # per round
SHAPE_LOOKBACK = 60          # recent simulations looked at to spot an overused shape
SHAPE_CROWDED_SHARE = 0.3    # a shape making up this share of them counts as overused

# Parallel simulations: BRAIN runs a few of yours at once. If it says it's busy (429), the loop
# drops to however many were already running and keeps going.
PARALLEL_SIMULATIONS = max(1, int(os.environ.get("PARALLEL_SIMULATIONS", 3)))
SIM_CHECK_SECONDS = 5            # how often running simulations are checked
BUSY_WAIT_SECONDS = 15           # BRAIN busy and nothing of ours running: wait this long, then retry
MAX_BUSY_WAIT_SECONDS = 10 * 60  # ...for at most this long before giving up on that idea

# Memory across runs: how many of your best past results the generator sees each round.
MEMORY_TOP_N = int(os.environ.get("MEMORY_TOP_N", 10))

SETTINGS = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
    "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "ON",
    "language": "FASTEXPR", "visualization": False,
}

OPERATORS_FILE = "operators.json"
RESULTS_FILE = "alpha_loop_results.csv"  # separate from brain_engine.py's own results.csv
SESSION_FILE = "session_cookies.json"
SESSION_MIN_SECONDS = 30 * 60   # reuse a saved BRAIN login only if it has at least this long left
SESSION_RENEW_SECONDS = 5 * 60  # during a run, log in again this long before the login runs out
SUBMISSION_LOG_FILE = "submissions_log.json"  # {"YYYY-MM-DD": count} - enforces the daily cap across runs
HELD_PREFIX = "QUALIFIED_NOT_SUBMITTED"
HELD_LABEL = f"{HELD_PREFIX} (daily cap reached)"  # cleared the bar, waiting for a free submission slot
REPLACE_ATTEMPTS = 5  # Windows blocks swapping in the updated CSV while another program is reading it
HELD_CHECK_BUDGET_SECONDS = int(os.environ.get("HELD_CHECK_BUDGET_SECONDS", 300))  # per run, for re-checking held alphas

RESULTS_HEADER = [
    "round", "timestamp", "expression", "rationale", "status", "sharpe", "fitness",
    "turnover", "checks_passed", "checks_total", "alpha_id", "alpha_link", "error", "submitted",
]


class StopLoop(Exception):
    """Raised to unwind out of a round immediately - e.g. session can't be renewed unattended."""


class LoginFailed(RuntimeError):
    """Couldn't log in to BRAIN unattended (wrong password, or BRAIN wants identity verification)."""


class GenerationFailed(Exception):
    """Raised when no ideas could be generated this attempt (main model and fallback both
    failed). The main loop catches it and does NOT count the attempt as a round."""


# ---- Gemini structured-output schema (Pydantic - this is what the current SDK's docs use;
# the old TypedDict-based approach was tied to the deprecated library) ----
class Idea(BaseModel):
    expression: str
    rationale: str


class AlphaIdeas(BaseModel):
    ideas: list[Idea]


def load_reference():
    with open(OPERATORS_FILE, encoding="utf-8") as f:
        ref = json.load(f)
    return ref["operators"], ref["fields"]


def save_session(session: requests.Session):
    Path(SESSION_FILE).write_text(
        json.dumps(requests.utils.dict_from_cookiejar(session.cookies)), encoding="utf-8"
    )


def load_session(session: requests.Session) -> bool:
    if not Path(SESSION_FILE).exists():
        return False
    try:
        cookies = json.loads(Path(SESSION_FILE).read_text(encoding="utf-8"))
        session.cookies.update(requests.utils.cookiejar_from_dict(cookies))
        return True
    except Exception:
        return False


def session_seconds_left(session):
    """How long BRAIN says this login has left: GET /authentication -> token.expiry, the same check
    WorldQuant's own ACE library uses. 0 if BRAIN says it isn't logged in, None if it can't tell.
    (Asking about an alpha can't tell: logged out, BRAIN may answer 'not found' instead.)"""
    try:
        resp = session.get(be.AUTH_URL, timeout=be.REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code in (401, 403) or be.looks_like_expired_session(resp):
        return 0.0
    if resp.status_code != 200:
        return None
    try:
        return max(0.0, float(resp.json()["token"]["expiry"]))
    except (ValueError, KeyError, TypeError):
        return None


def note_expiry(session):
    """Remember when this login runs out, so a long run can log in again before it does."""
    left = session_seconds_left(session)
    session.brain_expires_at = (time.time() + left) if left else None


def log_in(session, email, password) -> bool:
    """be.authenticate, made safe for runs nobody is watching: if BRAIN asks for identity verification
    and there's no keyboard to answer it, this returns False with a clear message instead of crashing."""
    try:
        ok = be.authenticate(session, email, password)
    except (EOFError, OSError, RuntimeError):
        logger.error("BRAIN wants you to confirm your identity, and nobody is here to do it. Run "
                     "python alpha_loop.py by hand once and open the link it prints.")
        return False
    if ok:
        save_session(session)
        note_expiry(session)
    return ok


def get_authenticated_session(email: str, password: str) -> requests.Session:
    session = requests.Session()
    if load_session(session):
        left = session_seconds_left(session)
        if left is not None and left >= SESSION_MIN_SECONDS:
            session.brain_expires_at = time.time() + left
            logger.info("Reused saved session (%d minutes left on it) - skipped login.", left // 60)
            return session
        logger.info("Saved session %s - logging in fresh.",
                    "couldn't be confirmed" if left is None else "has expired" if not left
                    else f"has under {SESSION_MIN_SECONDS // 60} minutes left")
        session.cookies.clear()

    if not log_in(session, email, password):
        raise LoginFailed("Could not log in to BRAIN - stopping before using any quota.")
    return session


def keep_logged_in(session, email, password):
    """Log in again shortly before BRAIN's login runs out, instead of waiting for requests to fail."""
    expires_at = getattr(session, "brain_expires_at", None)
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return
    if time.time() < expires_at - SESSION_RENEW_SECONDS:
        return
    logger.info("BRAIN login runs out in under %d minutes - logging in again now.", SESSION_RENEW_SECONDS // 60)
    session.cookies.clear()
    if not log_in(session, email, password):
        raise StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to complete it.")


def already_tried() -> set:
    if not Path(RESULTS_FILE).exists():
        return set()
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return {row["expression"] for row in csv.DictReader(f)}


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def brain_day(now_utc=None) -> date:
    """BRAIN's day starts at 3 AM US Eastern, not at your local midnight. Worked out from the US
    daylight-saving rules directly, so it needs no time-zone database (Windows doesn't ship one)."""
    now_utc = now_utc or datetime.now(timezone.utc)

    def nth_sunday(month, n):
        first_weekday = date(now_utc.year, month, 1).weekday()  # Monday=0 ... Sunday=6
        return 1 + (6 - first_weekday) % 7 + 7 * (n - 1)

    dst_start = datetime(now_utc.year, 3, nth_sunday(3, 2), 7, tzinfo=timezone.utc)   # 2 AM EST
    dst_end = datetime(now_utc.year, 11, nth_sunday(11, 1), 6, tzinfo=timezone.utc)   # 2 AM EDT
    eastern = timedelta(hours=-4) if dst_start <= now_utc < dst_end else timedelta(hours=-5)
    return (now_utc + eastern - timedelta(hours=3)).date()


def submissions_today() -> int:
    if not Path(SUBMISSION_LOG_FILE).exists():
        return 0
    try:
        data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
    except Exception:
        return 0
    return data.get(str(brain_day()), 0)


def record_submission():
    data = {}
    if Path(SUBMISSION_LOG_FILE).exists():
        try:
            data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
        except Exception:
            data = {}
    key = str(brain_day())
    data[key] = data.get(key, 0) + 1
    Path(SUBMISSION_LOG_FILE).write_text(json.dumps(data), encoding="utf-8")


def as_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def turnover_note(row) -> str:
    turnover = as_float(row.get("turnover"))
    if turnover is None:
        return ""
    if turnover > TURNOVER_MAX:
        return f" TURNOVER TOO HIGH (max {TURNOVER_MAX})"
    if turnover < TURNOVER_MIN:
        return f" TURNOVER TOO LOW (min {TURNOVER_MIN})"
    return ""


def summarize_round(rows: list) -> str:
    """Condense one round's results into text the next generation prompt can use."""
    if not rows:
        return "This is round 1 - no prior results yet."
    completed = [r for r in rows if r["status"] == "COMPLETE"]
    if not completed:
        return f"Last round: {len(rows)} attempts, none completed - check the error column pattern before trying similar shapes again."
    ranked = sorted(completed, key=lambda r: safe_float(r["fitness"]), reverse=True)
    lines = ["Last round's results, best fitness first:"]
    for r in ranked:
        lines.append(
            f"- {r['expression']} -> sharpe={r['sharpe']} fitness={r['fitness']} "
            f"turnover={r['turnover']}{turnover_note(r)} checks={r['checks_passed']}/{r['checks_total']} "
            f"submitted={r.get('submitted', 'NO')}"
            + (f" not_passing=[{r['failing_checks']}]" if r.get("failing_checks") else "")
        )
    failed = [r for r in rows if r["status"] != "COMPLETE"]
    if failed:
        lines.append(f"{len(failed)} other attempts failed outright (bad expression, timeout, etc) - avoid whatever pattern caused that.")
    return "\n".join(lines)


def completed_history() -> list:
    """Every completed simulation from every run so far, read from the results CSV."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r.get("status") == "COMPLETE" and as_float(r.get("fitness")) is not None]


def pool_tag(row) -> str:
    """Marks results that are already in (or headed for) your submitted pool."""
    submitted = (row.get("submitted") or "").strip()
    if submitted == "YES" or submitted.startswith("VERDICT_PENDING"):
        return " [submitted]"
    if submitted.startswith(HELD_PREFIX):
        return " [queued]"
    if submitted.startswith("TOO_CORRELATED"):
        return " [too similar to a submitted alpha]"
    return ""


# ---- Learning what doesn't work ----

_REJECTED = re.compile(r'unknown (?:variable|operator) "([^"]+)"', re.IGNORECASE)


def rejected_names() -> set:
    """Operators and fields BRAIN refused on your account ('unknown variable/operator "x"'), read
    from the error column of every run so far."""
    if not Path(RESULTS_FILE).exists():
        return set()
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        return {name for r in csv.DictReader(f) for name in _REJECTED.findall(r.get("error") or "")}


def dead_end_fields() -> dict:
    """{field: (simulations, best fitness)} for fields tried DEAD_END_MIN_TRIES+ times without fitness
    ever reaching DEAD_END_MAX_FITNESS. Basic price/volume fields are exempt - they're ingredients."""
    stats = {}
    for r in completed_history():
        fitness = as_float(r.get("fitness"))
        for name in {n for _, _, n in data_fields_in(r["expression"])}:
            tries, best = stats.get(name, (0, fitness))
            stats[name] = (tries + 1, max(best, fitness))
    return {name: (tries, best) for name, (tries, best) in stats.items()
            if name not in BASIC_FIELDS and tries >= DEAD_END_MIN_TRIES and best < DEAD_END_MAX_FITNESS}


def uses_any(expression: str, names) -> bool:
    return bool(set(_WORD.findall(_blank_quotes(expression))) & set(names))


def avoid_note(rejected, dead: dict) -> str:
    lines = []
    if rejected:
        lines.append("BRAIN rejected these names on your account - never use them: "
                     + ", ".join(sorted(rejected)) + ". (Write comparisons with >, <, == instead.)")
    if dead:
        lines.append(f"These fields were tried {DEAD_END_MIN_TRIES}+ times and never reached fitness "
                     f"{DEAD_END_MAX_FITNESS}, so they're left out: "
                     + ", ".join(f"{n} ({t} tries, best {b:.2f})" for n, (t, b) in sorted(dead.items())) + ".")
    return ("DON'T USE:\n" + "\n".join(f"- {line}" for line in lines)) if lines else ""


# ---- Variety: spread ideas across data types ----

def field_types(fields) -> dict:
    """{field name: data type} for the signal fields the generator can use. The data type is BRAIN's
    own category for the field (Option, Fundamental, Analyst...), read from data_fields_cache.json;
    without that file, the dataset name fetch_fields.py keeps in [brackets] in each description.
    Group fields (sector, industry...) are left out - every idea uses them for grouping."""
    categories = {}
    try:
        cache = json.loads(Path(FIELDS_CACHE_FILE).read_text(encoding="utf-8"))
        for f in cache.get("matrix_fields") or []:
            cat = f.get("category")
            name = (cat.get("name") or cat.get("id")) if isinstance(cat, dict) else cat
            if f.get("id") and name:
                categories[f["id"]] = str(name)
    except (OSError, ValueError, AttributeError):
        pass
    found = {}
    for f in fields:
        name, desc = f.get("name"), (f.get("description") or "").strip()
        if not name or desc.startswith("group field"):
            continue
        if name in categories:
            found[name] = categories[name]
        else:
            tag = re.search(r"\[([^\[\]]+)\]$", desc)
            if tag:
                found[name] = tag.group(1)
    return found


def expression_types(expression: str, type_map: dict) -> set:
    return {type_map[word] for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression) if word in type_map}


def crowded_types(type_map: dict) -> Counter:
    """Data types your submitted, queued or too-similar alphas are built on, with how many of each."""
    counts = Counter()
    if not type_map or not Path(RESULTS_FILE).exists():
        return counts
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if pool_tag(r):
                counts.update(expression_types(r.get("expression") or "", type_map))
    return counts


def per_type_cap(n: int, type_map: dict) -> int:
    """Ideas allowed per data type - loosened if there are too few types to fill a round otherwise."""
    return max(MAX_IDEAS_PER_TYPE, math.ceil(n / max(1, len(set(type_map.values())))))


def variety_rule(type_map: dict, crowded: Counter, n: int) -> str:
    all_types = sorted(set(type_map.values()))
    if len(all_types) < 2:
        return ""
    lines = ["VARIETY RULE - this is what gets new alphas accepted. BRAIN won't take an alpha whose "
             "self-correlation with your submitted alphas is 0.7 or more, and alphas built on the same "
             "kind of data usually are. Every field above is labelled with its data type."]
    if crowded:
        used = ", ".join(f"{t} ({c})" for t, c in crowded.most_common())
        lines.append(f"- Your submitted or queued alphas already use: {used}. At most "
                     f"{MAX_IDEAS_PER_CROWDED_TYPE} idea(s) this round may use each of these.")
    fresh = [t for t in all_types if t not in crowded]
    if fresh:
        lines.append(f"- Not used yet - favour these: {', '.join(fresh)}.")
    lines.append(f"- At most {per_type_cap(n, type_map)} ideas per data type. Carry the shapes that scored well "
                 f"(operators, windows, grouping, decay) over to other data types instead of repeating the same data.")
    return "\n".join(lines)


# ---- Variety of formula shape ----

SMOOTHERS = ("ts_decay_linear", "hump")


def strip_smoothing(expression: str) -> str:
    """The expression without any ts_decay_linear / hump wrapped around the outside."""
    expr = expression.strip()
    while True:
        call = outer_call(expr)
        if not call or call[0] not in SMOOTHERS or not call[1][0].strip():
            return expr
        expr = call[1][0].strip()


def skeleton(expression: str) -> str:
    """The formula's shape with the specifics stripped out: smoothing wrapped around the outside
    (ts_decay_linear, hump) is ignored - it controls turnover, it isn't the idea - then data fields -> F,
    groups -> G, numbers -> N, any ts_* operator -> ts_*, any group_* operator -> group_*. Two alphas with
    the same skeleton are the same idea with different data, windows or smoothing."""
    expression = strip_smoothing(expression)
    group_spans = list(_group_calls(expression))
    cuts = [(s, e, "G") for s, e in group_spans]
    cuts += [(s, e, "F") for s, e, _ in data_fields_in(expression)
             if not any(gs <= s and e <= ge for gs, ge in group_spans)]
    out = expression
    for s, e, token in sorted(cuts, reverse=True):
        out = out[:s] + token + out[e:]
    out = _QUOTED.sub("Q", out)
    out = re.sub(r"(?<![A-Za-z_])\d+(\.\d+)?([eE]-?\d+)?", "N", out)
    out = re.sub(r"\bts_\w+", "ts_*", out)
    out = re.sub(r"\bgroup_\w+", "group_*", out)
    return re.sub(r"\s+", "", out)


def crowded_shapes() -> dict:
    """{skeleton: example expression} for shapes your submitted, queued or too-similar alphas use, plus
    any shape that makes up SHAPE_CROWDED_SHARE or more of your last SHAPE_LOOKBACK simulations."""
    if not Path(RESULTS_FILE).exists():
        return {}
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        done = [r for r in csv.DictReader(f) if r.get("status") == "COMPLETE" and r.get("expression")]
    shapes = {}
    for r in done:
        if pool_tag(r):
            shapes[skeleton(r["expression"])] = r["expression"]
    recent = done[-SHAPE_LOOKBACK:]
    if len(recent) >= 10:
        counts = Counter(skeleton(r["expression"]) for r in recent)
        for r in reversed(recent):  # newest example, but a submitted alpha's from above wins
            shape = skeleton(r["expression"])
            if counts[shape] / len(recent) >= SHAPE_CROWDED_SHARE:
                shapes.setdefault(shape, r["expression"])
    return shapes


SHAPE_IDEAS = (  # (operators needed, suggestion) - only suggested if your account has those operators
    (("rank", "divide"), "a ratio or spread of two fields, ranked: rank(divide(x, y)) or rank(x - y)"),
    (("ts_corr", "rank"), "how two fields move together: -ts_corr(rank(x), rank(y), d)"),
    (("ts_regression",), "what's left after regressing one field on another: ts_regression(y, x, d)"),
    (("trade_when",), "a position that only updates when a condition holds: trade_when(condition, signal, -1)"),
    (("ts_delta", "ts_std_dev"), "a change scaled by its own volatility: ts_delta(x, d) / ts_std_dev(x, d)"),
    (("rank",), "a blend of two different data types: rank(x) + rank(y)"),
    (("ts_rank",), "where today's value sits in its own history: ts_rank(x, d)"),
    (("if_else",), "switching between two signals: if_else(condition, signal_a, signal_b)"),
)


def shape_rule(shapes: dict, operators) -> str:
    if not shapes:
        return ""
    names = {o.get("name") for o in operators or []}
    examples = "; ".join(list(shapes.values())[:3])
    lines = ["SHAPE RULE - vary the formula's structure, not just its data. These shapes are already "
             f"overused (same operators nested the same way, whatever the fields, windows, groups or "
             f"smoothing) - e.g. {examples}. At most {MAX_IDEAS_PER_CROWDED_SHAPE} ideas this round may "
             f"use any one of them."]
    ideas = [text for needed, text in SHAPE_IDEAS if not names or all(op in names for op in needed)]
    if ideas:
        lines.append("Other structures to try:\n" + "\n".join(f"- {text}" for text in ideas))
    lines.append("Smoothing is NOT part of the shape - keep using it. Wrap new structures in "
                 "ts_decay_linear(x, 10) to ts_decay_linear(x, 20), or hump(x, hump=0.01): most raw "
                 "structures trade far more than BRAIN's 70% turnover limit and get rejected without it.")
    return "\n".join(lines)


def label_fields(fields, type_map: dict) -> list:
    return [dict(f, description=f"[type: {type_map[f['name']]}] {f.get('description', '')}")
            if f.get("name") in type_map else f for f in fields]


def pick_varied(ideas, type_map: dict, crowded: Counter, n: int, overused_shapes=()) -> list:
    """Keep up to n ideas, in Gemini's order, without going over the per-type caps or using an
    overused formula shape more than MAX_IDEAS_PER_CROWDED_SHAPE times."""
    cap = per_type_cap(n, type_map)
    used, shapes_used, kept, by_type, by_shape = Counter(), Counter(), [], 0, 0
    for expression, rationale in ideas:
        types_used = expression_types(expression, type_map)
        if any(used[t] >= (MAX_IDEAS_PER_CROWDED_TYPE if t in crowded else cap) for t in types_used):
            by_type += 1
            continue
        shape = skeleton(expression)
        if shape in overused_shapes and shapes_used[shape] >= MAX_IDEAS_PER_CROWDED_SHAPE:
            by_shape += 1
            continue
        kept.append((expression, rationale))
        used.update(types_used)
        shapes_used[shape] += 1
        if len(kept) == n:
            break
    if by_type or by_shape:
        logger.info("Variety: skipped %d idea(s) leaning on data types already used enough this round and "
                    "%d repeating an overused formula shape.", by_type, by_shape)
    if kept:
        logger.info("This round: %d idea(s) in %d different formula shape(s).", len(kept), len(shapes_used))
    if used:
        logger.info("This round's data types: %s", ", ".join(f"{t} x{c}" for t, c in used.most_common()))
    return kept


def build_feedback(last_round_rows=None) -> str:
    """What the generator sees each round: your best results across ALL runs so far, so a new run
    builds on what already worked instead of starting from scratch, plus how the last round went."""
    history = completed_history()
    parts = []
    if history:
        best = sorted(history, key=lambda r: as_float(r["fitness"]), reverse=True)[:MEMORY_TOP_N]
        lines = [f"Your best results so far, out of {len(history)} simulated alphas across all runs "
                 f"(best fitness first). Learn from what works here - the operators, lookback windows, "
                 f"grouping and decay that score well, and what's holding each one back (usually fitness or "
                 f"turnover). But don't keep reusing the same data: a new alpha on the same data as one "
                 f"marked [submitted] or [queued] usually comes back too similar to submit."]
        for r in best:
            lines.append(f"- {r['expression']} -> sharpe={r['sharpe']} fitness={r['fitness']} "
                         f"turnover={r['turnover']}{turnover_note(r)}{pool_tag(r)}")
        parts.append("\n".join(lines))
    if last_round_rows:
        parts.append(summarize_round(last_round_rows))
    return "\n\n".join(parts) if parts else "This is round 1 - no prior results yet."


def generate_ideas(client: genai.Client, model: str, operators, fields, settings, feedback_text, avoid: set, n: int):
    op_list = "\n".join(f"- {o['name']}: {o['usage']}" for o in operators)
    field_list = "\n".join(f"- {f['name']}: {f['description']}" for f in fields)
    avoid_list = "\n".join(f"- {e}" for e in list(avoid)[:50]) or "(none yet)"

    prompt = (
        f"You are proposing WorldQuant BRAIN FASTEXPR alpha expressions for "
        f"region={settings['region']} universe={settings['universe']} delay={settings['delay']}.\n\n"
        f"ONLY use these operators, exactly as named (do not invent others):\n{op_list}\n\n"
        f"ONLY use these data fields, exactly as named (do not invent others):\n{field_list}\n\n"
        f"BRAIN only accepts an alpha with Sharpe >= {SUBMIT_MIN_SHARPE}, fitness >= {SUBMIT_MIN_FITNESS} "
        f"and turnover between {TURNOVER_MIN} and {TURNOVER_MAX} (1% to 70% of the book traded per day). "
        f"High turnover also drags fitness down, so favour smoother signals: longer lookback windows, "
        f"ts_decay_linear, or hump. Ideas that scored a strongly negative Sharpe are re-tested reversed "
        f"automatically, and promising ideas that trade too much are re-tested smoothed with "
        f"ts_decay_linear, so don't spend ideas on negating or smoothing past results.\n\n"
        f"{feedback_text}\n\n"
        f"Propose {n} NEW candidate expressions, genuinely different in shape from each "
        f"other (not the same template with one field swapped). Each needs a one-sentence "
        f"economic rationale for why it might predict returns.\n\n"
        f"Do not repeat any of these already-tried expressions:\n{avoid_list}"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AlphaIdeas,
            # We give the model no tools, so the SDK's automatic function calling has nothing
            # to do - turning it off just removes the "AFC is enabled" log noise.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    if not resp.text:
        raise ValueError("Model returned an empty response - possibly blocked by a safety filter.")

    data = json.loads(resp.text)
    return [(idea["expression"], idea["rationale"]) for idea in data["ideas"]]


def is_transient(exc: Exception) -> bool:
    """A busy (503) or rate-limited (429) answer can clear up on its own; a bad key or a
    malformed request can't. google-genai errors carry the HTTP status as `.code`."""
    if isinstance(exc, (httpx.TransportError, requests.RequestException)):
        return True  # dropped or timed-out connection
    return getattr(exc, "code", None) in TRANSIENT_STATUS_CODES


def short_error(exc: Exception) -> str:
    code, status = getattr(exc, "code", None), getattr(exc, "status", None)
    return f"{code} {status}" if code else f"{type(exc).__name__}: {str(exc)[:150]}"


def generate_with_fallback(client: genai.Client, operators, fields, settings, feedback_text, avoid: set, n: int):
    """Main model first, retried with backoff while it's busy or rate-limited. Then the
    fallback model once. Raises the last error only if both fail."""
    last_exc = None
    attempts = len(RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return generate_ideas(client, GEMINI_MODEL, operators, fields, settings, feedback_text, avoid, n)
        except Exception as exc:
            last_exc = exc
            if not is_transient(exc) or attempt == attempts:
                break  # a non-transient error won't fix itself by waiting
            delay = RETRY_DELAYS_SECONDS[attempt - 1]
            logger.warning("%s unavailable (%s) - waiting %ds, then retry %d of %d.",
                           GEMINI_MODEL, short_error(exc), delay, attempt, len(RETRY_DELAYS_SECONDS))
            time.sleep(delay)

    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        logger.warning("%s still failing (%s) - trying %s once.",
                       GEMINI_MODEL, short_error(last_exc), GEMINI_FALLBACK_MODEL)
        try:
            return generate_ideas(client, GEMINI_FALLBACK_MODEL, operators, fields, settings, feedback_text, avoid, n)
        except Exception as exc:
            last_exc = exc

    raise last_exc


def wait_seconds(resp, default=2.0) -> float:
    """BRAIN sends Retry-After as a float string ("5.0"); keep the wait between 0.5 and 30 seconds."""
    try:
        return min(max(float(resp.headers.get("Retry-After")), 0.5), 30.0)
    except (TypeError, ValueError):
        return default


def still_working(resp) -> bool:
    """BRAIN's 'not done yet' signals: a 303 back to the same address, or a Retry-After header."""
    return resp.status_code == 303 or resp.headers.get("Retry-After") is not None


def failing_reason(resp) -> str:
    """Names of the checks BRAIN marked FAIL in a reply, or "" if none are listed."""
    try:
        data = resp.json()
    except ValueError:
        return ""
    checks = []
    if isinstance(data, dict):
        checks = (data.get("is") or {}).get("checks") or data.get("checks") or []
    failed = [str(c.get("name")) for c in checks if isinstance(c, dict) and c.get("result") == "FAIL"]
    return ("failed " + ", ".join(failed)) if failed else ""


def self_correlation_max(session, alpha_id: str):
    """BRAIN's free pre-submit check: the highest correlation between this alpha and the ones you
    already submitted. Returns None if BRAIN didn't give a usable number in time."""
    url = CORRELATION_URL.format(alpha_id=alpha_id)
    deadline = time.time() + CORR_POLL_MAX_SECONDS
    while time.time() < deadline:
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException:
            return None
        if still_working(resp) and (resp.status_code == 503 or not resp.text.strip()):
            time.sleep(wait_seconds(resp))
            continue
        if resp.status_code != 200:
            return None
        try:
            value = resp.json().get("max")
            return float(value) if value is not None else None
        except (ValueError, TypeError, AttributeError):
            return None
    return None


def submit_alpha(session, alpha_id: str):
    """Submit, then poll until BRAIN gives its verdict. Returns (state, detail), state being one of
    YES, REJECTED, VERDICT_PENDING or FAILED. Following community clients that recorded BRAIN's
    live traffic: 503 or 303 means queued / still processing, Retry-After means poll again, and
    a 403 lists the failed checks. Anything ambiguous is reported as VERDICT_PENDING, never YES."""
    url = SUBMIT_URL.format(alpha_id=alpha_id)
    try:
        resp = session.post(url, timeout=30, allow_redirects=False)
    except requests.RequestException as exc:
        return "FAILED", f"couldn't reach BRAIN ({type(exc).__name__})"
    if resp.status_code == 403:
        return "REJECTED", failing_reason(resp) or f"403 {resp.text[:150]}"
    if resp.status_code == 429:
        return "FAILED", f"BRAIN is throttling submissions right now ({resp.text[:100]})"
    if resp.status_code not in (200, 201, 202, 303, 503):
        return "FAILED", f"{resp.status_code} {resp.text[:150]}"

    logger.info("BRAIN accepted the submission of %s - waiting for its verdict...", alpha_id)
    deadline = time.time() + SUBMIT_POLL_MAX_SECONDS
    while time.time() < deadline:
        time.sleep(wait_seconds(resp))
        try:
            resp = session.get(url, timeout=30, allow_redirects=False)
        except requests.RequestException:
            continue
        if still_working(resp) or resp.status_code == 429:
            continue
        if resp.status_code == 403:
            return "REJECTED", failing_reason(resp) or f"403 {resp.text[:150]}"
        if resp.status_code == 200 and resp.text.strip():
            reason = failing_reason(resp)
            return ("REJECTED", reason) if reason else ("YES", None)
        if resp.status_code == 200:
            return "VERDICT_PENDING", None  # empty answer with no 'wait' signal: can't tell either way
        return "FAILED", f"{resp.status_code} {resp.text[:150]}"
    return "VERDICT_PENDING", None


def is_active_on_brain(session, alpha_id: str) -> bool:
    """Submitted alphas are listed by BRAIN with status ACTIVE (unsubmitted ones: UNSUBMITTED)."""
    return be.fetch_alpha(session, alpha_id).get("status") == "ACTIVE"


_UNCHECKED = object()  # try_submit's default: no self-correlation reading taken yet


def too_similar(corr) -> bool:
    return corr is not None and corr >= MAX_SELF_CORRELATION


def try_submit(session, alpha_id: str, corr=_UNCHECKED) -> str:
    """Pre-check self-correlation (unless the caller just did), submit, wait for the verdict.
    Returns the CSV 'submitted' value."""
    if corr is _UNCHECKED:
        corr = self_correlation_max(session, alpha_id)
    if too_similar(corr):
        logger.info("Alpha %s clears the bar, but its self-correlation is %.2f (limit %.1f) - "
                    "not spending today's submission on it.", alpha_id, corr, MAX_SELF_CORRELATION)
        return f"TOO_CORRELATED (max {corr:.2f})"
    if corr is None:
        logger.warning("Couldn't read %s's self-correlation in advance - submitting anyway; "
                       "BRAIN runs the same check during submission.", alpha_id)
    else:
        logger.info("Self-correlation %.2f - under the %.1f limit.", corr, MAX_SELF_CORRELATION)

    logger.info("Alpha %s qualifies - submitting.", alpha_id)
    state, detail = submit_alpha(session, alpha_id)
    if state in ("YES", "REJECTED", "VERDICT_PENDING"):
        record_submission()  # BRAIN processed it, so it counts against today's limit
    if state == "YES":
        try:
            if not is_active_on_brain(session, alpha_id):
                state = "VERDICT_PENDING"  # only claim success once BRAIN itself lists it as submitted
        except requests.RequestException:
            state = "VERDICT_PENDING"  # couldn't confirm - the dashboard can, later
    if state == "YES":
        logger.info("SUCCESSFULLY SUBMITTED Alpha ID: %s", alpha_id)
        return "YES"
    if state == "REJECTED":
        logger.warning("BRAIN rejected the submission of %s: %s", alpha_id, detail)
        return f"REJECTED: {detail}"
    if state == "VERDICT_PENDING":
        logger.warning("BRAIN took %s but hasn't confirmed it as submitted yet - press Check with "
                       "BRAIN in the dashboard later.", alpha_id)
        return "VERDICT_PENDING (check with BRAIN in the dashboard)"
    logger.error("Couldn't submit %s: %s", alpha_id, detail)
    return f"FAILED: {detail}"


def needs_smoothing(sharpe, fitness, turnover) -> bool:
    """A decent Sharpe held back by trading too much. Over BRAIN's 70% turnover limit the alpha is
    rejected outright, so smoothing is tried from Sharpe SMOOTH_MIN_SHARPE_OVER_LIMIT (1.0); otherwise
    from the submission bar, when fitness is short. Fitness divides by turnover (floored at 0.125),
    so decay helps most when turnover is high."""
    if sharpe is None or fitness is None or turnover is None:
        return False
    if turnover > TURNOVER_MAX:
        return sharpe >= SMOOTH_MIN_SHARPE_OVER_LIMIT
    if sharpe < SUBMIT_MIN_SHARPE or turnover <= SMOOTH_MIN_TURNOVER:
        return False
    return fitness < SUBMIT_MIN_FITNESS


# ---- Auto-repair: strong alphas that fail one of BRAIN's fixable checks ----
# Fixes follow community write-ups of BRAIN's checks (rank / ts_backfill for concentrated weight;
# finer groups / no size tilt for weak sub-universe Sharpe). Each repair is one extra simulation,
# and BRAIN's own result decides whether it worked.

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COARSE_GROUPS = ("market", "sector", "industry")
NOT_FIELDS = {"true", "false", "nan", "inf"}


def _blank_quotes(text: str) -> str:
    """Same length, with quoted parts (like range='0, 1, 0.1') blanked out so commas and words inside
    them are ignored."""
    return _QUOTED.sub(lambda m: " " * len(m.group()), text)


def matching_paren(text: str, open_index: int) -> int:
    depth = 0
    plain = _blank_quotes(text)
    for i in range(open_index, len(plain)):
        if plain[i] == "(":
            depth += 1
        elif plain[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level(text: str) -> list:
    """Split an argument list at its own commas, not at commas inside nested calls or quotes."""
    plain, parts, depth, start = _blank_quotes(text), [], 0, 0
    for i, ch in enumerate(plain):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def outer_call(expression: str):
    """(name, args) when the whole expression is one call like ts_decay_linear(x, 10), else None."""
    expr = expression.strip()
    m = re.match(r"([A-Za-z_]\w*)\s*\(", expr)
    if not m:
        return None
    close = matching_paren(expr, m.end() - 1)
    if close != len(expr) - 1:
        return None
    return m.group(1), split_top_level(expr[m.end():close])


def _group_calls(expression: str):
    """(start of last argument, end of call) for every group_* call - the last argument is the group."""
    plain = _blank_quotes(expression)
    for m in re.finditer(r"\bgroup_\w+\s*\(", plain):
        close = matching_paren(expression, m.end() - 1)
        if close < 0:
            continue
        args = split_top_level(expression[m.end():close])
        yield close - len(args[-1]), close


def data_fields_in(expression: str) -> list:
    """(start, end, name) of each data-field reference: bare words that aren't operators, keyword
    arguments, quoted text or the group argument of a group_* call."""
    plain = _blank_quotes(expression)
    group_args = [(s, e) for s, e in _group_calls(expression)]
    found = []
    for m in _WORD.finditer(plain):
        after = plain[m.end():].lstrip()
        if after.startswith("(") or (after.startswith("=") and not after.startswith("==")):
            continue
        if m.group().lower() in NOT_FIELDS or plain[max(0, m.start() - 1)].isdigit():
            continue
        if any(s <= m.start() and m.end() <= e and expression[s:e].strip() == m.group() for s, e in group_args):
            continue
        found.append((m.start(), m.end(), m.group()))
    return found


def field_exists(name: str) -> bool:
    """Whether BRAIN listed this data field for your account (data_fields_cache.json from fetch_fields.py)."""
    try:
        cache = json.loads(Path(FIELDS_CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(f.get("id") == name for f in cache.get("matrix_fields") or [] if isinstance(f, dict))


def _wrap_in_rank(expr):
    call = outer_call(expr)
    if call and call[0] == "rank":
        return None
    return (f"rank({expr.strip()})",
            "rank() spreads the weight evenly across stocks instead of a few big bets")


def _backfill_fields(expr):
    fields = data_fields_in(expr)
    if not fields or "ts_backfill(" in expr:
        return None
    for start, end, name in reversed(fields):
        expr = expr[:start] + f"ts_backfill({name}, {BACKFILL_DAYS})" + expr[end:]
    return (expr.strip(), f"ts_backfill fills gaps in sparse data (up to {BACKFILL_DAYS} days), so more stocks "
                          f"get a position")


def _finer_groups(expr):
    changed = False
    for start, end in sorted(_group_calls(expr), reverse=True):
        arg = expr[start:end]
        if arg.strip() in COARSE_GROUPS:
            expr = expr[:start] + arg[:len(arg) - len(arg.lstrip())] + "subindustry" + expr[end:]
            changed = True
    if not changed:
        return None
    return (expr.strip(), "ranking within subindustries removes the size and sector tilts that weaken it in "
                          "the larger, more liquid stocks")


def _size_neutral(expr):
    if "bucket(rank(cap)" in expr or not field_exists("cap"):
        return None
    return (f"group_neutralize({expr.strip()}, bucket(rank(cap), range='0, 1, 0.1'))",
            "neutralizing against market-cap buckets removes the size tilt that weakens it in the more "
            "liquid stocks")


def _less_smoothing(expr):
    call = outer_call(expr)
    if not call or call[0] not in ("ts_decay_linear", "hump") or not call[1][0].strip():
        return None
    return (call[1][0].strip(), f"without the outer {call[0]} it trades enough to clear the 1% minimum")


def needs_sharpening(row) -> bool:
    """Sharpe is over the bar but fitness isn't, while turnover is already under BRAIN's 12.5% fitness
    floor - so the smoothing wrapped around it can't help fitness any more, only dampen returns."""
    sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))
    if sharpe is None or fitness is None or turnover is None:
        return False
    return (sharpe >= SUBMIT_MIN_SHARPE and fitness < SUBMIT_MIN_FITNESS and turnover < FITNESS_TURNOVER_FLOOR
            and _less_smoothing(row.get("expression") or "") is not None)


REPAIRS = {  # check name -> fixes to try, most promising first
    "CONCENTRATED_WEIGHT": (_wrap_in_rank, _backfill_fields),
    "LOW_SUB_UNIVERSE_SHARPE": (_finer_groups, _size_neutral),
    "LOW_TURNOVER": (_less_smoothing,),
}


def repair_candidates(expression: str, failed_checks) -> list:
    """(new expression, why) fixes for these failed checks - one per check first, then the next fix for
    each, up to MAX_REPAIRS_PER_ALPHA. Empty if any failed check isn't one we know how to fix."""
    failed = [c for c in dict.fromkeys(failed_checks)]
    if not failed or any(c not in REPAIRS for c in failed):
        return []
    out, seen = [], {expression.strip()}
    for depth in range(max(len(REPAIRS[c]) for c in failed)):
        for check in failed:
            if depth < len(REPAIRS[check]):
                fix = REPAIRS[check][depth](expression)
                if fix and fix[0] not in seen:
                    seen.add(fix[0])
                    out.append((fix[0], f"failed {check}; {fix[1]}"))
    return out[:MAX_REPAIRS_PER_ALPHA]


def all_repair_variants(expression: str) -> set:
    """Every repair any fixable check could produce - lets a run skip alphas whose repairs were all tried."""
    return {fix[0] for fixes in REPAIRS.values() for f in fixes for fix in [f(expression)] if fix}


def strong_numbers(row) -> bool:
    sharpe, fitness = as_float(row.get("sharpe")), as_float(row.get("fitness"))
    return (sharpe is not None and fitness is not None
            and sharpe >= SUBMIT_MIN_SHARPE and fitness >= SUBMIT_MIN_FITNESS)


def qualifies(metrics: dict, checks: list) -> bool:
    """Every check must pass, except SELF_CORRELATION while it reads PENDING: BRAIN only works
    that one out during submission, so it shows PENDING on every unsubmitted alpha."""
    if not checks:
        return False
    if (safe_float(metrics.get("sharpe")) < SUBMIT_MIN_SHARPE
            or safe_float(metrics.get("fitness")) < SUBMIT_MIN_FITNESS):
        return False
    for c in checks:
        if c.get("result") == "PASS":
            continue
        if c.get("name") == "SELF_CORRELATION" and c.get("result") == "PENDING":
            continue
        return False
    return True


def append_result(row: dict):
    is_new = not Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": rows carry in-memory extras (failing_checks) the CSV doesn't store
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---- Held alphas: cleared the bar on a day the submission slot was already used ----

def held_candidates() -> list:
    """Held alphas still waiting, best fitness first (Sharpe breaks ties). An alpha that already
    has an outcome logged in another row (mark_held's fallback) is left out."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    held, decided = {}, set()
    for r in rows:
        alpha_id = (r.get("alpha_id") or "").strip()
        submitted = (r.get("submitted") or "").strip()
        if not alpha_id:
            continue
        if submitted.startswith(HELD_PREFIX):
            held.setdefault(alpha_id, r)
        elif submitted not in ("", "NO"):
            decided.add(alpha_id)
    waiting = [r for alpha_id, r in held.items() if alpha_id not in decided]
    return sorted(waiting, key=lambda r: (safe_float(r.get("fitness")), safe_float(r.get("sharpe"))),
                  reverse=True)


def mark_held(alpha_id: str, outcome: str):
    """Swap a held alpha's 'submitted' value for what happened to it, so the dashboard and later
    runs still see one row per alpha. The CSV is written to a temporary file and swapped in, so a
    crash can't leave it half-written. If Windows blocks the swap (another program has the file
    open), the outcome is appended as a new row instead, so it's never lost."""
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or RESULTS_HEADER
        rows = list(reader)
    target = None
    for r in rows:
        if r.get("alpha_id") == alpha_id and (r.get("submitted") or "").startswith(HELD_PREFIX):
            r["submitted"] = outcome
            target = target or r
    if target is None:
        return

    tmp = Path(RESULTS_FILE + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, RESULTS_FILE)
            return
        except OSError:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(1)
    try:
        tmp.unlink()
    except OSError:
        pass
    logger.warning("Couldn't update %s in place (is it open in another program?) - adding the outcome "
                   "for %s as a new row instead.", RESULTS_FILE, alpha_id)
    append_result(dict(target, round="held", timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))


def submit_held(session):
    """Start-of-run step: if today's slot is free, submit the best held alpha. Returns the outcome
    that used the slot (YES, REJECTED..., VERDICT_PENDING...), or None. Never stops the run."""
    try:
        return _submit_held(session)
    except Exception:
        logger.exception("The held-alpha queue hit an unexpected problem - skipping it this run. "
                         "Held alphas stay queued.")
        return None


def _submit_held(session):
    """Re-check every held alpha with BRAIN - including its free similarity check, which grows stricter
    as you submit more - so the queue only keeps alphas that could still be submitted. If today's slot
    is free, the best remaining one (highest fitness) is submitted. Stops after HELD_CHECK_BUDGET_SECONDS;
    anything not checked yet waits for the next run."""
    queue = held_candidates()
    if not queue:
        return None
    slot_free = submissions_today() < MAX_SUBMISSIONS_PER_DAY
    logger.info("%d held alpha(s) - checking they're still worth submitting. %s", len(queue),
                "Today's submission slot is free, so the best one goes first." if slot_free else
                "Today's slot is used, so they wait for BRAIN's next day (3 AM US Eastern).")

    deadline = time.time() + HELD_CHECK_BUDGET_SECONDS
    first_outcome, submitted, waiting, dropped = None, 0, 0, 0
    for index, row in enumerate(queue):
        if time.time() > deadline:
            waiting += len(queue) - index
            logger.info("Held-queue check time is up - the rest are checked next run.")
            break
        alpha_id = row["alpha_id"]
        logger.info("Held alpha %s (Sharpe %s, fitness %s): %s", alpha_id, row.get("sharpe"),
                    row.get("fitness"), row.get("expression"))
        try:
            alpha = be.fetch_alpha(session, alpha_id)
        except requests.RequestException as exc:
            logger.warning("Couldn't reach BRAIN to check held alpha %s (%s) - the rest stay queued "
                           "for the next run.", alpha_id, type(exc).__name__)
            waiting += len(queue) - index
            break

        if alpha.get("status") == "ACTIVE":
            logger.info("BRAIN already lists %s as submitted - marking it YES.", alpha_id)
            mark_held(alpha_id, "YES")
            continue
        checks = (alpha.get("is") or {}).get("checks") or []
        if not checks:
            logger.warning("BRAIN didn't return %s's checks - leaving it queued.", alpha_id)
            waiting += 1
            continue
        metrics = be.metrics_from_alpha(alpha_id, alpha)
        if not qualifies(metrics, checks):
            failed = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
            if (failed or safe_float(metrics.get("sharpe")) < SUBMIT_MIN_SHARPE
                    or safe_float(metrics.get("fitness")) < SUBMIT_MIN_FITNESS):
                reason = ("failed " + ", ".join(failed)) if failed else "below the Sharpe/fitness bar"
                logger.warning("Held alpha %s no longer qualifies on BRAIN (%s) - dropping it from the queue.",
                               alpha_id, reason)
                mark_held(alpha_id, f"NOT_SUBMITTED (no longer qualifies: {reason})")
                dropped += 1
            else:
                logger.info("A check on %s isn't settled yet - leaving it queued.", alpha_id)
                waiting += 1
            continue

        corr = self_correlation_max(session, alpha_id)
        if too_similar(corr):
            logger.info("Held alpha %s is now too similar to your submitted alphas (self-correlation %.2f) - "
                        "dropping it from the queue.", alpha_id, corr)
            mark_held(alpha_id, f"TOO_CORRELATED (max {corr:.2f})")
            dropped += 1
            continue
        if submissions_today() >= MAX_SUBMISSIONS_PER_DAY:
            waiting += 1  # still a good candidate - stays queued for a free slot
            continue

        outcome = try_submit(session, alpha_id, corr=corr)
        if outcome.startswith("FAILED"):
            logger.warning("Held alpha %s stays queued - the submit didn't go through.", alpha_id)
            waiting += 1
            continue
        mark_held(alpha_id, outcome)
        submitted += 1  # BRAIN processed it, so it used a slot
        first_outcome = first_outcome or outcome

    logger.info("Held queue: %d submitted, %d dropped (too similar or no longer qualifying), %d still waiting.",
                submitted, dropped, waiting)
    return first_outcome


BUSY = "BUSY"  # BRAIN is already running as many of your simulations as it allows


def new_row(expression, rationale, round_num) -> dict:
    row = {k: "" for k in RESULTS_HEADER}
    row.update(round=round_num, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               expression=expression, rationale=rationale, submitted="NO")
    return row


def renew_session(session, email, password, row):
    """One unattended re-login. If BRAIN wants identity verification, logs this idea as STOPPED
    and raises StopLoop - nobody is here to click the link."""
    logger.warning("Session expired mid-run - attempting one re-auth...")
    session.cookies.clear()
    if not log_in(session, email, password):
        row["status"], row["error"] = "STOPPED", "session expired and could not renew unattended"
        append_result(row)
        raise StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to complete it.")


def is_busy(err) -> bool:
    """BRAIN's 'too many simulations at once' answer (429, or a CONCURRENT_SIMULATION... message)."""
    text = str(err or "").upper()
    return text.startswith("SUBMIT FAILED (429)") or "CONCURRENT_SIMULATION" in text or "SIMULATION_LIMIT" in text


def start_idea(session, row, settings, email, password):
    """Start the simulation. Returns (location, None), (None, BUSY) or (None, error)."""
    location, err = be.submit_simulation(session, row["expression"], settings)
    if err == "EXPIRED_SESSION":
        renew_session(session, email, password, row)
        location, err = be.submit_simulation(session, row["expression"], settings)
    if err and is_busy(err):
        return None, BUSY
    return location, err


def fail(row, error) -> dict:
    row["status"], row["error"] = "FAILED", str(error)[:300]
    append_result(row)
    return row


def finish_idea(session, row, alpha_id) -> dict:
    """A simulation finished: read BRAIN's result, log it, and submit the alpha if it qualifies."""
    expression = row["expression"]
    alpha = be.fetch_alpha(session, alpha_id)
    if not (alpha.get("is") or {}):
        row.update(alpha_id=alpha_id, alpha_link=f"https://platform.worldquantbrain.com/alpha/{alpha_id}")
        logger.warning("  %s finished, but BRAIN didn't send its result.", expression)
        return fail(row, "BRAIN finished the simulation but didn't send its result - open its BRAIN page")
    metrics = be.metrics_from_alpha(alpha_id, alpha)
    checks = (alpha.get("is") or {}).get("checks") or []
    row.update(status="COMPLETE", alpha_id=alpha_id, **metrics)
    # Kept in memory for the next round's feedback; the CSV doesn't store it.
    row["failing_checks"] = ", ".join(f"{c.get('name')}:{c.get('result')}" for c in checks if c.get("result") != "PASS")
    row["failed_checks"] = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]

    if qualifies(metrics, checks):
        if submissions_today() >= MAX_SUBMISSIONS_PER_DAY:
            # Only hold it if it could actually be submitted later: BRAIN's similarity check is free.
            corr = self_correlation_max(session, alpha_id)
            if too_similar(corr):
                logger.info("Alpha %s clears the bar, but it's too similar to your submitted alphas "
                            "(self-correlation %.2f, limit %.1f) - not holding it.",
                            alpha_id, corr, MAX_SELF_CORRELATION)
                row["submitted"] = f"TOO_CORRELATED (max {corr:.2f})"
            else:
                logger.warning("Alpha %s qualifies%s, but today's submission limit (%d) is used - holding "
                               "it. The first run with a free slot submits held alphas first.", alpha_id,
                               "" if corr is None else f" (self-correlation {corr:.2f})", MAX_SUBMISSIONS_PER_DAY)
                row["submitted"] = HELD_LABEL
        else:
            row["submitted"] = try_submit(session, alpha_id)

    append_result(row)
    logger.info("  %s -> sharpe=%s fitness=%s turnover=%s checks=%s/%s submitted=%s", expression,
                metrics.get("sharpe"), metrics.get("fitness"), metrics.get("turnover"),
                metrics.get("checks_passed"), metrics.get("checks_total"), row["submitted"])
    return row


def run_round(session, client, operators, fields, settings, email, password, round_num, feedback_text, avoid):
    logger.info("=== Round %d: generating %d ideas ===", round_num, IDEAS_PER_ROUND)
    rejected, dead = rejected_names(), dead_end_fields()
    blocked = rejected | set(dead)
    operators = [o for o in operators if o.get("name") not in rejected]
    fields = [f for f in fields if f.get("name") not in blocked]
    type_map = field_types(fields)
    crowded = crowded_types(type_map)
    overused = crowded_shapes()
    rules = [r for r in (variety_rule(type_map, crowded, IDEAS_PER_ROUND), shape_rule(overused, operators),
                         avoid_note(rejected, dead)) if r]
    asked = IDEAS_PER_ROUND + (VARIETY_EXTRA_IDEAS if rules else 0)
    prompt_text = "\n\n".join([feedback_text, *rules])
    try:
        ideas = generate_with_fallback(client, operators, label_fields(fields, type_map), settings,
                                       prompt_text, avoid, asked)
    except Exception as e:
        raise GenerationFailed(short_error(e)) from e

    if blocked:
        usable = [(e, r) for e, r in ideas if not uses_any(e, blocked)]
        if len(usable) < len(ideas):
            logger.info("Skipped %d idea(s) using names BRAIN rejected or dead-end fields.", len(ideas) - len(usable))
        ideas = usable
    if rules:
        fresh_ideas = []
        for expression, rationale in ideas:
            if expression in avoid:
                logger.info("Skipping duplicate: %s", expression)
            else:
                fresh_ideas.append((expression, rationale))
        ideas = pick_varied(fresh_ideas, type_map, crowded, IDEAS_PER_ROUND, overused)

    return process_queue(session, [(e, r, "idea") for e, r in ideas], settings, email, password, round_num, avoid)


def process_queue(session, items, settings, email, password, round_num, avoid) -> list:
    """Test each (expression, rationale, kind), up to PARALLEL_SIMULATIONS at once. A flip or
    smooth goes to the FRONT of the queue, so it starts in the next free slot, right after the
    alpha that triggered it - even a short or interrupted run tests its best leads first.
    Finished simulations are handled one at a time, so submissions never overlap."""
    queue = deque(items)
    running = []  # each: {"row", "kind", "location", "started"}
    rows = []
    counts = {"flips": 0, "smooths": 0, "repairs": 0, "sharpens": 0, "network": 0}
    limit = PARALLEL_SIMULATIONS
    busy_since = None
    last_note = time.time()

    def network_trouble(row, exc):
        counts["network"] += 1
        logger.warning("Couldn't reach BRAIN while testing %s (%s), even after retrying - skipping it. "
                       "Retry it later with --try.", row["expression"], type(exc).__name__)
        rows.append(fail(row, f"Couldn't reach BRAIN ({type(exc).__name__}) - not a verdict on the idea; "
                              f"retry it with --try"))
        if counts["network"] >= MAX_NETWORK_FAILURES:
            raise StopLoop(f"Couldn't reach BRAIN for {counts['network']} ideas in a row - your connection "
                           f"seems to be down. Stopped cleanly; everything so far is saved.")

    def unexpected(row, exc):  # an unexpected reply shouldn't end the whole run
        logger.error("Unexpected error while testing %s (%s: %s) - skipping it and carrying on.",
                     row["expression"], type(exc).__name__, exc, exc_info=exc)
        rows.append(fail(row, f"Unexpected error: {type(exc).__name__}: {exc}"))

    try:
        while queue or running:
            keep_logged_in(session, email, password)
            # 1. Start ideas while there's a free slot.
            while queue and len(running) < limit:
                expression, rationale, kind = queue[0]
                if expression in avoid:
                    queue.popleft()
                    logger.info("Skipping duplicate: %s", expression)
                    continue
                row = new_row(expression, rationale, round_num)
                try:
                    location, err = start_idea(session, row, settings, email, password)
                except StopLoop:
                    raise
                except requests.RequestException as exc:
                    queue.popleft()
                    avoid.add(expression)
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    queue.popleft()
                    avoid.add(expression)
                    unexpected(row, exc)
                    continue

                if err == BUSY:
                    if running:  # BRAIN's limit is however many we already have going
                        if limit != len(running):
                            limit = len(running)
                            logger.info("BRAIN is running as many of your simulations as it allows - "
                                        "carrying on with %d at a time.", limit)
                        break
                    busy_since = busy_since or time.time()
                    if time.time() - busy_since >= MAX_BUSY_WAIT_SECONDS:
                        queue.popleft()
                        avoid.add(expression)
                        busy_since = None
                        rows.append(fail(row, "BRAIN stayed busy for 10 minutes (simulations running "
                                              "elsewhere?) - retry it with --try"))
                        continue
                    logger.info("BRAIN says it's busy - waiting %ds before starting the next idea.",
                                BUSY_WAIT_SECONDS)
                    time.sleep(BUSY_WAIT_SECONDS)
                    continue

                queue.popleft()
                avoid.add(expression)
                busy_since = None
                counts["network"] = 0
                if err:
                    rows.append(fail(row, err))
                    continue
                running.append({"row": row, "kind": kind, "location": location, "started": time.time()})
                logger.info("Simulating (%d running): %s", len(running), expression)

            if not running:
                continue

            # 2. Check each running simulation once.
            time.sleep(SIM_CHECK_SECONDS)
            if time.time() - last_note >= 60:
                last_note = time.time()
                logger.info("  ... %d simulation(s) running, longest for %ds", len(running),
                            int(time.time() - min(i["started"] for i in running)))
            for item in list(running):
                row = item["row"]
                try:
                    state, info = be.poll_once(session, item["location"])
                except requests.RequestException as exc:
                    running.remove(item)
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    running.remove(item)
                    unexpected(row, exc)
                    continue

                if state == "RUNNING":
                    if time.time() - item["started"] > be.MAX_POLL_SECONDS:
                        running.remove(item)
                        rows.append(fail(row, f"timed out after {be.MAX_POLL_SECONDS}s"))
                    continue
                running.remove(item)
                if state == "EXPIRED":
                    renew_session(session, email, password, row)  # raises StopLoop if it can't
                    running.append(item)                            # renewed: keep checking it
                    continue
                if state == "ERROR":
                    rows.append(fail(row, info))
                    continue

                counts["network"] = 0
                try:
                    try:
                        row = finish_idea(session, row, info)
                    except be.SessionExpired:
                        renew_session(session, email, password, row)  # raises StopLoop if it can't
                        row = finish_idea(session, row, info)
                except requests.RequestException as exc:
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    unexpected(row, exc)
                    continue
                rows.append(row)
                queue_followups(session, row, item["kind"], queue, avoid, counts)
    except StopLoop:
        for item in running:  # don't lose track of simulations still going
            item["row"].update(status="STOPPED",
                               error="the run stopped before this simulation finished - retry it with --try")
            append_result(item["row"])
        raise
    return rows


def queue_followups(session, row, kind, queue, avoid, counts):
    """Flip, smooth or repair a finished alpha when that could rescue it. Goes to the front of the queue."""
    if row["status"] != "COMPLETE":
        return
    expression = row["expression"]
    sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))

    # Auto-flip: a strongly negative Sharpe means a real signal pointing the wrong way.
    if (kind == "idea" and sharpe is not None and sharpe <= FLIP_SHARPE_AT_OR_BELOW
            and counts["flips"] < MAX_FLIPS_PER_ROUND and not expression.startswith("reverse(")):
        flipped = f"reverse({expression})"
        if flipped not in avoid:
            counts["flips"] += 1
            logger.info("  Sharpe %.2f points the wrong way - testing the flipped version next.", sharpe)
            queue.appendleft((flipped, f"Auto-flip of {expression}, which scored Sharpe {sharpe:.2f}; "
                                       f"reversed, it should score about {-sharpe:.2f}.", "flip"))

    # Auto-smooth: the Sharpe is there, but trading too much is holding it back.
    if (kind not in ("smooth", "sharpen") and counts["smooths"] < MAX_SMOOTHS_PER_ROUND
            and needs_smoothing(sharpe, fitness, turnover)):  # a sharpened result isn't smoothed back
        smoothed = f"ts_decay_linear({expression}, {SMOOTH_DECAY_DAYS})"
        if smoothed not in avoid:
            counts["smooths"] += 1
            logger.info("  Sharpe %.2f is there, but fitness %.2f at %.0f%% turnover isn't - "
                        "testing a smoothed version next.", sharpe, fitness, turnover * 100)
            queue.appendleft((smoothed, f"Auto-smooth of {expression}: Sharpe {sharpe:.2f} but fitness "
                                        f"{fitness:.2f} at {turnover:.0%} turnover; decaying it over "
                                        f"{SMOOTH_DECAY_DAYS} days should cut turnover.", "smooth"))

    # Auto-sharpen: Sharpe is there, fitness isn't, and turnover is already under the 12.5% floor.
    if (kind not in ("smooth", "sharpen", "repair") and counts["sharpens"] < MAX_SHARPENS_PER_ROUND
            and row.get("submitted", "NO") == "NO" and needs_sharpening(row)):
        sharper = _less_smoothing(expression)[0]
        if sharper not in avoid and not too_similar_to_pool(session, row, "sharpening"):
            counts["sharpens"] += 1
            logger.info("  Sharpe %.2f but fitness %.2f at only %.1f%% turnover - testing it without the outer "
                        "smoothing next.", sharpe, fitness, turnover * 100)
            queue.appendleft((sharper, sharpen_reason(expression, sharpe, fitness, turnover), "sharpen"))

    # Auto-repair: strong numbers held back by one of BRAIN's fixable checks.
    if kind != "repair" and row.get("submitted", "NO") == "NO" and strong_numbers(row):
        failed = row.get("failed_checks") or []
        possible = repair_candidates(expression, failed)
        if not possible:
            if failed:
                logger.info("  Strong numbers but failed %s - no automatic fix for that check.", ", ".join(failed))
            elif row.get("failing_checks"):
                logger.info("  Strong numbers but not every check passed (%s).", row["failing_checks"])
        fixes = [f for f in possible if f[0] not in avoid]
        room = MAX_REPAIRS_PER_ROUND - counts["repairs"]
        if fixes and room > 0 and worth_repairing(session, row):
            counts["repairs"] += min(room, len(fixes))
            for new_expression, why in reversed(fixes[:room]):  # reversed: appendleft keeps their order
                queue.appendleft((new_expression, f"Auto-repair of {expression}: {why}.", "repair"))


def sharpen_reason(expression, sharpe, fitness, turnover) -> str:
    return (f"Auto-sharpen of {expression}: Sharpe {sharpe:.2f} but fitness {fitness:.2f} at {turnover:.1%} "
            f"turnover - under BRAIN's 12.5% fitness floor, so the outer smoothing only dampens returns.")


def too_similar_to_pool(session, row, purpose: str) -> bool:
    """A follow-up is a close copy of the original, so if the original is already too similar to your
    submitted alphas, the follow-up would be too. BRAIN's similarity check is free; asking first saves
    the simulation."""
    corr = self_correlation_max(session, row["alpha_id"])
    if too_similar(corr):
        logger.info("  %s is too similar to your submitted alphas (self-correlation %.2f) - not %s it.",
                    row["expression"], corr, purpose)
        return True
    return False


def worth_repairing(session, row) -> bool:
    if too_similar_to_pool(session, row, "repairing"):
        return False
    logger.info("  Strong numbers but failed %s - testing a repaired version next.",
                ", ".join(row.get("failed_checks") or []))
    return True


def near_miss_rows() -> list:
    """Earlier alphas with Sharpe and fitness over the bar that weren't submitted - most recent first."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "COMPLETE" and r.get("alpha_id") and strong_numbers(r)
                and (r.get("submitted") or "NO").strip() == "NO"]
    unique = {r["alpha_id"]: r for r in rows}  # one row per alpha
    return sorted(unique.values(), key=lambda r: r.get("timestamp") or "", reverse=True)


def sharpen_candidates() -> list:
    """Earlier alphas that needs_sharpening() and weren't submitted - most recent first."""
    rows = [r for r in completed_history() if r.get("alpha_id") and needs_sharpening(r)
            and (r.get("submitted") or "NO").strip() == "NO"]
    unique = {r["alpha_id"]: r for r in rows}
    return sorted(unique.values(), key=lambda r: r.get("timestamp") or "", reverse=True)


def repair_past_near_misses(session, settings, email, password, avoid) -> list:
    """Start-of-run step: re-examine up to MAX_PAST_REPAIRS recent near-misses whose follow-ups haven't
    been tried - repair the ones failing a fixable check (asking BRAIN which check failed), and test the
    slow ones without their outer smoothing. Every near-miss looked at gets a log line saying why."""
    candidates = [("repair", r) for r in near_miss_rows()] + [("sharpen", r) for r in sharpen_candidates()]
    candidates.sort(key=lambda kr: kr[1].get("timestamp") or "", reverse=True)
    items, looked = [], 0
    for kind, row in candidates:
        if looked >= MAX_PAST_REPAIRS:
            break
        expression = row["expression"]
        if kind == "sharpen":
            sharper = _less_smoothing(expression)[0]
            if sharper in avoid:
                continue  # already tried
            looked += 1
            logger.info("Earlier near-miss %s (Sharpe %s, fitness %s, turnover %s):", expression,
                        row.get("sharpe"), row.get("fitness"), row.get("turnover"))
            if too_similar_to_pool(session, row, "sharpening"):
                continue
            logger.info("  Turnover is under BRAIN's 12.5% fitness floor - testing it without the outer smoothing.")
            items.append((sharper, sharpen_reason(expression, *(as_float(row[k]) for k in
                                                                ("sharpe", "fitness", "turnover"))), "sharpen"))
            continue

        if all_repair_variants(expression) <= avoid:
            continue  # every possible fix was already tried
        looked += 1
        try:
            alpha = be.fetch_alpha(session, row["alpha_id"])
        except requests.RequestException as exc:
            logger.warning("Couldn't reach BRAIN to look at earlier near-misses (%s) - skipping them this run.",
                           type(exc).__name__)
            break
        checks = (alpha.get("is") or {}).get("checks") or []
        row["failed_checks"] = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
        label = f"Earlier near-miss {expression} (Sharpe {row.get('sharpe')}, fitness {row.get('fitness')})"
        if not checks:
            logger.info("%s: couldn't read BRAIN's record of it - trying again next run.", label)
            continue
        if not row["failed_checks"]:
            pending = ", ".join(f"{c.get('name')}:{c.get('result')}" for c in checks
                                if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION")
            logger.info("%s: BRAIN lists no failed check%s.", label, f" ({pending})" if pending else "")
            continue
        possible = repair_candidates(expression, row["failed_checks"])
        if not possible:
            logger.info("%s fails %s - no automatic fix for that check.", label, ", ".join(row["failed_checks"]))
            continue
        fixes = [f for f in possible if f[0] not in avoid]
        if not fixes:
            continue  # its fixes were already tried
        logger.info("%s:", label)
        if not worth_repairing(session, row):
            continue
        items.extend((e, f"Auto-repair of {expression}: {why}.", "repair") for e, why in fixes)
    if not items:
        return []
    logger.info("=== Following up earlier near-misses: %d version(s) to test ===", len(items))
    return process_queue(session, items, settings, email, password, "repair", avoid)


def run_try(session, expressions, settings, email, password) -> list:
    """--try: your own expressions through the full pipeline, flips and smooths included.
    Skips anything already simulated successfully; retries ones that failed before."""
    done = {r["expression"] for r in completed_history()}
    items = []
    for expression in (e.strip() for e in expressions):
        if not expression:
            continue
        if expression in done:
            logger.info("Skipping %s - it was already simulated; its result is in the dashboard.", expression)
            continue
        items.append((expression, "Requested with --try", "idea"))
    if not items:
        logger.info("Nothing new to try.")
        return []
    avoid = already_tried() - {e for e, _, _ in items}  # still avoid repeating everything else
    logger.info("Trying %d expression(s) you gave.", len(items))
    return process_queue(session, items, settings, email, password, "try", avoid)


LOCK_FILE = "alpha_loop.lock"


def acquire_run_lock(path=LOCK_FILE):
    """Hold an operating-system lock on a small file for the whole run, so a scheduled run and a manual
    run never overlap - they'd share the results file and the daily submission count. The OS drops the
    lock when the process ends, even after a crash. Returns the open file (keep it open), or None if
    another run holds it."""
    handle = open(path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def add_log_file(log_dir) -> logging.Handler:
    """Also write this run's log to <log_dir>/alpha_loop_YYYY-MM-DD.log (UTF-8, appended), for runs
    nobody is watching, such as the daily scheduled one."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / f"alpha_loop_{time.strftime('%Y-%m-%d')}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate, test and submit WorldQuant BRAIN alphas.")
    parser.add_argument("--try", dest="try_expressions", nargs="+", metavar="EXPRESSION",
                        help="simulate these exact expressions (and submit any that qualify) instead of "
                             "asking Gemini for ideas; wrap each one in double quotes")
    parser.add_argument("--log-dir", metavar="FOLDER",
                        help="also write the log to FOLDER/alpha_loop_YYYY-MM-DD.log (for scheduled runs)")
    args = parser.parse_args(argv)

    log_file = add_log_file(args.log_dir) if args.log_dir else None
    try:
        lock = acquire_run_lock()
        if lock is None:
            logger.error("Another alpha_loop run is still going (it holds %s) - not starting a second one. "
                         "Try again once it prints Done.", LOCK_FILE)
            return
        try:
            run(args)
        except LoginFailed as e:
            logger.error("Stopping: %s", e)
        except Exception:
            logger.exception("The run crashed - the details below show where.")
            raise
        finally:
            lock.close()
    finally:
        if log_file:
            logging.getLogger().removeHandler(log_file)
            log_file.close()


def run(args):
    email, password = be.get_credentials()
    if args.try_expressions:
        session = get_authenticated_session(email, password)
        submit_held(session)
        try:
            rows = run_try(session, args.try_expressions, SETTINGS, email, password)
        except StopLoop as e:
            logger.error("Stopping: %s", e)
            return
        logger.info("Done: %d simulation(s), including any automatic flips or smooths. Results are in %s.",
                    len(rows), RESULTS_FILE)
        return

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in your .env file too (same format as WQ_EMAIL/WQ_PASSWORD).")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    operators, fields = load_reference()
    if len(operators) <= 6:
        logger.warning(
            "operators.json still has the small starter set (%d operators) - BRAIN's real "
            "list has hundreds. Ideas will be limited and repetitive until you replace it "
            "with your export from platform.worldquantbrain.com/learn/operators.",
            len(operators),
        )

    settings = SETTINGS
    session = get_authenticated_session(email, password)
    submit_held(session)  # before building feedback, so the generator sees the updated outcome
    avoid = already_tried()
    try:
        repair_past_near_misses(session, settings, email, password, avoid)
    except StopLoop as e:
        logger.error("Stopping: %s", e)
        return
    except Exception:
        logger.exception("Repairing earlier near-misses hit an unexpected problem - skipping it this run.")
    feedback_text = build_feedback()
    past = len(completed_history())
    if past:
        logger.info("Starting from your best results across %d past simulations.", past)

    logger.info("Starting: up to %d rounds x %d ideas (%d max simulations, %d at a time), %ds runtime cap, "
                "%d submission(s)/day max.", MAX_ROUNDS, IDEAS_PER_ROUND, MAX_ROUNDS * IDEAS_PER_ROUND,
                PARALLEL_SIMULATIONS, MAX_RUNTIME_SECONDS, MAX_SUBMISSIONS_PER_DAY)
    type_map = field_types(fields)
    if type_map:
        logger.info("Variety: your %d fields span %d data types (%s).", len(type_map),
                    len(set(type_map.values())), ", ".join(sorted(set(type_map.values()))))
    rejected, dead = rejected_names(), dead_end_fields()
    if rejected or dead:
        logger.info("Leaving out %d name(s) BRAIN rejected (%s) and %d dead-end field(s) (%s).", len(rejected),
                    ", ".join(sorted(rejected)) or "none", len(dead), ", ".join(sorted(dead)) or "none")

    started = time.time()
    rounds_done = 0
    failures_in_a_row = 0
    while rounds_done < MAX_ROUNDS:
        if time.time() - started > MAX_RUNTIME_SECONDS:
            logger.info("Hit MAX_RUNTIME_SECONDS - stopping cleanly.")
            break
        round_num = rounds_done + 1
        try:
            round_rows = run_round(session, client, operators, fields, settings, email, password,
                                    round_num, feedback_text, avoid)
        except GenerationFailed as e:
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_GENERATION_FAILURES:
                logger.error(
                    "Idea generation failed %d times in a row (last: %s). Stopping - nothing was "
                    "simulated, so no BRAIN quota was used. Gemini is likely overloaded; try again later.",
                    failures_in_a_row, e)
                break
            logger.warning(
                "Round %d didn't start - idea generation failed (%s). Waiting %ds and trying again; "
                "this does not use up a round.", round_num, e, GENERATION_COOLDOWN_SECONDS)
            time.sleep(GENERATION_COOLDOWN_SECONDS)
            continue
        except StopLoop as e:
            logger.error("Stopping: %s", e)
            break
        failures_in_a_row = 0
        rounds_done += 1
        feedback_text = build_feedback(round_rows)
        time.sleep(5)

    logger.info("Done: %d round(s) of ideas tested. Full history is in %s.", rounds_done, RESULTS_FILE)


if __name__ == "__main__":
    main()