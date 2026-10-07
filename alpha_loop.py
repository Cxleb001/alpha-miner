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
- Auto-flip: when an alpha scores Sharpe -1.0 or worse, the loop also tests reverse(<expression>)
  right after it (up to MAX_FLIPS_PER_ROUND per round, default 3), since a strongly negative signal
  usually turns positive when flipped. Flips use extra simulations.
- Auto-smooth: when an alpha has Sharpe >= 1.25 but fitness under 1.0 (or turnover over 70%) and
  turnover above SMOOTH_MIN_TURNOVER (default 0.2), the loop also tests
  ts_decay_linear(<expression>, 10) right after it (up to MAX_SMOOTHS_PER_ROUND per round, default 3).
  Fitness divides by turnover, so trading less can lift it over the bar.
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
import os
import time
from collections import deque
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
MAX_SMOOTHS_PER_ROUND = int(os.environ.get("MAX_SMOOTHS_PER_ROUND", 3))  # caps the extra simulations smoothing adds

# Network resilience: one dropped connection skips that idea; this many in a row stops the run cleanly.
MAX_NETWORK_FAILURES = int(os.environ.get("MAX_NETWORK_FAILURES", 3))

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
SUBMISSION_LOG_FILE = "submissions_log.json"  # {"YYYY-MM-DD": count} - enforces the daily cap across runs

RESULTS_HEADER = [
    "round", "timestamp", "expression", "rationale", "status", "sharpe", "fitness",
    "turnover", "checks_passed", "checks_total", "alpha_id", "alpha_link", "error", "submitted",
]


class StopLoop(Exception):
    """Raised to unwind out of a round immediately - e.g. session can't be renewed unattended."""


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


def get_authenticated_session(email: str, password: str) -> requests.Session:
    session = requests.Session()
    if load_session(session):
        try:
            probe = session.get(be.ALPHA_URL.format(alpha_id="session_check"), timeout=be.REQUEST_TIMEOUT)
        except requests.RequestException:
            probe = None
        if probe is not None and not be.looks_like_expired_session(probe):
            logger.info("Reused saved session - skipped login.")
            return session
        logger.info("Saved session no longer valid, logging in fresh.")
        session.cookies.clear()

    if not be.authenticate(session, email, password):
        raise RuntimeError("Could not authenticate - stopping before burning any quota.")
    save_session(session)
    return session


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


def build_feedback(last_round_rows=None) -> str:
    """What the generator sees each round: your best results across ALL runs so far, so a new run
    builds on what already worked instead of starting from scratch, plus how the last round went."""
    history = completed_history()
    parts = []
    if history:
        best = sorted(history, key=lambda r: as_float(r["fitness"]), reverse=True)[:MEMORY_TOP_N]
        lines = [f"Your best results so far, out of {len(history)} simulated alphas across all runs "
                 f"(best fitness first). Build on what works here - keep the data and the idea, fix what's "
                 f"holding it back (usually fitness or turnover) - but keep some ideas genuinely new:"]
        for r in best:
            lines.append(f"- {r['expression']} -> sharpe={r['sharpe']} fitness={r['fitness']} "
                         f"turnover={r['turnover']}{turnover_note(r)}")
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


def try_submit(session, alpha_id: str) -> str:
    """Pre-check self-correlation, submit, wait for the verdict. Returns the CSV 'submitted' value."""
    corr = self_correlation_max(session, alpha_id)
    if corr is not None and corr >= MAX_SELF_CORRELATION:
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
    """A good Sharpe held back by trading too much: fitness under the bar, or turnover over BRAIN's
    limit. Fitness divides by turnover (floored at 0.125), so decay helps most when turnover is high."""
    if sharpe is None or fitness is None or turnover is None:
        return False
    if sharpe < SUBMIT_MIN_SHARPE or turnover <= SMOOTH_MIN_TURNOVER:
        return False
    return fitness < SUBMIT_MIN_FITNESS or turnover > TURNOVER_MAX


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


def evaluate_idea(session, expression, rationale, settings, email, password, round_num) -> dict:
    """Simulate one expression, log it, and submit it if it qualifies. Returns its row."""
    row = {k: "" for k in RESULTS_HEADER}
    row.update(round=round_num, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               expression=expression, rationale=rationale, submitted="NO")

    location, err = be.submit_simulation(session, expression, settings)
    if err == "EXPIRED_SESSION":
        logger.warning("Session expired mid-run - attempting one re-auth...")
        session.cookies.clear()
        if not be.authenticate(session, email, password):
            row["status"], row["error"] = "STOPPED", "session expired and could not renew unattended"
            append_result(row)
            raise StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to complete it.")
        save_session(session)
        location, err = be.submit_simulation(session, expression, settings)

    if err:
        row["status"], row["error"] = "FAILED", str(err)[:300]
        append_result(row)
        return row

    alpha_id, err = be.poll_simulation(session, location)
    if err:
        row["status"], row["error"] = "FAILED", str(err)[:300]
        append_result(row)
        return row

    alpha = be.fetch_alpha(session, alpha_id)
    metrics = be.metrics_from_alpha(alpha_id, alpha)
    checks = (alpha.get("is") or {}).get("checks") or []
    row.update(status="COMPLETE", alpha_id=alpha_id, **metrics)
    # Kept in memory for the next round's feedback; the CSV doesn't store it.
    row["failing_checks"] = ", ".join(f"{c.get('name')}:{c.get('result')}" for c in checks if c.get("result") != "PASS")

    if qualifies(metrics, checks):
        if submissions_today() >= MAX_SUBMISSIONS_PER_DAY:
            logger.warning("Alpha %s qualifies but today's submission limit (%d) is used - logging it, "
                           "not submitting.", alpha_id, MAX_SUBMISSIONS_PER_DAY)
            row["submitted"] = "QUALIFIED_NOT_SUBMITTED (daily cap reached)"
        else:
            row["submitted"] = try_submit(session, alpha_id)

    append_result(row)
    logger.info("  %s -> sharpe=%s fitness=%s turnover=%s checks=%s/%s submitted=%s", expression,
                metrics.get("sharpe"), metrics.get("fitness"), metrics.get("turnover"),
                metrics.get("checks_passed"), metrics.get("checks_total"), row["submitted"])
    return row


def run_round(session, client, operators, fields, settings, email, password, round_num, feedback_text, avoid):
    logger.info("=== Round %d: generating %d ideas ===", round_num, IDEAS_PER_ROUND)
    try:
        ideas = generate_with_fallback(client, operators, fields, settings, feedback_text, avoid, IDEAS_PER_ROUND)
    except Exception as e:
        raise GenerationFailed(short_error(e)) from e

    return process_queue(session, [(e, r, "idea") for e, r in ideas], settings, email, password, round_num, avoid)


def process_queue(session, items, settings, email, password, round_num, avoid) -> list:
    """Test each (expression, rationale, kind) in order. A flip or smooth goes to the FRONT of
    the queue, so it runs right after the alpha that triggered it - even a short or interrupted
    run tests its best leads first."""
    queue = deque(items)
    round_rows, flips, smooths, network_failures = [], 0, 0, 0
    while queue:
        expression, rationale, kind = queue.popleft()
        if expression in avoid:
            logger.info("Skipping duplicate: %s", expression)
            continue
        avoid.add(expression)
        try:
            row = evaluate_idea(session, expression, rationale, settings, email, password, round_num)
            network_failures = 0
        except StopLoop:
            raise
        except requests.RequestException as exc:
            network_failures += 1
            logger.warning("Couldn't reach BRAIN while testing %s (%s), even after retrying - skipping it. "
                           "Retry it later with --try.", expression, type(exc).__name__)
            round_rows.append(failure_row(expression, rationale, round_num,
                                          f"Couldn't reach BRAIN ({type(exc).__name__}) - not a verdict on the "
                                          f"idea; retry it with --try"))
            if network_failures >= MAX_NETWORK_FAILURES:
                raise StopLoop(f"Couldn't reach BRAIN for {network_failures} ideas in a row - your connection "
                               f"seems to be down. Stopped cleanly; everything so far is saved.")
            continue
        except Exception as exc:  # an unexpected reply shouldn't end the whole run
            logger.exception("Unexpected error while testing %s - skipping it and carrying on.", expression)
            round_rows.append(failure_row(expression, rationale, round_num,
                                          f"Unexpected error: {type(exc).__name__}: {exc}"))
            continue
        round_rows.append(row)

        if row["status"] == "COMPLETE":
            sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))

            # Auto-flip: a strongly negative Sharpe means a real signal pointing the wrong way.
            if (kind == "idea" and sharpe is not None and sharpe <= FLIP_SHARPE_AT_OR_BELOW
                    and flips < MAX_FLIPS_PER_ROUND and not expression.startswith("reverse(")):
                flipped = f"reverse({expression})"
                if flipped not in avoid:
                    flips += 1
                    logger.info("  Sharpe %.2f points the wrong way - testing the flipped version next.", sharpe)
                    queue.appendleft((flipped, f"Auto-flip of {expression}, which scored Sharpe {sharpe:.2f}; "
                                               f"reversed, it should score about {-sharpe:.2f}.", "flip"))

            # Auto-smooth: the Sharpe is there, but trading too much is holding it back.
            if kind != "smooth" and smooths < MAX_SMOOTHS_PER_ROUND and needs_smoothing(sharpe, fitness, turnover):
                smoothed = f"ts_decay_linear({expression}, {SMOOTH_DECAY_DAYS})"
                if smoothed not in avoid:
                    smooths += 1
                    logger.info("  Sharpe %.2f is there, but fitness %.2f at %.0f%% turnover isn't - "
                                "testing a smoothed version next.", sharpe, fitness, turnover * 100)
                    queue.appendleft((smoothed, f"Auto-smooth of {expression}: Sharpe {sharpe:.2f} but fitness "
                                                f"{fitness:.2f} at {turnover:.0%} turnover; decaying it over "
                                                f"{SMOOTH_DECAY_DAYS} days should cut turnover.", "smooth"))
        time.sleep(3)

    return round_rows


def failure_row(expression, rationale, round_num, error) -> dict:
    row = {k: "" for k in RESULTS_HEADER}
    row.update(round=round_num, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), expression=expression,
               rationale=rationale, status="FAILED", error=str(error)[:300], submitted="NO")
    append_result(row)
    return row


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate, test and submit WorldQuant BRAIN alphas.")
    parser.add_argument("--try", dest="try_expressions", nargs="+", metavar="EXPRESSION",
                        help="simulate these exact expressions (and submit any that qualify) instead of "
                             "asking Gemini for ideas; wrap each one in double quotes")
    args = parser.parse_args(argv)

    email, password = be.get_credentials()
    if args.try_expressions:
        session = get_authenticated_session(email, password)
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
    avoid = already_tried()
    feedback_text = build_feedback()
    past = len(completed_history())
    if past:
        logger.info("Starting from your best results across %d past simulations.", past)

    logger.info("Starting: up to %d rounds x %d ideas (%d max simulations), %ds runtime cap, "
                "%d submission(s)/day max.", MAX_ROUNDS, IDEAS_PER_ROUND,
                MAX_ROUNDS * IDEAS_PER_ROUND, MAX_RUNTIME_SECONDS, MAX_SUBMISSIONS_PER_DAY)

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