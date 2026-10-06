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
    python alpha_loop.py

NOTE on the LLM library: this uses google-genai (import as `from google import genai`),
NOT google-generativeai (`import google.generativeai as genai`) - that package was
deprecated in August 2025 and is now archived/legacy. If you see import errors or silent
generation failures with the old package, that's why. Model defaults to gemini-3.8-flash
(current as of Sept 2026) - Google ships new Flash versions every few weeks, so this is
overridable via GEMINI_MODEL without touching code.

READ BEFORE YOU RUN THIS:
- It spends real BRAIN simulation quota and real Gemini API cost, unattended, up to the
  caps below (defaults: 5 rounds x 8 ideas = 40 simulations max per run). Raise
  IDEAS_PER_ROUND / MAX_ROUNDS via env vars once you trust it, not before.
- This version can AUTONOMOUSLY SUBMIT alphas to BRAIN (not just simulate them) when one
  clears the qualification bar below. That's a bigger commitment than testing - submission
  is rate-limited to MAX_SUBMISSIONS_PER_DAY (default 1, matching what BRAIN itself appears
  to expect) regardless of how many alphas qualify in a run. The qualification thresholds
  (checks all pass, sharpe >= 1.25, fitness >= 1.0) are what was in the version you sent me -
  I haven't independently verified those are the right bar for a real submission decision,
  that's a domain call for you to make, not something I'm endorsing by leaving it as-is.
- operators.json ships with a 5-operator starter set. BRAIN has hundreds. Until you
  replace it with your real export from platform.worldquantbrain.com/learn/operators,
  the generator can only reshuffle those 5 - swap the file's contents, no code changes
  needed.
- "Unattended" has a real ceiling: BRAIN sessions expire (roughly 4 hours) and can
  require a one-time identity check to renew. This script saves its session to disk
  so a re-run doesn't need to log in again unnecessarily, but if BRAIN asks for
  verification mid-run with nobody there to click the link, it STOPS CLEANLY and
  logs why. It does not hang waiting for input that will never come.
"""
import csv
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

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
        probe = session.get(be.ALPHA_URL.format(alpha_id="session_check"))
        if not be.looks_like_expired_session(probe):
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


def submissions_today() -> int:
    if not Path(SUBMISSION_LOG_FILE).exists():
        return 0
    try:
        data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
    except Exception:
        return 0
    return data.get(str(date.today()), 0)


def record_submission():
    data = {}
    if Path(SUBMISSION_LOG_FILE).exists():
        try:
            data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
        except Exception:
            data = {}
    key = str(date.today())
    data[key] = data.get(key, 0) + 1
    Path(SUBMISSION_LOG_FILE).write_text(json.dumps(data), encoding="utf-8")


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
            f"turnover={r['turnover']} checks={r['checks_passed']}/{r['checks_total']} submitted={r.get('submitted', 'NO')}"
        )
    failed = [r for r in rows if r["status"] != "COMPLETE"]
    if failed:
        lines.append(f"{len(failed)} other attempts failed outright (bad expression, timeout, etc) - avoid whatever pattern caused that.")
    return "\n".join(lines)


def generate_ideas(client: genai.Client, operators, fields, settings, feedback_text, avoid: set, n: int):
    op_list = "\n".join(f"- {o['name']}: {o['usage']}" for o in operators)
    field_list = "\n".join(f"- {f['name']}: {f['description']}" for f in fields)
    avoid_list = "\n".join(f"- {e}" for e in list(avoid)[:50]) or "(none yet)"

    prompt = (
        f"You are proposing WorldQuant BRAIN FASTEXPR alpha expressions for "
        f"region={settings['region']} universe={settings['universe']} delay={settings['delay']}.\n\n"
        f"ONLY use these operators, exactly as named (do not invent others):\n{op_list}\n\n"
        f"ONLY use these data fields, exactly as named (do not invent others):\n{field_list}\n\n"
        f"{feedback_text}\n\n"
        f"Propose {n} NEW candidate expressions, genuinely different in shape from each "
        f"other (not the same template with one field swapped). Each needs a one-sentence "
        f"economic rationale for why it might predict returns.\n\n"
        f"Do not repeat any of these already-tried expressions:\n{avoid_list}"
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AlphaIdeas,
        ),
    )

    if not resp.text:
        raise ValueError("Model returned an empty response - possibly blocked by a safety filter.")

    data = json.loads(resp.text)
    return [(idea["expression"], idea["rationale"]) for idea in data["ideas"]]


def submit_alpha(session, alpha_id: str):
    """Triggers official alpha submission on WorldQuant BRAIN.
    NOTE: this endpoint hasn't been independently verified the way the others in
    brain_engine.py have been - treat the first real call as a test, not a certainty."""
    url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/submit"
    resp = session.post(url)

    if resp.status_code in (200, 201):
        logger.info("SUCCESSFULLY SUBMITTED Alpha ID: %s", alpha_id)
        return True, None
    logger.error("Failed to submit Alpha ID %s: %d - %s", alpha_id, resp.status_code, resp.text)
    return False, resp.text


def append_result(row: dict):
    is_new = not Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_round(session, client, operators, fields, settings, email, password, round_num, feedback_text, avoid):
    logger.info("=== Round %d: generating %d ideas ===", round_num, IDEAS_PER_ROUND)
    try:
        ideas = generate_ideas(client, operators, fields, settings, feedback_text, avoid, IDEAS_PER_ROUND)
    except Exception as e:
        logger.error("Idea generation failed this round: %s", e)
        return []

    round_rows = []
    for expression, rationale in ideas:
        if expression in avoid:
            logger.info("Skipping duplicate: %s", expression)
            continue
        avoid.add(expression)

        row = {k: "" for k in RESULTS_HEADER}
        row.update(round=round_num, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                   expression=expression, rationale=rationale)

        location, err = be.submit_simulation(session, expression, settings)
        if err == "EXPIRED_SESSION":
            logger.warning("Session expired mid-run - attempting one re-auth...")
            session.cookies.clear()
            if not be.authenticate(session, email, password):
                row["status"], row["error"], row["submitted"] = "STOPPED", "session expired and could not renew unattended", "NO"
                append_result(row)
                round_rows.append(row)
                raise StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to complete it.")
            save_session(session)
            location, err = be.submit_simulation(session, expression, settings)

        if err:
            row["status"], row["error"], row["submitted"] = "FAILED", str(err)[:300], "NO"
            append_result(row)
            round_rows.append(row)
            continue

        alpha_id, err = be.poll_simulation(session, location)
        if err:
            row["status"], row["error"], row["submitted"] = "FAILED", str(err)[:300], "NO"
            append_result(row)
            round_rows.append(row)
            continue

        metrics = be.fetch_alpha_metrics(session, alpha_id)
        row.update(status="COMPLETE", alpha_id=alpha_id, **metrics)

        # --- Autonomous submission, rate-limited ---
        checks_passed = safe_float(metrics.get("checks_passed"))
        checks_total = safe_float(metrics.get("checks_total"))
        sharpe = safe_float(metrics.get("sharpe"))
        fitness = safe_float(metrics.get("fitness"))
        qualifies = checks_total > 0 and checks_passed == checks_total and sharpe >= 1.25 and fitness >= 1.0

        if not qualifies:
            row["submitted"] = "NO"
        elif submissions_today() >= MAX_SUBMISSIONS_PER_DAY:
            logger.warning(
                "Alpha %s qualifies but the daily submission cap (%d) is already reached - "
                "logging as qualified, NOT auto-submitting.", alpha_id, MAX_SUBMISSIONS_PER_DAY,
            )
            row["submitted"] = "QUALIFIED_NOT_SUBMITTED (daily cap reached)"
        else:
            logger.info("Alpha %s qualified for submission! Submitting...", alpha_id)
            submitted, sub_err = submit_alpha(session, alpha_id)
            if submitted:
                record_submission()
                row["submitted"] = "YES"
            else:
                row["submitted"] = f"FAILED: {sub_err}"

        append_result(row)
        round_rows.append(row)
        logger.info("  %s -> sharpe=%s fitness=%s checks=%s/%s submitted=%s", expression,
                    metrics.get("sharpe"), metrics.get("fitness"),
                    metrics.get("checks_passed"), metrics.get("checks_total"), row["submitted"])
        time.sleep(3)

    return round_rows


def main():
    email, password = be.get_credentials()
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

    settings = {
        "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
        "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
        "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "ON",
        "language": "FASTEXPR", "visualization": False,
    }

    session = get_authenticated_session(email, password)
    avoid = already_tried()
    feedback_text = "This is round 1 - no prior results yet."

    logger.info("Starting: up to %d rounds x %d ideas (%d max simulations), %ds runtime cap, "
                "%d submission(s)/day max.", MAX_ROUNDS, IDEAS_PER_ROUND,
                MAX_ROUNDS * IDEAS_PER_ROUND, MAX_RUNTIME_SECONDS, MAX_SUBMISSIONS_PER_DAY)

    started = time.time()
    for round_num in range(1, MAX_ROUNDS + 1):
        if time.time() - started > MAX_RUNTIME_SECONDS:
            logger.info("Hit MAX_RUNTIME_SECONDS - stopping cleanly.")
            break
        try:
            round_rows = run_round(session, client, operators, fields, settings, email, password,
                                    round_num, feedback_text, avoid)
        except StopLoop as e:
            logger.error("Stopping: %s", e)
            break
        feedback_text = summarize_round(round_rows)
        time.sleep(5)

    logger.info("Done. Full history (every round, every attempt) is in %s.", RESULTS_FILE)


if __name__ == "__main__":
    main()
