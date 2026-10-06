import csv
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the same folder and loads it into os.environ

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

AUTH_URL = "https://api.worldquantbrain.com/authentication"
SIM_URL = "https://api.worldquantbrain.com/simulations"
ALPHA_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}"

RESULTS_FILE = "results.csv"
RESULTS_HEADER = [
    "timestamp", "expression", "status", "sharpe", "fitness", "turnover",
    "checks_passed", "checks_total", "alpha_id", "alpha_link", "error",
]

POLL_INTERVAL_SECONDS = 5
MAX_POLL_SECONDS = 20 * 60  # bail out rather than hang forever on a stuck sim - adjust once you know typical runtimes


def get_credentials():
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Set WQ_EMAIL and WQ_PASSWORD as environment variables before running this "
            "script - never hardcode BRAIN credentials in source."
        )
    return email, password


def authenticate(session: requests.Session, email: str, password: str) -> bool:
    """
    Logs in and returns True only once the session is actually usable. BRAIN can
    require a one-time identity check ("Persona") on top of the password, so a 2xx
    response here doesn't necessarily mean you're logged in yet - this checks the
    response body, not just the status code, and walks you through verification
    interactively if it's required.
    """
    session.auth = (email, password)
    try:
        resp = session.post(AUTH_URL)
    except requests.RequestException as e:
        logger.error("Could not reach BRAIN's auth endpoint: %s", e)
        return False

    if resp.status_code not in (200, 201):
        logger.error("Auth failed (%s): %s", resp.status_code, resp.text[:300])
        return False

    try:
        data = resp.json()
    except ValueError:
        logger.error("Auth response wasn't JSON: %s", resp.text[:300])
        return False

    if "user" in data:
        logger.info("Authenticated as %s.", email)
        return True

    if "inquiry" in data:
        verify_url = f"{resp.url}/persona?inquiry={data['inquiry']}"
        input(
            f"\nBRAIN wants identity verification for this login.\n"
            f"Open this URL, complete it, then press Enter here to continue:\n{verify_url}\n"
        )
        follow_up = session.post(f"{resp.url}/persona", json=data)
        if follow_up.status_code in (200, 201):
            logger.info("Verification complete - authenticated as %s.", email)
            return True
        logger.error("Verification didn't complete: %s", follow_up.text[:300])
        return False

    logger.error("Unexpected auth response (no 'user' or 'inquiry' field): %s", data)
    return False


def looks_like_expired_session(resp) -> bool:
    if resp.status_code == 401:
        return True
    try:
        detail = str(resp.json().get("detail", "")).lower()
    except ValueError:
        return False
    return "credentials" in detail


def submit_simulation(session: requests.Session, expression: str, settings: dict):
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    resp = session.post(SIM_URL, json=payload)

    if resp.status_code == 201 and "Location" in resp.headers:
        return resp.headers["Location"], None
    if looks_like_expired_session(resp):
        return None, "EXPIRED_SESSION"
    return None, f"submit failed ({resp.status_code}): {resp.text[:300]}"


def poll_simulation(session: requests.Session, location: str):
    """
    Polls until the simulation resolves, times out, or errors. Different BRAIN
    client implementations disagree on the exact response shape (a top-level
    'status' field vs. an 'alpha'/'progress' field) - rather than bet on one,
    this checks for both so it can't hang silently on a shape it doesn't recognise.
    """
    started = time.time()
    while time.time() - started < MAX_POLL_SECONDS:
        resp = session.get(location)
        if looks_like_expired_session(resp):
            return None, "EXPIRED_SESSION"
        try:
            data = resp.json()
        except ValueError:
            return None, f"non-JSON poll response ({resp.status_code}): {resp.text[:200]}"

        status = data.get("status")
        alpha_id = data.get("alpha")
        progress = data.get("progress")

        if status == "COMPLETE" or alpha_id:
            if alpha_id:
                return alpha_id, None
            return None, f"marked complete but no alpha id in response: {data}"

        if status == "ERROR" or (progress is None and "message" in data):
            return None, data.get("message", data)

        elapsed = int(time.time() - started)
        if progress is not None:
            logger.info("  ... %d%% (%ds elapsed)", int(progress * 100), elapsed)
        else:
            logger.info("  ... still running (%ds elapsed)", elapsed)
        time.sleep(POLL_INTERVAL_SECONDS)

    return None, f"timed out after {MAX_POLL_SECONDS}s"


def fetch_alpha(session: requests.Session, alpha_id: str) -> dict:
    """BRAIN's full record of one alpha, or {} if the reply wasn't readable."""
    resp = session.get(ALPHA_URL.format(alpha_id=alpha_id))
    try:
        alpha = resp.json()
    except ValueError:
        return {}
    return alpha if isinstance(alpha, dict) else {}


def fetch_alpha_metrics(session: requests.Session, alpha_id: str) -> dict:
    return metrics_from_alpha(alpha_id, fetch_alpha(session, alpha_id))


def metrics_from_alpha(alpha_id: str, alpha: dict) -> dict:
    if not alpha:
        return {}
    is_stats = alpha.get("is", {})
    checks = is_stats.get("checks", [])
    return {
        "sharpe": is_stats.get("sharpe"),
        "fitness": is_stats.get("fitness"),
        "turnover": is_stats.get("turnover"),
        "checks_passed": sum(1 for c in checks if c.get("result") == "PASS"),
        "checks_total": len(checks),
        "alpha_link": f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
    }


def append_result(row: dict):
    is_new = not Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_one(session: requests.Session, expression: str, settings: dict, email: str, password: str):
    logger.info("Submitting: %s", expression)
    row = {key: "" for key in RESULTS_HEADER}
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["expression"] = expression

    location, err = submit_simulation(session, expression, settings)
    if err == "EXPIRED_SESSION":
        logger.warning("Session expired mid-run - re-authenticating once...")
        if not authenticate(session, email, password):
            row["status"], row["error"] = "FAILED", "re-auth failed"
            append_result(row)
            return
        location, err = submit_simulation(session, expression, settings)

    if err:
        logger.error("  submission error: %s", err)
        row["status"], row["error"] = "FAILED", str(err)[:300]
        append_result(row)
        return

    alpha_id, err = poll_simulation(session, location)
    if err:
        logger.error("  simulation error: %s", err)
        row["status"], row["error"] = "FAILED", str(err)[:300]
        append_result(row)
        return

    metrics = fetch_alpha_metrics(session, alpha_id)
    row.update({"status": "COMPLETE", "alpha_id": alpha_id, **metrics})
    append_result(row)
    logger.info(
        "  done - sharpe=%s fitness=%s turnover=%s checks=%s/%s",
        metrics.get("sharpe"), metrics.get("fitness"), metrics.get("turnover"),
        metrics.get("checks_passed"), metrics.get("checks_total"),
    )


def main():
    email, password = get_credentials()

    settings = {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "ON",
        "language": "FASTEXPR",
        "visualization": False,
    }

    # Your Day-1 template, crossed against a handful of fundamental fields.
    data_fields = [
        "operating_income",
        "sales",
        "assets",
        "free_cash_flow",
        "retained_earnings",
        "net_income",
    ]

    session = requests.Session()
    logger.info("Authenticating...")
    if not authenticate(session, email, password):
        logger.error("Could not authenticate - stopping before burning any simulation quota.")
        sys.exit(1)

    logger.info("Starting batch of %d expressions. Results streaming to %s", len(data_fields), RESULTS_FILE)
    for field in data_fields:
        expression = f"group_neutralize(ts_rank({field}, 252), industry) * ts_rank(volume, 20)"
        try:
            run_one(session, expression, settings, email, password)
        except Exception:
            logger.exception("Unexpected failure on %s - skipping to next field", field)
        time.sleep(3)

    logger.info("Batch complete. See %s for the full ranked results.", RESULTS_FILE)


if __name__ == "__main__":
    main()