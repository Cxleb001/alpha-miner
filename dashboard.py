"""
dashboard.py - see what alpha_loop.py has done, and ask BRAIN what it actually recorded.

Run it from the alpha-miner folder:
    pip install streamlit
    streamlit run dashboard.py

A browser tab opens at http://localhost:8501. The page reads alpha_loop_results.csv and
submissions_log.json. It never simulates or submits anything. The one button that talks to
BRAIN only asks for the status of alphas BRAIN already has.

Why the BRAIN check matters: our log marks an alpha "Submitted" when BRAIN accepted the
submit request. That request hasn't been verified end to end yet, so the check shows BRAIN's
own record (its status, stage, submission date and any checks not passing) word for word.
"""
import csv
import json
import os
import time

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import alpha_loop as al
import brain_engine as be

DATA_DIR = Path(os.environ.get("ALPHA_MINER_DIR") or Path(__file__).resolve().parent)
RESULTS_PATH = DATA_DIR / al.RESULTS_FILE
SUBMISSION_LOG_PATH = DATA_DIR / al.SUBMISSION_LOG_FILE
SESSION_PATH = DATA_DIR / al.SESSION_FILE

SUBMITTED = "Submitted"
AWAITING = "Submitted, waiting for BRAIN's verdict"
REJECTED = "Submitted, BRAIN rejected it"
TOO_SIMILAR = "Cleared the bar, too similar to your submitted alphas"
HELD = "Cleared the bar, held by the daily limit"
SUBMIT_FAILED = "Cleared the bar, submit request failed"
NEAR_MISS = "Strong numbers, didn't pass every check"
BELOW_BAR = "Simulated, below the bar"
SIM_FAILED = "Simulation failed"
STOPPED = "Run stopped here"

# Worth asking BRAIN about: anything that cleared the bar or came close to it.
WORTH_CHECKING = (SUBMITTED, AWAITING, REJECTED, TOO_SIMILAR, HELD, SUBMIT_FAILED, NEAR_MISS)
CLEARED = (SUBMITTED, AWAITING, REJECTED, TOO_SIMILAR, HELD, SUBMIT_FAILED)
ALL_OUTCOMES = (SUBMITTED, AWAITING, REJECTED, TOO_SIMILAR, HELD, SUBMIT_FAILED, NEAR_MISS,
                BELOW_BAR, SIM_FAILED, STOPPED)


# ---- Reading the local log ----

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path=None):
    path = Path(path or RESULTS_PATH)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def outcome(row):
    submitted = (row.get("submitted") or "").strip()
    status = (row.get("status") or "").strip()
    if submitted == "YES":
        return SUBMITTED
    if submitted.startswith("VERDICT_PENDING"):
        return AWAITING
    if submitted.startswith("REJECTED"):
        return REJECTED
    if submitted.startswith("TOO_CORRELATED"):
        return TOO_SIMILAR
    if submitted.startswith("QUALIFIED_NOT_SUBMITTED"):
        return HELD
    if submitted.startswith("FAILED"):
        return SUBMIT_FAILED
    if status == "COMPLETE":
        sharpe, fitness = to_float(row.get("sharpe")), to_float(row.get("fitness"))
        if (sharpe is not None and fitness is not None
                and sharpe >= al.SUBMIT_MIN_SHARPE and fitness >= al.SUBMIT_MIN_FITNESS):
            return NEAR_MISS
        return BELOW_BAR
    if status == "STOPPED":
        return STOPPED
    return SIM_FAILED


def submissions_on(day, path=None):
    try:
        log = json.loads(Path(path or SUBMISSION_LOG_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return 0
    return log.get(str(day), 0)


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def headline(rows, today_count):
    """The one-line answer to 'did anything submit?' - shown as the page title."""
    if not rows:
        return "No results yet"
    if today_count:
        return f"{plural(today_count, 'alpha')} submitted today"
    total = sum(1 for r in rows if outcome(r) == SUBMITTED)
    if total:
        return f"{plural(total, 'alpha')} submitted so far, none today"
    return "Nothing submitted yet"


def summary(rows, today_count, cap):
    outcomes = [outcome(r) for r in rows]
    simulated = sum(1 for o in outcomes if o not in (SIM_FAILED, STOPPED))
    cleared = sum(1 for o in outcomes if o in CLEARED)
    last = max((r.get("timestamp") or "" for r in rows), default="")
    return (f"{plural(len(rows), 'idea')} tried, {simulated} simulated, {cleared} cleared the bar. "
            f"{today_count} of {cap} daily submissions used. Last activity {last or 'unknown'}.")


def table(rows, with_rationale=False):
    records = []
    for r in sorted(rows, key=lambda r: r.get("timestamp") or "", reverse=True):
        record = {
            "When": r.get("timestamp", ""),
            "Round": r.get("round", ""),
            "Outcome": outcome(r),
            "Expression": r.get("expression", ""),
            "Sharpe": to_float(r.get("sharpe")),
            "Fitness": to_float(r.get("fitness")),
            "Turnover": to_float(r.get("turnover")),
            "Checks passed": (f"{r.get('checks_passed')}/{r.get('checks_total')}"
                              if r.get("checks_total") else ""),
            "BRAIN page": r.get("alpha_link") or None,
        }
        if with_rationale:
            record["Why it was tried"] = r.get("rationale", "")
            record["Error"] = r.get("error", "")
        records.append(record)
    return pd.DataFrame(records)


# ---- Talking to BRAIN (read-only) ----

def load_saved_session():
    session = requests.Session()
    try:
        cookies = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        session.cookies.update(requests.utils.cookiejar_from_dict(cookies))
    except (FileNotFoundError, ValueError):
        pass
    return session


def save_session(session):
    SESSION_PATH.write_text(
        json.dumps(requests.utils.dict_from_cookiejar(session.cookies)), encoding="utf-8")


def session_is_valid(session):
    """Asks BRAIN how long this login has left (GET /authentication -> token.expiry, the check
    WorldQuant's ACE library uses). Asking about an alpha can't tell: logged out, BRAIN may answer
    'not found' rather than 'not logged in'."""
    try:
        resp = session.get(be.AUTH_URL, timeout=20)
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    try:
        return float(resp.json()["token"]["expiry"]) > 60
    except (ValueError, KeyError, TypeError):
        return False


def log_in(session, email, password):
    """Same login as brain_engine.authenticate, minus its input() prompt - a prompt would
    freeze the page. Returns ("ok", None), ("verify", info) or ("error", message)."""
    session.auth = (email, password)
    try:
        resp = session.post(be.AUTH_URL, timeout=30)
    except requests.RequestException as exc:
        return "error", f"Couldn't reach BRAIN to log in: {exc}"
    if resp.status_code not in (200, 201):
        return "error", (f"BRAIN refused the login ({resp.status_code}). "
                         "Check WQ_EMAIL and WQ_PASSWORD in your .env file.")
    try:
        data = resp.json()
    except ValueError:
        return "error", "BRAIN's reply to the login wasn't readable. Try again in a minute."
    if "user" in data:
        return "ok", None
    if "inquiry" in data:
        return "verify", {"url": f"{resp.url}/persona?inquiry={data['inquiry']}",
                          "post_url": f"{resp.url}/persona", "data": data}
    return "error", "BRAIN's login reply had neither an account nor a verification step."


def finish_verification(session, info):
    try:
        resp = session.post(info["post_url"], json=info["data"], timeout=30)
    except requests.RequestException:
        return False
    return resp.status_code in (200, 201)


def fetch_brain_record(session, alpha_id):
    """BRAIN's own record of one alpha, field values shown exactly as BRAIN sends them."""
    resp = session.get(be.ALPHA_URL.format(alpha_id=alpha_id), timeout=30)
    if be.looks_like_expired_session(resp):
        raise PermissionError("BRAIN session expired")
    if resp.status_code == 404:
        return {"BRAIN status": "BRAIN has no alpha with this ID", "Stage": "",
                "Submitted on": "", "Checks not passing": ""}
    try:
        alpha = resp.json()
    except ValueError:
        return {"BRAIN status": f"Unreadable reply ({resp.status_code})", "Stage": "",
                "Submitted on": "", "Checks not passing": ""}
    checks = (alpha.get("is") or {}).get("checks") or []
    not_passing = [f"{c.get('name')}: {c.get('result')}" for c in checks if c.get("result") != "PASS"]
    return {
        "BRAIN status": alpha.get("status") or "",
        "Stage": alpha.get("stage") or "",
        "Submitted on": alpha.get("dateSubmitted") or "",
        "Checks not passing": ", ".join(not_passing) or "None",
    }


def check_with_brain(rows):
    """Log in only if the saved session has expired, then fetch BRAIN's record of each alpha."""
    session = st.session_state.get("brain_session") or load_saved_session()
    try:
        valid = session_is_valid(session)
    except requests.RequestException as exc:
        st.error(f"Couldn't reach BRAIN: {exc}")
        return
    if not valid:
        try:
            email, password = be.get_credentials()
        except RuntimeError as exc:
            st.error(str(exc))
            return
        state, info = log_in(session, email, password)
        if state == "verify":
            st.session_state["brain_session"] = session
            st.session_state["verify"] = info
            return
        if state == "error":
            st.error(info)
            return
    save_session(session)
    st.session_state["brain_session"] = session
    st.session_state.pop("verify", None)

    results = []
    targets = [r for r in rows if r.get("alpha_id")]
    with st.spinner(f"Asking BRAIN about {plural(len(targets), 'alpha')}..."):
        for i, row in enumerate(targets):
            try:
                record = fetch_brain_record(session, row["alpha_id"])
            except PermissionError:
                st.warning("BRAIN ended the session partway through. Press Check with BRAIN again.")
                break
            except requests.RequestException as exc:
                record = {"BRAIN status": f"Couldn't reach BRAIN ({type(exc).__name__})",
                          "Stage": "", "Submitted on": "", "Checks not passing": ""}
            results.append({"Expression": row.get("expression", ""), "Our log says": outcome(row),
                            **record, "BRAIN page": row.get("alpha_link") or None})
            if i < len(targets) - 1:
                time.sleep(0.5)  # be gentle with BRAIN's rate limits
    st.session_state["brain_results"] = results
    st.session_state["brain_checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


# ---- The page ----

LINK = {"BRAIN page": st.column_config.LinkColumn("BRAIN page", display_text="Open")}


def render():
    st.set_page_config(page_title="Alpha Miner", layout="wide")
    rows = load_rows()
    today_count = submissions_on(al.brain_day())

    st.title(headline(rows, today_count))
    if not rows:
        st.write(f"Run `python alpha_loop.py` in `{DATA_DIR}`. "
                 "Each simulation appears here as soon as it finishes.")
        st.button("Reload results")
        return

    st.write(summary(rows, today_count, al.MAX_SUBMISSIONS_PER_DAY))
    st.caption("“Submitted” here means BRAIN accepted the submit request. "
               "Check with BRAIN shows what BRAIN itself recorded.")
    st.button("Reload results")

    worth = [r for r in rows if outcome(r) in WORTH_CHECKING]
    st.subheader("Alphas that cleared or nearly cleared the bar")
    if not worth:
        st.write(f"None yet. To clear the bar, an alpha needs every check passing, Sharpe of "
                 f"at least {al.SUBMIT_MIN_SHARPE} and fitness of at least {al.SUBMIT_MIN_FITNESS}.")
    else:
        st.dataframe(table(worth), hide_index=True, column_config=LINK)
        st.write("Ask BRAIN for its own record of these alphas. Nothing is submitted or changed.")
        if st.button("Check with BRAIN"):
            check_with_brain(worth)

        info = st.session_state.get("verify")
        if info:
            st.warning("BRAIN wants you to confirm your identity before it shares account data.")
            st.link_button("Open BRAIN's verification page", info["url"])
            if st.button("I've finished verifying"):
                if finish_verification(st.session_state["brain_session"], info):
                    st.session_state.pop("verify", None)
                    check_with_brain(worth)
                    st.rerun()  # redraw without the verification notice
                else:
                    st.error("BRAIN hasn't confirmed the verification yet. "
                             "Finish it on the page that opened, then press this again.")

        results = st.session_state.get("brain_results")
        if results:
            st.caption(f"BRAIN's record as of {st.session_state.get('brain_checked_at')}, "
                       "shown exactly as BRAIN sent it.")
            st.dataframe(pd.DataFrame(results), hide_index=True,
                         column_config=LINK)

    st.subheader("Every attempt")
    present = [o for o in ALL_OUTCOMES if any(outcome(r) == o for r in rows)]
    choice = st.selectbox("Show", ["All outcomes", *present])
    shown = rows if choice == "All outcomes" else [r for r in rows if outcome(r) == choice]
    st.dataframe(table(shown, with_rationale=True), hide_index=True, column_config=LINK)


if __name__ == "__main__":
    render()