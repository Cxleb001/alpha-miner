"""Tests for dashboard.py. Run: python test_dashboard.py  (no network - BRAIN is mocked)."""
import csv
import json
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Tests never read your real .env or credentials: python-dotenv is switched off before brain_engine
# loads, and any miner setting already in the environment is cleared, so every test sees the defaults
# it was written for - whatever your .env sets (MAX_SUBMISSIONS_PER_DAY, GEMINI_MODEL, ...).
import re as _re
import dotenv as _dotenv
_dotenv.load_dotenv = lambda *args, **kwargs: False
for _setting in _re.findall(r'os\.environ\.get\("([A-Z0-9_]+)"',
                            (Path(__file__).resolve().parent / "alpha_loop.py").read_text(encoding="utf-8")):
    os.environ.pop(_setting, None)
os.environ.update(WQ_EMAIL="e@x.com", WQ_PASSWORD="pw", GEMINI_API_KEY="fake-test-key")

import alpha_loop as al
import brain_engine as be
import dashboard as db
from streamlit.testing.v1 import AppTest


def row(**kw):
    base = {k: "" for k in al.RESULTS_HEADER}
    base.update(kw)
    return base


SAMPLE = [
    row(round="1", timestamp="2026-10-06 21:10:00", expression="rank(close)", status="COMPLETE",
        sharpe="1.6", fitness="1.3", turnover="0.2", checks_passed="8", checks_total="8",
        alpha_id="aid_sub", alpha_link="https://platform.worldquantbrain.com/alpha/aid_sub", submitted="YES"),
    row(round="1", timestamp="2026-10-06 21:20:00", expression="rank(volume)", status="COMPLETE",
        sharpe="1.5", fitness="1.2", turnover="0.2", checks_passed="8", checks_total="8",
        alpha_id="aid_held", alpha_link="https://platform.worldquantbrain.com/alpha/aid_held",
        submitted="QUALIFIED_NOT_SUBMITTED (daily cap reached)"),
    row(round="2", timestamp="2026-10-06 21:30:00", expression="rank(sales)", status="COMPLETE",
        sharpe="1.4", fitness="1.1", turnover="0.3", checks_passed="7", checks_total="8",
        alpha_id="aid_near", alpha_link="https://platform.worldquantbrain.com/alpha/aid_near", submitted="NO"),
    row(round="2", timestamp="2026-10-06 21:40:00", expression="rank(assets)", status="COMPLETE",
        sharpe="0.4", fitness="0.2", turnover="0.1", checks_passed="5", checks_total="8",
        alpha_id="aid_low", submitted="NO"),
    row(round="2", timestamp="2026-10-06 21:50:00", expression="rank(bad(", status="FAILED",
        error="submit failed (400): bad expression", submitted="NO"),
]


def write_data(folder, rows=SAMPLE, today_count=1):
    folder = Path(folder)
    with open(folder / al.RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=al.RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    (folder / al.SUBMISSION_LOG_FILE).write_text(json.dumps({str(al.brain_day()): today_count}))


def response(status, payload=None, url=""):
    r = MagicMock()
    r.status_code = status
    r.url = url
    r.json.return_value = payload if payload is not None else {}
    return r


def app(folder):
    os.environ["ALPHA_MINER_DIR"] = str(folder)
    return AppTest.from_file(str(HERE / "dashboard.py"), default_timeout=30)


def press(at, label):
    return next(b for b in at.button if b.label == label).click().run()


# ---- Pure logic ----

def test_outcome_covers_every_case():
    assert [db.outcome(r) for r in SAMPLE] == [db.SUBMITTED, db.HELD, db.NEAR_MISS, db.BELOW_BAR, db.SIM_FAILED]
    assert db.outcome(row(submitted="FAILED: 500 oops", status="COMPLETE")) == db.SUBMIT_FAILED
    assert db.outcome(row(status="STOPPED", submitted="NO")) == db.STOPPED


def test_near_miss_uses_the_same_bar_as_alpha_loop():
    just_under = row(status="COMPLETE", sharpe=str(al.SUBMIT_MIN_SHARPE - 0.01), fitness="2", submitted="NO")
    at_bar = row(status="COMPLETE", sharpe=str(al.SUBMIT_MIN_SHARPE), fitness=str(al.SUBMIT_MIN_FITNESS), submitted="NO")
    assert db.outcome(just_under) == db.BELOW_BAR
    assert db.outcome(at_bar) == db.NEAR_MISS


def test_headline_wording():
    assert db.headline([], 0) == "No results yet"
    assert db.headline(SAMPLE, 1) == "1 alpha submitted today"
    assert db.headline(SAMPLE, 2) == "2 alphas submitted today"
    assert db.headline(SAMPLE, 0) == "1 alpha submitted so far, none today"
    assert db.headline(SAMPLE[2:], 0) == "Nothing submitted yet"


def test_summary_counts():
    text = db.summary(SAMPLE, 1, 1)
    assert text.startswith("5 ideas tried, 4 simulated, 2 cleared the bar.")
    assert "1 of 1 daily submissions used" in text
    assert "Last activity 2026-10-06 21:50:00" in text


def test_missing_files_are_handled():
    with tempfile.TemporaryDirectory() as d:
        assert db.load_rows(Path(d) / "nope.csv") == []
        assert db.submissions_on(date.today(), Path(d) / "nope.json") == 0


def test_fetch_brain_record_reports_brains_own_words():
    session = MagicMock()
    session.get.return_value = response(200, {
        "status": "ACTIVE", "stage": "OS", "dateSubmitted": "2026-10-06T21:12:00",
        "is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS"},
                          {"name": "SELF_CORRELATION", "result": "PENDING"}]}})
    rec = db.fetch_brain_record(session, "aid_sub")
    assert rec == {"BRAIN status": "ACTIVE", "Stage": "OS", "Submitted on": "2026-10-06T21:12:00",
                   "Checks not passing": "SELF_CORRELATION: PENDING"}


def test_fetch_brain_record_not_found_and_expired():
    session = MagicMock()
    session.get.return_value = response(404, {"detail": "Not found."})
    assert "no alpha" in db.fetch_brain_record(session, "x")["BRAIN status"]
    session.get.return_value = response(401, {"detail": "Incorrect authentication credentials."})
    try:
        db.fetch_brain_record(session, "x")
        assert False, "should have raised"
    except PermissionError:
        pass


def test_log_in_never_prompts():
    session = MagicMock()
    session.post.return_value = response(201, {"user": {"id": "u"}})
    assert db.log_in(session, "e", "p") == ("ok", None)

    session.post.return_value = response(201, {"inquiry": "inq1"}, url=be.AUTH_URL)
    with patch("builtins.input", side_effect=AssertionError("must never prompt")):
        state, info = db.log_in(session, "e", "p")
    assert state == "verify" and info["url"].endswith("/persona?inquiry=inq1")

    session.post.return_value = response(401, {"detail": "bad"})
    state, message = db.log_in(session, "e", "p")
    assert state == "error" and "WQ_PASSWORD" in message


def test_new_submit_outcomes_are_recognised_and_counted_as_cleared():
    rows = [row(status="COMPLETE", sharpe="1.6", fitness="1.2", submitted="VERDICT_PENDING (check with BRAIN in the dashboard)"),
            row(status="COMPLETE", sharpe="1.6", fitness="1.2", submitted="REJECTED: failed SELF_CORRELATION"),
            row(status="COMPLETE", sharpe="1.6", fitness="1.2", submitted="TOO_CORRELATED (max 0.83)")]
    assert [db.outcome(r) for r in rows] == [db.AWAITING, db.REJECTED, db.TOO_SIMILAR]
    assert all(db.outcome(r) in db.WORTH_CHECKING for r in rows)
    assert "3 cleared the bar" in db.summary(rows, 0, 1)


# ---- The page itself (rendered by Streamlit's own test harness) ----

def test_page_with_no_results():
    with tempfile.TemporaryDirectory() as d:
        at = app(d).run()
        assert not at.exception
        assert at.title[0].value == "No results yet"
        assert "python alpha_loop.py" in at.markdown[0].value


def test_page_with_results():
    with tempfile.TemporaryDirectory() as d:
        write_data(d)
        at = app(d).run()
        assert not at.exception
        assert at.title[0].value == "1 alpha submitted today"
        assert len(at.dataframe) == 2
        worth = at.dataframe[0].value
        assert list(worth["Outcome"]) == [db.NEAR_MISS, db.HELD, db.SUBMITTED]  # newest first
        assert len(at.dataframe[1].value) == 5
        at = at.selectbox[0].select(db.SUBMITTED).run()   # filter down to one outcome
        assert list(at.dataframe[1].value["Expression"]) == ["rank(close)"]


def test_check_with_brain_shows_brains_record():
    records = {
        "aid_sub": {"status": "ACTIVE", "stage": "OS", "dateSubmitted": "2026-10-06T21:12:00",
                    "is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS"}]}},
        "aid_held": {"status": "UNSUBMITTED", "stage": "IS",
                     "is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS"}]}},
        "aid_near": {"status": "UNSUBMITTED", "stage": "IS",
                     "is": {"checks": [{"name": "SELF_CORRELATION", "result": "PENDING"}]}},
    }

    def fake_get(self, url, **kw):
        if url == be.AUTH_URL:   # logged in, with plenty of time left
            return response(200, {"token": {"expiry": 9000}})
        return response(200, records[url.rsplit("/", 1)[-1]])

    with tempfile.TemporaryDirectory() as d:
        write_data(d)
        with patch("requests.Session.get", new=fake_get), \
             patch("requests.Session.post", side_effect=AssertionError("no login or submit expected")):
            at = press(app(d).run(), "Check with BRAIN")
        assert not at.exception
        brain = at.dataframe[1].value
        by_expr = dict(zip(brain["Expression"], brain["BRAIN status"]))
        assert by_expr == {"rank(sales)": "UNSUBMITTED", "rank(volume)": "UNSUBMITTED", "rank(close)": "ACTIVE"}
        near = brain[brain["Expression"] == "rank(sales)"].iloc[0]
        assert near["Checks not passing"] == "SELF_CORRELATION: PENDING"
        assert (Path(d) / al.SESSION_FILE).exists()  # session saved for next time


def test_identity_verification_flow():
    state = {"verified": False}

    def fake_get(self, url, **kw):
        if url == be.AUTH_URL:
            return response(200, {"token": {"expiry": 9000}}) if state["verified"] else response(401, {"detail": "x"})
        return response(200, {"status": "ACTIVE", "is": {"checks": []}})

    def fake_post(self, url, **kw):
        if url.endswith("/persona"):
            state["verified"] = True
            return response(201, {})
        return response(201, {"inquiry": "inq42"}, url=be.AUTH_URL)

    with tempfile.TemporaryDirectory() as d:
        write_data(d)
        with patch("requests.Session.get", new=fake_get), patch("requests.Session.post", new=fake_post), \
             patch("builtins.input", side_effect=AssertionError("must never prompt")):
            at = press(app(d).run(), "Check with BRAIN")
            assert not at.exception
            assert "confirm your identity" in at.warning[0].value
            assert at.session_state["verify"]["url"].endswith("inquiry=inq42")
            at = press(at, "I've finished verifying")
        assert not at.exception
        assert len(at.warning) == 0                 # notice gone after the redraw
        assert len(at.dataframe) == 3               # BRAIN's record now shown


def test_session_check_asks_brain_how_long_is_left():
    session = MagicMock()
    for reply, valid in ((response(200, {"token": {"expiry": 9000}}), True),
                         (response(200, {"token": {"expiry": 30}}), False),
                         (response(401, {"detail": "x"}), False), (response(200, {}), False)):
        session.get.return_value = reply
        assert db.session_is_valid(session) is valid
        assert session.get.call_args.args[0] == be.AUTH_URL       # never inferred from an alpha lookup


if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)