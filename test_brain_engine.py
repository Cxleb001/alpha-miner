import csv
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
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

import brain_engine as be


def fake_response(status_code=200, json_data=None, headers=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.text = text
    r.url = "https://api.worldquantbrain.com/authentication"
    if json_data is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = json_data
    return r


def test_authenticate_success_immediate():
    session = MagicMock()
    session.post.return_value = fake_response(201, {"user": {"id": "abc"}})
    assert be.authenticate(session, "e@x.com", "pw") is True


def test_authenticate_requires_persona_then_succeeds():
    session = MagicMock()
    first = fake_response(201, {"inquiry": "inq_123"})
    followup = fake_response(201, {"user": {"id": "abc"}})
    session.post.side_effect = [first, followup]
    with patch("builtins.input", return_value=""):
        assert be.authenticate(session, "e@x.com", "pw") is True


def test_authenticate_bad_status():
    session = MagicMock()
    session.post.return_value = fake_response(401, text="bad creds")
    assert be.authenticate(session, "e@x.com", "pw") is False


def test_submit_simulation_ok():
    session = MagicMock()
    session.post.return_value = fake_response(201, headers={"Location": "https://api.worldquantbrain.com/simulations/123"})
    loc, err = be.submit_simulation(session, "close", {})
    assert loc == "https://api.worldquantbrain.com/simulations/123" and err is None


def test_submit_simulation_expired_session_detected():
    session = MagicMock()
    session.post.return_value = fake_response(401, {"detail": "Invalid credentials"})
    loc, err = be.submit_simulation(session, "close", {})
    assert loc is None and err == "EXPIRED_SESSION"


def test_poll_once_reports_each_state_without_waiting():
    session = MagicMock()
    cases = [(fake_response(200, {"progress": 0.4}), ("RUNNING", 0.4)),
             (fake_response(200, {"status": "COMPLETE", "alpha": "a1"}), ("DONE", "a1")),
             (fake_response(200, {"alpha": "a2"}), ("DONE", "a2")),
             (fake_response(200, {"status": "ERROR", "message": "bad field"}), ("ERROR", "bad field")),
             (fake_response(401, {"detail": "Incorrect authentication credentials."}), ("EXPIRED", None))]
    with patch("time.sleep") as sleep:
        for resp, expected in cases:
            session.get.return_value = resp
            assert be.poll_once(session, "loc") == expected
    sleep.assert_not_called()


def test_poll_simulation_status_shape_complete():
    session = MagicMock()
    session.get.return_value = fake_response(200, {"status": "COMPLETE", "alpha": "alpha_1"})
    alpha_id, err = be.poll_simulation(session, "loc")
    assert alpha_id == "alpha_1" and err is None


def test_poll_simulation_alpha_key_shape_complete():
    # the "no explicit status field, just an 'alpha' key" shape from the other client
    session = MagicMock()
    session.get.return_value = fake_response(200, {"alpha": "alpha_2"})
    alpha_id, err = be.poll_simulation(session, "loc")
    assert alpha_id == "alpha_2" and err is None


def test_poll_simulation_still_running_then_complete():
    session = MagicMock()
    session.get.side_effect = [
        fake_response(200, {"progress": 0.4}),
        fake_response(200, {"status": "COMPLETE", "alpha": "alpha_3"}),
    ]
    with patch("time.sleep", return_value=None):
        alpha_id, err = be.poll_simulation(session, "loc")
    assert alpha_id == "alpha_3" and err is None


def test_poll_simulation_error_message_shape():
    session = MagicMock()
    session.get.return_value = fake_response(200, {"message": "expression invalid"})
    alpha_id, err = be.poll_simulation(session, "loc")
    assert alpha_id is None and err == "expression invalid"


def test_fetch_alpha_metrics():
    session = MagicMock()
    session.get.return_value = fake_response(200, {
        "is": {
            "sharpe": 1.23, "fitness": 0.8, "turnover": 0.15,
            "checks": [{"name": "X", "result": "PASS"}, {"name": "Y", "result": "FAIL"}],
        }
    })
    m = be.fetch_alpha_metrics(session, "alpha_9")
    assert m["sharpe"] == 1.23 and m["checks_passed"] == 1 and m["checks_total"] == 2
    assert m["alpha_link"] == "https://platform.worldquantbrain.com/alpha/alpha_9"


def test_append_result_writes_header_once_and_rows(tmp_path):
    be.RESULTS_FILE = str(tmp_path / "results.csv")
    row1 = {k: "" for k in be.RESULTS_HEADER}
    row1["expression"] = "expr_a"
    row2 = {k: "" for k in be.RESULTS_HEADER}
    row2["expression"] = "expr_b"
    be.append_result(row1)
    be.append_result(row2)
    with open(be.RESULTS_FILE) as f:
        rows = list(csv.reader(f))
    assert rows[0] == be.RESULTS_HEADER
    assert len(rows) == 3  # header + 2 rows


def test_run_one_full_success_flow(tmp_path):
    be.RESULTS_FILE = str(tmp_path / "results.csv")
    session = MagicMock()
    session.post.return_value = fake_response(201, headers={"Location": "loc1"})
    session.get.side_effect = [
        fake_response(200, {"status": "COMPLETE", "alpha": "aid1"}),
        fake_response(200, {"is": {"sharpe": 2.0, "fitness": 1.0, "turnover": 0.1, "checks": []}}),
    ]
    be.run_one(session, "some_expr", {}, "e@x.com", "pw")
    with open(be.RESULTS_FILE) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "COMPLETE"
    assert rows[0]["sharpe"] == "2.0"


def test_run_one_reauth_on_expired_session(tmp_path):
    be.RESULTS_FILE = str(tmp_path / "results.csv")
    session = MagicMock()
    # first submit looks expired, then auth call succeeds, then second submit succeeds
    session.post.side_effect = [
        fake_response(401, {"detail": "invalid credentials"}),   # submit #1 -> expired
        fake_response(201, {"user": {"id": "abc"}}),               # re-auth
        fake_response(201, headers={"Location": "loc2"}),          # submit #2 -> ok
    ]
    session.get.side_effect = [
        fake_response(200, {"status": "COMPLETE", "alpha": "aid2"}),
        fake_response(200, {"is": {"sharpe": 1.5, "fitness": 0.5, "turnover": 0.2, "checks": []}}),
    ]
    be.run_one(session, "some_expr", {}, "e@x.com", "pw")
    with open(be.RESULTS_FILE) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "COMPLETE"
    assert rows[0]["alpha_id"] == "aid2"


def test_append_result_handles_non_ascii_text(tmp_path):
    # Same class of bug fixed in alpha_loop.py: writing without an explicit encoding
    # silently used a Windows codepage that can't represent characters like U+2011.
    be.RESULTS_FILE = str(tmp_path / "results.csv")
    row = {k: "" for k in be.RESULTS_HEADER}
    row["expression"] = "short\u2011term test"
    be.append_result(row)  # must not raise
    with open(be.RESULTS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "\u2011" in rows[0]["expression"]



# ---- Network resilience ----

import requests as _requests


def test_dropped_connection_is_retried_when_starting_a_simulation():
    session = MagicMock()
    session.post.side_effect = [_requests.ConnectionError("dropped"),
                                fake_response(201, headers={"Location": "https://api.worldquantbrain.com/simulations/1"})]
    with patch("time.sleep") as sleep:
        loc, err = be.submit_simulation(session, "close", {})
    assert loc.endswith("/simulations/1") and err is None
    sleep.assert_called_once_with(5)


def test_polling_survives_a_dropped_connection():
    session = MagicMock()
    session.get.side_effect = [_requests.Timeout("stalled"), fake_response(200, {"status": "COMPLETE", "alpha": "a1"})]
    with patch("time.sleep"):
        assert be.poll_simulation(session, "loc") == ("a1", None)


def test_gives_up_after_three_retries():
    session = MagicMock()
    session.post.side_effect = _requests.ConnectionError("down")
    with patch("time.sleep") as sleep:
        try:
            be.submit_simulation(session, "close", {})
            assert False, "should have raised"
        except _requests.ConnectionError:
            pass
    assert [c.args[0] for c in sleep.call_args_list] == [5, 15, 45]
    assert session.post.call_count == 4


def test_every_brain_request_has_a_timeout():
    session = MagicMock()
    session.post.return_value = fake_response(201, headers={"Location": "loc"})
    session.get.side_effect = [fake_response(200, {"status": "COMPLETE", "alpha": "a1"}), fake_response(200, {"is": {}})]
    be.submit_simulation(session, "close", {})
    be.poll_simulation(session, "loc")
    be.fetch_alpha(session, "a1")
    calls = session.post.call_args_list + session.get.call_args_list
    assert calls and all(c.kwargs.get("timeout") == be.REQUEST_TIMEOUT for c in calls)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    import inspect
    import tempfile
    passed, failed = 0, 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)