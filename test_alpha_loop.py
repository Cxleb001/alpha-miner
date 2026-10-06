import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("WQ_EMAIL", "e@x.com")
os.environ.setdefault("WQ_PASSWORD", "pw")
os.environ.setdefault("GEMINI_API_KEY", "fake-test-key")

import alpha_loop as al
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


def fake_gemini_response(ideas):
    resp = MagicMock()
    resp.text = json.dumps({"ideas": ideas})
    return resp


def in_tmp(tmp_path, fn, *a, **kw):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return fn(*a, **kw)
    finally:
        os.chdir(cwd)


def test_load_reference(tmp_path):
    def run():
        Path(al.OPERATORS_FILE).write_text(json.dumps({
            "operators": [{"name": "rank", "usage": "x"}],
            "fields": [{"name": "close", "description": "y"}],
        }))
        ops, fields = al.load_reference()
        assert ops[0]["name"] == "rank" and fields[0]["name"] == "close"
    in_tmp(tmp_path, run)


def test_session_save_and_load_roundtrip(tmp_path):
    def run():
        import requests as real_requests
        real_session = real_requests.Session()
        real_session.cookies.set("sid", "abc123")
        al.save_session(real_session)

        loaded = real_requests.Session()
        ok = al.load_session(loaded)
        assert ok is True
        assert loaded.cookies.get("sid") == "abc123"
    in_tmp(tmp_path, run)


def test_load_session_missing_file(tmp_path):
    def run():
        import requests as real_requests
        s = real_requests.Session()
        assert al.load_session(s) is False
    in_tmp(tmp_path, run)


def test_already_tried_reads_prior_csv(tmp_path):
    def run():
        row = {k: "" for k in al.RESULTS_HEADER}
        row["expression"] = "rank(close)"
        al.append_result(row)
        assert al.already_tried() == {"rank(close)"}
    in_tmp(tmp_path, run)


def test_safe_float_handles_junk():
    assert al.safe_float("1.5") == 1.5
    assert al.safe_float("") == 0.0
    assert al.safe_float("None") == 0.0
    assert al.safe_float(None) == 0.0


def test_submissions_today_and_record_roundtrip(tmp_path):
    def run():
        assert al.submissions_today() == 0
        al.record_submission()
        assert al.submissions_today() == 1
        al.record_submission()
        assert al.submissions_today() == 2
    in_tmp(tmp_path, run)


def test_summarize_round_no_prior():
    assert "round 1" in al.summarize_round([])


def test_summarize_round_all_failed():
    rows = [{"status": "FAILED", "expression": "bad(", "error": "syntax"}]
    text = al.summarize_round(rows)
    assert "none completed" in text


def test_summarize_round_ranks_by_fitness():
    rows = [
        {"status": "COMPLETE", "expression": "a", "sharpe": "1.0", "fitness": "0.3", "turnover": "0.1", "checks_passed": "2", "checks_total": "3", "submitted": "NO"},
        {"status": "COMPLETE", "expression": "b", "sharpe": "2.0", "fitness": "0.9", "turnover": "0.1", "checks_passed": "3", "checks_total": "3", "submitted": "NO"},
        {"status": "FAILED", "expression": "c", "sharpe": "", "fitness": "", "turnover": "", "checks_passed": "", "checks_total": "", "submitted": "NO"},
    ]
    text = al.summarize_round(rows)
    assert text.index("- b ->") < text.index("- a ->")  # higher fitness ranked first
    assert "1 other attempts failed" in text


def test_generate_ideas_parses_structured_output():
    client = MagicMock()
    client.models.generate_content.return_value = fake_gemini_response([
        {"expression": "rank(ts_delta(close, 5))", "rationale": "short-term momentum"},
        {"expression": "group_neutralize(ts_rank(sales, 252), industry)", "rationale": "fundamental strength vs peers"},
    ])
    operators = [{"name": "rank", "usage": "u"}]
    fields = [{"name": "close", "description": "d"}]
    ideas = al.generate_ideas(client, operators, fields, {"region": "USA", "universe": "TOP3000", "delay": 1},
                               "round 1", set(), 2)
    assert ideas[0][0] == "rank(ts_delta(close, 5))"
    assert ideas[1][1] == "fundamental strength vs peers"


def test_generate_ideas_raises_on_empty_response():
    client = MagicMock()
    resp = MagicMock()
    resp.text = None
    client.models.generate_content.return_value = resp
    try:
        al.generate_ideas(client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1}, "", set(), 1)
        assert False, "should have raised"
    except ValueError as e:
        assert "empty response" in str(e)


def test_run_round_full_success_flow_no_submit(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response([
            {"expression": "rank(close)", "rationale": "r1"},
        ])
        session = MagicMock()
        session.post.return_value = fake_response(201, headers={"Location": "loc1"})
        session.get.side_effect = [
            fake_response(200, {"status": "COMPLETE", "alpha": "aid1"}),
            fake_response(200, {"is": {"sharpe": 0.4, "fitness": 0.2, "turnover": 0.1,
                                        "checks": [{"name": "X", "result": "PASS"}]}}),
        ]
        with patch("time.sleep", return_value=None):
            rows = al.run_round(session, client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1},
                                 "e@x.com", "pw", 1, "round 1", set())
        assert rows[0]["status"] == "COMPLETE"
        assert rows[0]["submitted"] == "NO"  # below sharpe/fitness thresholds
    in_tmp(tmp_path, run)


def test_run_round_auto_submits_qualifying_alpha(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response([
            {"expression": "rank(close)", "rationale": "r1"},
        ])
        session = MagicMock()
        session.post.side_effect = [
            fake_response(201, headers={"Location": "loc1"}),  # submit_simulation
            fake_response(201, {}),                             # submit_alpha
        ]
        session.get.side_effect = [
            fake_response(200, {"status": "COMPLETE", "alpha": "aid1"}),
            fake_response(200, {"is": {"sharpe": 2.0, "fitness": 1.5, "turnover": 0.1,
                                        "checks": [{"name": "X", "result": "PASS"}]}}),
        ]
        with patch("time.sleep", return_value=None):
            rows = al.run_round(session, client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1},
                                 "e@x.com", "pw", 1, "round 1", set())
        assert rows[0]["submitted"] == "YES"
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_run_round_respects_daily_submission_cap(tmp_path):
    def run():
        al.record_submission()  # simulate the day's cap (default 1) already used

        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response([
            {"expression": "rank(close)", "rationale": "r1"},
        ])
        session = MagicMock()
        session.post.return_value = fake_response(201, headers={"Location": "loc1"})  # submit_simulation only
        session.get.side_effect = [
            fake_response(200, {"status": "COMPLETE", "alpha": "aid1"}),
            fake_response(200, {"is": {"sharpe": 2.0, "fitness": 1.5, "turnover": 0.1,
                                        "checks": [{"name": "X", "result": "PASS"}]}}),
        ]
        with patch("time.sleep", return_value=None):
            rows = al.run_round(session, client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1},
                                 "e@x.com", "pw", 1, "round 1", set())
        assert "daily cap reached" in rows[0]["submitted"]
        assert al.submissions_today() == 1  # unchanged - second submission was blocked, not sent
        session.post.assert_called_once()   # only the simulation submit, never a submit_alpha call
    in_tmp(tmp_path, run)


def test_run_round_stops_loop_when_session_cant_renew(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response([
            {"expression": "rank(close)", "rationale": "r1"},
        ])
        session = MagicMock()
        session.post.side_effect = [
            fake_response(401, {"detail": "invalid credentials"}),  # submit -> expired
            fake_response(401, text="still bad"),                    # authenticate() retry -> fails
        ]
        raised = False
        try:
            with patch("time.sleep", return_value=None):
                al.run_round(session, client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1},
                              "e@x.com", "pw", 1, "round 1", set())
        except al.StopLoop:
            raised = True
        assert raised, "expected StopLoop when session can't be renewed unattended"

        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["status"] == "STOPPED"
    in_tmp(tmp_path, run)


def test_run_round_skips_duplicate_ideas(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response([
            {"expression": "rank(close)", "rationale": "r1"},
        ])
        session = MagicMock()
        avoid = {"rank(close)"}  # already tried
        with patch("time.sleep", return_value=None):
            rows = al.run_round(session, client, [], [], {"region": "USA", "universe": "TOP3000", "delay": 1},
                                 "e@x.com", "pw", 1, "round 1", avoid)
        assert rows == []  # skipped, nothing submitted
        session.post.assert_not_called()
    in_tmp(tmp_path, run)


def test_main_respects_max_rounds_cap(tmp_path):
    def run():
        Path(al.OPERATORS_FILE).write_text(json.dumps({
            "operators": [{"name": "rank", "usage": "u"}],
            "fields": [{"name": "close", "description": "d"}],
        }))
        call_count = {"n": 0}

        def fake_generate(*a, **kw):
            call_count["n"] += 1
            return [(f"rank(close)_{call_count['n']}", "r")]

        fake_session = MagicMock()
        with patch.object(al, "IDEAS_PER_ROUND", 1), \
             patch.object(al, "MAX_ROUNDS", 3), \
             patch.object(al, "generate_ideas", side_effect=fake_generate), \
             patch.object(al, "get_authenticated_session", return_value=fake_session), \
             patch.object(be, "submit_simulation", return_value=(None, "submit failed (500): boom")), \
             patch("google.genai.Client"), \
             patch("time.sleep", return_value=None):
            al.main()

        assert call_count["n"] == 3  # exactly MAX_ROUNDS generation calls, then stopped
    in_tmp(tmp_path, run)


def test_append_result_handles_llm_style_unicode(tmp_path):
    # Reproduces the earlier Windows encoding crash: LLM-generated rationale text containing
    # typographic characters used to blow up without an explicit encoding on the file open.
    def run():
        row = {k: "" for k in al.RESULTS_HEADER}
        row["expression"] = "rank(ts_delta(close, 5))"
        row["rationale"] = "captures short\u2011term risk\u2014adjusted momentum, a \u201cclassic\u201d factor"
        al.append_result(row)
        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert "\u2011" in rows[0]["rationale"]
        assert "\u201cclassic\u201d" in rows[0]["rationale"]
    in_tmp(tmp_path, run)


if __name__ == "__main__":
    import inspect
    import tempfile
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
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