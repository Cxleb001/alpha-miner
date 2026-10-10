import csv
import json
import os
import sys
from collections import Counter
import time as _time
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

import alpha_loop as al
import brain_engine as be
from google.genai import errors as genai_errors


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
    ideas = al.generate_ideas(client, "some-model", operators, fields,
                               {"region": "USA", "universe": "TOP3000", "delay": 1}, "round 1", set(), 2)
    assert ideas[0][0] == "rank(ts_delta(close, 5))"
    assert ideas[1][1] == "fundamental strength vs peers"
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "some-model"  # the model passed in is the one actually called
    assert kwargs["config"].automatic_function_calling.disable is True  # AFC off -> no AFC log noise


def test_generate_ideas_raises_on_empty_response():
    client = MagicMock()
    resp = MagicMock()
    resp.text = None
    client.models.generate_content.return_value = resp
    try:
        al.generate_ideas(client, "some-model", [], [], {"region": "USA", "universe": "TOP3000", "delay": 1}, "", set(), 1)
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
            fake_response(200, {"max": 0.41}, text='{"max": 0.41}'),   # free similarity check before holding
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


# ---- Retry / fallback / round accounting (the fix for the all-503 run) ----

SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1}
ONE_IDEA = [{"expression": "rank(close)", "rationale": "r1"}]


def busy():
    # The exact error the real run hit: 503 UNAVAILABLE, "high demand".
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def bad_request():
    return genai_errors.ClientError(400, {"error": {"code": 400, "message": "invalid", "status": "INVALID_ARGUMENT"}})


def models_called(client):
    return [c.kwargs["model"] for c in client.models.generate_content.call_args_list]


def test_generation_retries_busy_model_then_succeeds():
    client = MagicMock()
    client.models.generate_content.side_effect = [busy(), busy(), fake_gemini_response(ONE_IDEA)]
    with patch("time.sleep") as sleep:
        ideas = al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1)
    assert ideas == [("rank(close)", "r1")]
    assert [c.args[0] for c in sleep.call_args_list] == [10, 20]  # backoff, in order
    assert models_called(client) == [al.GEMINI_MODEL] * 3          # never needed the fallback


def test_generation_falls_back_after_retries_run_out():
    client = MagicMock()
    client.models.generate_content.side_effect = [busy()] * 5 + [fake_gemini_response(ONE_IDEA)]
    with patch("time.sleep") as sleep:
        ideas = al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1)
    assert ideas == [("rank(close)", "r1")]
    assert [c.args[0] for c in sleep.call_args_list] == [10, 20, 40, 80]
    assert models_called(client) == [al.GEMINI_MODEL] * 5 + [al.GEMINI_FALLBACK_MODEL]


def test_generation_does_not_wait_out_a_non_transient_error():
    client = MagicMock()
    client.models.generate_content.side_effect = [bad_request(), fake_gemini_response(ONE_IDEA)]
    with patch("time.sleep") as sleep:
        ideas = al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1)
    assert ideas == [("rank(close)", "r1")]
    sleep.assert_not_called()  # waiting won't fix a bad request - straight to the fallback
    assert models_called(client) == [al.GEMINI_MODEL, al.GEMINI_FALLBACK_MODEL]


def test_generation_raises_when_both_models_fail():
    client = MagicMock()
    client.models.generate_content.side_effect = [busy()] * 6
    with patch("time.sleep"):
        try:
            al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1)
            assert False, "should have raised"
        except genai_errors.ServerError as e:
            assert e.code == 503
    assert client.models.generate_content.call_count == 6  # 1 try + 4 retries + 1 fallback


def test_generation_with_fallback_turned_off():
    client = MagicMock()
    client.models.generate_content.side_effect = [busy()] * 5
    with patch.object(al, "GEMINI_FALLBACK_MODEL", ""), patch("time.sleep"):
        try:
            al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1)
            assert False, "should have raised"
        except genai_errors.ServerError:
            pass
    assert models_called(client) == [al.GEMINI_MODEL] * 5


def test_run_round_raises_generation_failed_and_touches_nothing(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.side_effect = [busy()] * 6
        session = MagicMock()
        with patch("time.sleep"):
            try:
                al.run_round(session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
                assert False, "should have raised"
            except al.GenerationFailed as e:
                assert "503" in str(e)
        session.post.assert_not_called()             # nothing reached BRAIN
        assert not Path(al.RESULTS_FILE).exists()    # and nothing was logged as a result
    in_tmp(tmp_path, run)


def _write_operators():
    Path(al.OPERATORS_FILE).write_text(json.dumps({
        "operators": [{"name": "rank", "usage": "u"}],
        "fields": [{"name": "close", "description": "d"}],
    }))


def test_main_failed_generation_does_not_use_up_rounds(tmp_path):
    def run():
        _write_operators()
        outcomes = iter(["busy", "busy", "ok", "ok", "ok"])
        calls = {"n": 0}

        def fake_generate(*a, **kw):
            calls["n"] += 1
            if next(outcomes) == "busy":
                raise busy()
            return [(f"rank(close)_{calls['n']}", "r")]

        with patch.object(al, "IDEAS_PER_ROUND", 1), \
             patch.object(al, "MAX_ROUNDS", 3), \
             patch.object(al, "generate_with_fallback", side_effect=fake_generate), \
             patch.object(al, "get_authenticated_session", return_value=MagicMock()), \
             patch.object(be, "submit_simulation", return_value=(None, "submit failed (500): boom")), \
             patch("google.genai.Client"), \
             patch("time.sleep"):
            al.main()

        assert calls["n"] == 5  # 2 failed attempts + 3 real rounds
        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            rounds = [row["round"] for row in csv.DictReader(f)]
        assert rounds == ["1", "2", "3"]  # the failed attempts didn't consume round numbers
    in_tmp(tmp_path, run)


def test_main_stops_after_repeated_generation_failures(tmp_path):
    def run():
        _write_operators()
        submit = MagicMock()
        with patch.object(al, "MAX_GENERATION_FAILURES", 3), \
             patch.object(al, "generate_with_fallback", side_effect=busy()) as gen, \
             patch.object(al, "get_authenticated_session", return_value=MagicMock()), \
             patch.object(be, "submit_simulation", submit), \
             patch("google.genai.Client"), \
             patch("time.sleep"):
            al.main()
        assert gen.call_count == 3   # stopped after 3 in a row instead of looping for 4 hours
        submit.assert_not_called()   # no BRAIN quota spent
    in_tmp(tmp_path, run)



# ---- Submit path, auto-flip, turnover feedback ----

from datetime import date as _date, datetime as _dt, timezone as _tz

QUALIFYING_CHECKS = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "LOW_FITNESS", "result": "PASS"},
                     {"name": "HIGH_TURNOVER", "result": "PASS"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]


def alpha_json(sharpe=1.6, fitness=1.2, turnover=0.3, checks=None, status="UNSUBMITTED"):
    return {"status": status, "is": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
                                     "checks": QUALIFYING_CHECKS if checks is None else checks}}


class FakeBrain:
    """Answers by URL, the way community clients recorded BRAIN behaving."""
    def __init__(self, alphas, corr=0.3, submit_post=503, verdict=None):
        self.alphas = alphas                # {alpha_id: alpha json}; simulations hand them out in order
        self.order = list(alphas)
        self.corr = corr
        self.submit_post = submit_post
        self.verdict = verdict if verdict is not None else {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]}}
        self.submitted = set()
        self.session = MagicMock()
        self.session.get.side_effect = self.get
        self.session.post.side_effect = self.post
        self.submit_polls = 0

    def post(self, url, **kw):
        if url.endswith("/simulations"):
            return fake_response(201, headers={"Location": f"sim-{self.order.pop(0)}"})
        if url.endswith("/submit"):
            if self.submit_post in (200, 201, 202, 503):
                return fake_response(self.submit_post, {}, headers={"Retry-After": "1.0"})
            return fake_response(self.submit_post, {"is": {"checks": [{"name": "LOW_SHARPE", "result": "FAIL"}]}})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, **kw):
        if url == be.AUTH_URL:   # "how long has this login got left?"
            return fake_response(200, {"token": {"expiry": 14400}})
        if url.startswith("sim-"):
            return fake_response(200, {"status": "COMPLETE", "alpha": url[4:]})
        if url.endswith("/correlations/self"):
            alpha_id = url.split("/")[-3]
            corr = self.corr.get(alpha_id, 0.3) if isinstance(self.corr, dict) else self.corr
            return fake_response(200, {"max": corr})
        if url.endswith("/submit"):
            self.submit_polls += 1
            if self.submit_polls == 1:  # first poll: still working
                return fake_response(200, None, headers={"Retry-After": "2.5"}, text="")
            alpha_id = url.split("/")[-2]
            if not failing_names(self.verdict):
                self.submitted.add(alpha_id)
            return fake_response(200, self.verdict, text=json.dumps(self.verdict))
        alpha_id = url.rsplit("/", 1)[-1]
        data = dict(self.alphas[alpha_id])
        if alpha_id in self.submitted:
            data["status"] = "ACTIVE"
        return fake_response(200, data)


def failing_names(verdict):
    return [c for c in (verdict.get("is") or {}).get("checks", []) if c.get("result") == "FAIL"]


def run_one_idea(brain, expression="rank(close)"):
    client = MagicMock()
    client.models.generate_content.return_value = fake_gemini_response([{"expression": expression, "rationale": "r"}])
    with patch("time.sleep"):
        return al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())


def test_brain_day_rolls_at_3am_us_eastern():
    # Summer (EDT, UTC-4): 3 AM Eastern = 07:00 UTC
    assert al.brain_day(_dt(2026, 10, 7, 6, 59, tzinfo=_tz.utc)) == _date(2026, 10, 6)
    assert al.brain_day(_dt(2026, 10, 7, 7, 0, tzinfo=_tz.utc)) == _date(2026, 10, 7)
    # Winter (EST, UTC-5): 3 AM Eastern = 08:00 UTC
    assert al.brain_day(_dt(2026, 12, 1, 7, 59, tzinfo=_tz.utc)) == _date(2026, 11, 30)
    assert al.brain_day(_dt(2026, 12, 1, 8, 0, tzinfo=_tz.utc)) == _date(2026, 12, 1)


def test_qualifies_ignores_only_a_pending_self_correlation():
    good = {"sharpe": 1.6, "fitness": 1.2}
    assert al.qualifies(good, QUALIFYING_CHECKS) is True
    failed_turnover = QUALIFYING_CHECKS[:2] + [{"name": "HIGH_TURNOVER", "result": "FAIL"}] + QUALIFYING_CHECKS[3:]
    assert al.qualifies(good, failed_turnover) is False
    failed_corr = QUALIFYING_CHECKS[:3] + [{"name": "SELF_CORRELATION", "result": "FAIL"}]
    assert al.qualifies(good, failed_corr) is False
    assert al.qualifies({"sharpe": 1.2, "fitness": 1.2}, QUALIFYING_CHECKS) is False
    assert al.qualifies(good, []) is False


def test_self_correlation_waits_while_brain_computes():
    session = MagicMock()
    session.get.side_effect = [
        fake_response(503, {}, headers={"Retry-After": "1.5"}),
        fake_response(200, None, headers={"Retry-After": "2.0"}, text=""),
        fake_response(200, {"max": 0.42}, text='{"max": 0.42}'),
    ]
    with patch("time.sleep") as sleep:
        assert al.self_correlation_max(session, "a1") == 0.42
    assert [c.args[0] for c in sleep.call_args_list] == [1.5, 2.0]


def test_full_submit_path_ends_in_yes_only_when_brain_lists_it_active(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()})
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "YES"
        assert al.submissions_today() == 1
        assert brain.submit_polls == 2  # waited through the 'still working' reply
    in_tmp(tmp_path, run)


def test_rejected_verdict_is_recorded_and_still_uses_the_slot(tmp_path):
    def run():
        verdict = {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]}}
        brain = FakeBrain({"a1": alpha_json()}, verdict=verdict)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "REJECTED: failed SELF_CORRELATION"
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_immediate_403_is_a_rejection(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()}, submit_post=403)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "REJECTED: failed LOW_SHARPE"
    in_tmp(tmp_path, run)


def test_too_correlated_is_skipped_without_using_the_slot(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()}, corr=0.83)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "TOO_CORRELATED (max 0.83)"
        assert al.submissions_today() == 0
        assert not any(c.args[0].endswith("/submit") for c in brain.session.post.call_args_list)
    in_tmp(tmp_path, run)


def test_unconfirmed_success_is_reported_as_pending_not_yes(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()})
        brain.submitted = set()
        original_get = brain.get
        def get(url, **kw):  # verdict looks fine, but BRAIN never lists the alpha as ACTIVE
            r = original_get(url, **kw)
            brain.submitted.clear()
            return r
        brain.session.get.side_effect = get
        rows = run_one_idea(brain)
        assert rows[0]["submitted"].startswith("VERDICT_PENDING")
    in_tmp(tmp_path, run)


def test_strongly_negative_sharpe_gets_flipped_once(tmp_path):
    def run():
        brain = FakeBrain({"neg": alpha_json(sharpe=-1.76, fitness=-0.92, checks=[]),
                           "pos": alpha_json(sharpe=1.76, fitness=0.92, turnover=0.1, checks=[])})  # too calm to smooth
        rows = run_one_idea(brain, "rank(ts_av_diff(close, 10))")
        assert [r["expression"] for r in rows] == ["rank(ts_av_diff(close, 10))",
                                                   "reverse(rank(ts_av_diff(close, 10)))"]
        assert rows[1]["rationale"].startswith("Auto-flip of rank(ts_av_diff(close, 10))")
        assert "1.76" in rows[1]["rationale"]
    in_tmp(tmp_path, run)


def test_flips_are_capped_per_round(tmp_path):
    def run():
        ideas = [{"expression": f"rank(f{i})", "rationale": "r"} for i in range(5)]
        alphas = {f"n{i}": alpha_json(sharpe=-1.5, checks=[]) for i in range(5)}
        alphas.update({f"p{i}": alpha_json(sharpe=1.5, checks=[]) for i in range(5)})
        brain = FakeBrain(alphas)
        brain.order = [f"n{i}" for i in range(5)] + [f"p{i}" for i in range(5)]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ideas)
        with patch.object(al, "MAX_FLIPS_PER_ROUND", 3), patch("time.sleep"):
            rows = al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        flipped = [r for r in rows if r["expression"].startswith("reverse(")]
        assert len(flipped) == 3 and len(rows) == 8
    in_tmp(tmp_path, run)


def test_mild_negative_sharpe_is_not_flipped(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json(sharpe=-0.5, checks=[])})
        rows = run_one_idea(brain)
        assert len(rows) == 1
    in_tmp(tmp_path, run)


def test_prompt_tells_the_generator_the_turnover_range():
    client = MagicMock()
    client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
    al.generate_ideas(client, "m", [], [], SETTINGS, "", set(), 1)
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "turnover between 0.01 and 0.7" in prompt
    assert "re-tested reversed" in prompt


def test_feedback_flags_turnover_and_failing_checks():
    rows = [{"status": "COMPLETE", "expression": "fast", "sharpe": "1.59", "fitness": "0.49", "turnover": "0.7264",
             "checks_passed": "4", "checks_total": "8", "submitted": "NO", "failing_checks": "HIGH_TURNOVER:FAIL"},
            {"status": "COMPLETE", "expression": "slow", "sharpe": "0.2", "fitness": "0.1", "turnover": "0.004",
             "checks_passed": "4", "checks_total": "8", "submitted": "NO"}]
    text = al.summarize_round(rows)
    assert "TURNOVER TOO HIGH" in text and "not_passing=[HIGH_TURNOVER:FAIL]" in text
    assert "TURNOVER TOO LOW" in text


def test_in_memory_extras_never_break_the_csv(tmp_path):
    def run():
        row = {k: "" for k in al.RESULTS_HEADER}
        row.update(expression="x", failing_checks="A:FAIL")
        al.append_result(row)  # must not raise on the extra key
        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            assert next(csv.reader(f)) == al.RESULTS_HEADER
    in_tmp(tmp_path, run)



# ---- Auto-smooth and memory across runs ----

def test_good_sharpe_low_fitness_high_turnover_gets_smoothed(tmp_path):
    def run():
        expr = "group_rank(ts_av_diff(implied_volatility_call_60, 60), sector)"  # your 1.67 / 0.64 / 60% alpha
        brain = FakeBrain({"fast": alpha_json(sharpe=1.67, fitness=0.64, turnover=0.6033, checks=[]),
                           "calm": alpha_json(sharpe=1.6, fitness=1.1, turnover=0.15, checks=[])})
        rows = run_one_idea(brain, expr)
        assert [r["expression"] for r in rows] == [expr, f"ts_decay_linear({expr}, 10)"]
        assert rows[1]["rationale"].startswith(f"Auto-smooth of {expr}: Sharpe 1.67 but fitness 0.64")
    in_tmp(tmp_path, run)


def test_turnover_over_the_limit_gets_smoothed_even_with_good_fitness(tmp_path):
    def run():
        brain = FakeBrain({"a": alpha_json(sharpe=1.59, fitness=1.05, turnover=0.7264, checks=[]),
                           "b": alpha_json(sharpe=1.5, fitness=1.0, turnover=0.2, checks=[])})
        rows = run_one_idea(brain)
        assert len(rows) == 2 and rows[1]["expression"].startswith("ts_decay_linear(")
    in_tmp(tmp_path, run)


def test_no_smoothing_when_it_cannot_help(tmp_path):
    def run():
        for sharpe, fitness, turnover in [(1.17, 0.81, 0.16),   # turnover already low
                                          (1.0, 0.5, 0.6),     # Sharpe below the bar
                                          (1.6, 1.2, 0.3)]:    # already clears fitness
            brain = FakeBrain({"a": alpha_json(sharpe=sharpe, fitness=fitness, turnover=turnover, checks=[])})
            assert len(run_one_idea(brain)) == 1, (sharpe, fitness, turnover)
    in_tmp(tmp_path, run)


def test_flip_then_smooth_chain_and_no_smoothing_twice(tmp_path):
    def run():
        brain = FakeBrain({"neg": alpha_json(sharpe=-1.76, fitness=-0.92, turnover=0.53, checks=[]),
                           "flip": alpha_json(sharpe=1.76, fitness=0.92, turnover=0.53, checks=[]),
                           "smooth": alpha_json(sharpe=1.7, fitness=0.95, turnover=0.4, checks=[])})
        rows = run_one_idea(brain, "rank(ts_av_diff(close, 10))")
        assert [r["expression"] for r in rows] == [
            "rank(ts_av_diff(close, 10))",
            "reverse(rank(ts_av_diff(close, 10)))",
            "ts_decay_linear(reverse(rank(ts_av_diff(close, 10))), 10)",
        ]  # the smoothed one still misses fitness, but is never smoothed again
    in_tmp(tmp_path, run)


def test_smooths_are_capped_per_round(tmp_path):
    def run():
        ideas = [{"expression": f"rank(f{i})", "rationale": "r"} for i in range(5)]
        alphas = {f"n{i}": alpha_json(sharpe=1.5, fitness=0.5, turnover=0.6, checks=[]) for i in range(5)}
        alphas.update({f"s{i}": alpha_json(sharpe=1.4, fitness=0.9, turnover=0.15, checks=[]) for i in range(5)})
        brain = FakeBrain(alphas)
        brain.order = [f"n{i}" for i in range(5)] + [f"s{i}" for i in range(5)]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ideas)
        with patch.object(al, "MAX_SMOOTHS_PER_ROUND", 3), patch("time.sleep"):
            rows = al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert sum(r["expression"].startswith("ts_decay_linear(") for r in rows) == 3 and len(rows) == 8
    in_tmp(tmp_path, run)


def _history_row(expression, sharpe, fitness, turnover, status="COMPLETE"):
    row = {k: "" for k in al.RESULTS_HEADER}
    row.update(expression=expression, sharpe=str(sharpe), fitness=str(fitness), turnover=str(turnover), status=status)
    return row


def test_feedback_remembers_best_results_across_runs(tmp_path):
    def run():
        assert al.build_feedback() == "This is round 1 - no prior results yet."
        al.append_result(_history_row("rank(weak)", 0.2, 0.1, 0.3))
        al.append_result(_history_row("rank(best)", 1.67, 0.64, 0.6033))
        al.append_result(_history_row("rank(middle)", 1.17, 0.81, 0.16))
        al.append_result(_history_row("rank(bad(", "", "", "", status="FAILED"))
        text = al.build_feedback()
        assert "out of 3 simulated alphas across all runs" in text
        assert text.index("rank(middle)") < text.index("rank(best)") < text.index("rank(weak)")  # by fitness
        assert "rank(bad(" not in text
        with patch.object(al, "MEMORY_TOP_N", 1):
            assert "rank(best)" not in al.build_feedback() and "rank(middle)" in al.build_feedback()
    in_tmp(tmp_path, run)


def test_feedback_after_a_round_keeps_memory_and_adds_last_round(tmp_path):
    def run():
        al.append_result(_history_row("rank(old_best)", 1.5, 0.9, 0.3))
        last_round = [dict(_history_row("rank(new)", 0.5, 0.2, 0.75), checks_passed="4", checks_total="8")]
        text = al.build_feedback(last_round)
        assert "rank(old_best)" in text and "Last round's results" in text and "TURNOVER TOO HIGH" in text
    in_tmp(tmp_path, run)


def test_a_new_run_starts_from_past_results(tmp_path):
    def run():
        _write_operators()
        al.append_result(_history_row("rank(implied_volatility_call_60)", 1.67, 0.64, 0.6))
        seen = []

        def fake_generate(client, operators, fields, settings, feedback_text, avoid, n):
            seen.append(feedback_text)
            return [("rank(close)", "r")]

        with patch.object(al, "MAX_ROUNDS", 1), \
             patch.object(al, "generate_with_fallback", side_effect=fake_generate), \
             patch.object(al, "get_authenticated_session", return_value=MagicMock()), \
             patch.object(be, "submit_simulation", return_value=(None, "submit failed (500): boom")), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main()
        assert "rank(implied_volatility_call_60)" in seen[0]  # round 1 already knows the best past idea
    in_tmp(tmp_path, run)



# ---- Follow-ups first, and --try ----

THREE = [
    "ts_decay_linear(group_rank(ts_av_diff(unsystematic_risk_last_90_days, 60), industry), 10)",
    "ts_decay_linear(group_rank(ts_av_diff(implied_volatility_call_60, 60), sector), 10)",
    "ts_decay_linear(reverse(rank(ts_av_diff(close, 10))), 10)",
]


def calm(sharpe=0.5):
    return alpha_json(sharpe=sharpe, fitness=0.2, turnover=0.1, checks=[])


def test_followups_run_right_after_their_trigger(tmp_path):
    def run():
        ideas = [{"expression": e, "rationale": "r"} for e in ("rank(a)", "rank(b)", "rank(c)")]
        brain = FakeBrain({"neg": alpha_json(sharpe=-1.5, checks=[]), "flip": calm(), "b": calm(), "c": calm()})
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ideas)
        with patch("time.sleep"), patch.object(al, "PARALLEL_SIMULATIONS", 1):  # one at a time: exact order
            rows = al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert [r["expression"] for r in rows] == ["rank(a)", "reverse(rank(a))", "rank(b)", "rank(c)"]
    in_tmp(tmp_path, run)


def run_main_try(brain, argv):
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}  # --try must not need Gemini
    with patch.dict(os.environ, env, clear=True), \
         patch.object(al, "get_authenticated_session", return_value=brain.session), \
         patch("google.genai.Client", side_effect=AssertionError("--try must not call Gemini")), \
         patch("time.sleep"):
        al.main(argv)
    with open(al.RESULTS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_try_runs_your_exact_expressions_without_gemini(tmp_path):
    def run():
        brain = FakeBrain({"a": calm(), "b": calm(), "c": calm()})
        rows = run_main_try(brain, ["--try", *THREE])
        assert [r["expression"] for r in rows] == THREE          # commas and spaces arrive intact
        assert all(r["round"] == "try" and r["rationale"] == "Requested with --try" for r in rows)
        sent = [c.kwargs["json"]["regular"] for c in brain.session.post.call_args_list if c.args[0].endswith("/simulations")]
        assert sent == THREE                                     # exactly what BRAIN received
    in_tmp(tmp_path, run)


def test_try_submits_a_qualifying_expression(tmp_path):
    def run():
        brain = FakeBrain({"a": alpha_json(sharpe=1.61, fitness=1.12, turnover=0.18)})
        rows = run_main_try(brain, ["--try", THREE[0]])
        assert rows[0]["submitted"] == "YES"
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_try_followups_run_immediately(tmp_path):
    def run():
        brain = FakeBrain({"fast": alpha_json(sharpe=1.70, fitness=0.80, turnover=0.399, checks=[]),
                           "smooth": calm(), "next": calm()})
        expr = "group_rank(ts_av_diff(unsystematic_risk_last_90_days, 60), industry)"
        with patch.object(al, "PARALLEL_SIMULATIONS", 1):  # one at a time: exact order
            rows = run_main_try(brain, ["--try", expr, "rank(close)"])
        assert [r["expression"] for r in rows] == [expr, f"ts_decay_linear({expr}, 10)", "rank(close)"]
    in_tmp(tmp_path, run)


def test_try_skips_finished_expressions_but_retries_failed_ones(tmp_path):
    def run():
        al.append_result(_history_row("rank(done)", 1.0, 0.5, 0.3))
        al.append_result(_history_row("rank(flaky)", "", "", "", status="FAILED"))
        brain = FakeBrain({"x": calm()})
        rows = run_main_try(brain, ["--try", "rank(done)", "rank(flaky)"])
        new = [r for r in rows if r["round"] == "try"]
        assert [r["expression"] for r in new] == ["rank(flaky)"]
    in_tmp(tmp_path, run)



# ---- Network resilience ----

import requests as _requests
import httpx as _httpx


def flaky_sim_posts(brain, failing_ideas):
    """Make the first `failing_ideas` simulation starts fail on every attempt, like a dropped connection."""
    original_post = brain.post
    state = {"sim_posts": 0}

    def post(url, **kw):
        if url.endswith("/simulations"):
            state["sim_posts"] += 1
            if state["sim_posts"] <= failing_ideas * 4:  # 1 try + 3 retries per idea
                raise _requests.ConnectionError("connection dropped")
        return original_post(url, **kw)

    brain.session.post.side_effect = post


def test_dropped_connection_skips_one_idea_and_carries_on(tmp_path):
    def run():
        ideas = [{"expression": "rank(a)", "rationale": "r"}, {"expression": "rank(b)", "rationale": "r"}]
        brain = FakeBrain({"b": calm()})
        flaky_sim_posts(brain, failing_ideas=1)
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ideas)
        with patch("time.sleep"):
            rows = al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert [(r["expression"], r["status"]) for r in rows] == [("rank(a)", "FAILED"), ("rank(b)", "COMPLETE")]
        assert "Couldn't reach BRAIN (ConnectionError)" in rows[0]["error"]
        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 2  # both recorded, nothing lost
    in_tmp(tmp_path, run)


def test_connection_down_stops_the_run_cleanly(tmp_path):
    def run():
        _write_operators()
        brain = FakeBrain({})
        brain.session.post.side_effect = _requests.ConnectionError("down")
        gen = MagicMock(return_value=[(f"rank(x{i})", "r") for i in range(8)])
        with patch.object(al, "generate_with_fallback", gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])  # must return normally, not crash with a traceback
        with open(al.RESULTS_FILE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == al.MAX_NETWORK_FAILURES and all(r["status"] == "FAILED" for r in rows)
        assert gen.call_count == 1  # stopped instead of burning through more rounds
    in_tmp(tmp_path, run)


def test_unexpected_error_skips_only_that_idea(tmp_path):
    def run():
        ideas = [{"expression": "rank(a)", "rationale": "r"}, {"expression": "rank(b)", "rationale": "r"}]
        brain = FakeBrain({"a": calm(), "b": calm()})
        real_fetch = be.fetch_alpha
        calls = {"n": 0}

        def fetch(session, alpha_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyError("a reply shape nobody expected")
            return real_fetch(session, alpha_id)

        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ideas)
        with patch.object(be, "fetch_alpha", side_effect=fetch), patch("time.sleep"):
            rows = al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert [r["status"] for r in rows] == ["FAILED", "COMPLETE"]
        assert rows[0]["error"].startswith("Unexpected error: KeyError")
    in_tmp(tmp_path, run)


def test_lost_connection_after_submitting_is_pending_not_lost(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()})
        with patch.object(al, "is_active_on_brain", side_effect=_requests.ConnectionError("dropped")):
            rows = run_one_idea(brain)
        assert rows[0]["submitted"].startswith("VERDICT_PENDING")
        assert al.submissions_today() == 1  # BRAIN took it, so the slot still counts
    in_tmp(tmp_path, run)


def test_gemini_connection_drops_are_retried():
    assert al.is_transient(_httpx.ConnectError("down")) and al.is_transient(_httpx.ReadTimeout("slow"))
    client = MagicMock()
    client.models.generate_content.side_effect = [_httpx.ConnectError("down"), _httpx.ConnectError("down"),
                                                  fake_gemini_response(ONE_IDEA)]
    with patch("time.sleep") as sleep:
        assert al.generate_with_fallback(client, [], [], SETTINGS, "", set(), 1) == [("rank(close)", "r1")]
    assert [c.args[0] for c in sleep.call_args_list] == [10, 20]


def test_startup_survives_a_dropped_connection(tmp_path):
    def run():
        Path(al.SESSION_FILE).write_text('{"sid": "old"}', encoding="utf-8")
        fresh = MagicMock()
        fresh.get.side_effect = _requests.ConnectionError("dropped")
        with patch.object(al.requests, "Session", return_value=fresh), \
             patch.object(be, "authenticate", return_value=True) as auth:
            assert al.get_authenticated_session("e@x.com", "pw") is fresh
        auth.assert_called_once()  # couldn't check the old session, so it logged in fresh
    in_tmp(tmp_path, run)



# ---- Held alphas go first ----

def held_row(alpha_id, fitness, sharpe=1.4, expression=None, submitted=al.HELD_LABEL):
    row = _history_row(expression or f"rank({alpha_id})", sharpe, fitness, 0.11)
    row.update(alpha_id=alpha_id, submitted=submitted, round="1", checks_passed="7", checks_total="8")
    return row


def csv_rows():
    with open(al.RESULTS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def submitted_value(alpha_id):
    return next(r["submitted"] for r in csv_rows() if r["alpha_id"] == alpha_id)


def submit_posts(brain):
    return [c.args[0].split("/")[-2] for c in brain.session.post.call_args_list if c.args[0].endswith("/submit")]


def run_held(brain):
    with patch("time.sleep"):
        return al.submit_held(brain.session)


def test_held_candidates_best_fitness_first_and_skips_decided(tmp_path):
    def run():
        al.append_result(held_row("low", 1.05))
        al.append_result(held_row("high", 1.30))
        al.append_result(held_row("done", 1.10, submitted="YES"))
        al.append_result(held_row("fallback", 1.50))
        al.append_result(held_row("fallback", 1.50, submitted="YES"))  # outcome appended later
        al.append_result(_history_row("rank(failed(", "", "", "", status="FAILED"))
        assert [r["alpha_id"] for r in al.held_candidates()] == ["high", "low"]
    in_tmp(tmp_path, run)


def test_best_held_alpha_is_submitted_and_its_row_updated_in_place(tmp_path):
    def run():
        al.append_result(_history_row("rank(other)", 0.5, 0.2, 0.1))
        al.append_result(held_row("h_low", 1.05))
        al.append_result(held_row("h_best", 1.27))
        before = csv_rows()
        brain = FakeBrain({"h_low": alpha_json(), "h_best": alpha_json()})
        assert run_held(brain) == "YES"
        assert submit_posts(brain) == ["h_best"]           # best fitness only - one slot a day
        assert al.submissions_today() == 1
        after = csv_rows()
        assert len(after) == len(before)                    # updated in place, no extra row
        assert submitted_value("h_best") == "YES"
        assert submitted_value("h_low") == al.HELD_LABEL    # still queued for tomorrow
        assert [r for r in after if r["alpha_id"] != "h_best"] == [r for r in before if r["alpha_id"] != "h_best"]
        assert not Path(al.RESULTS_FILE + ".tmp").exists()
    in_tmp(tmp_path, run)


def test_held_alphas_wait_when_todays_slot_is_used(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.2))
        al.record_submission()
        brain = FakeBrain({"h1": alpha_json()})
        assert run_held(brain) is None
        assert not brain.session.post.called                     # checked, never submitted
        assert submitted_value("h1") == al.HELD_LABEL             # still a good candidate: stays queued
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_too_correlated_held_alpha_is_marked_and_the_next_one_tried(tmp_path):
    def run():
        al.append_result(held_row("similar", 1.4))
        al.append_result(held_row("fresh", 1.1))
        brain = FakeBrain({"similar": alpha_json(), "fresh": alpha_json()}, corr={"similar": 0.81})
        assert run_held(brain) == "YES"
        assert submitted_value("similar") == "TOO_CORRELATED (max 0.81)"
        assert submitted_value("fresh") == "YES"
        assert submit_posts(brain) == ["fresh"]
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_held_alpha_already_submitted_on_brain_is_marked_yes_without_resubmitting(tmp_path):
    def run():
        al.append_result(held_row("by_hand", 1.3))
        al.append_result(held_row("next", 1.1))
        brain = FakeBrain({"by_hand": alpha_json(status="ACTIVE"), "next": alpha_json()})
        assert run_held(brain) == "YES"
        assert submitted_value("by_hand") == "YES" and submitted_value("next") == "YES"
        assert submit_posts(brain) == ["next"]   # the slot went to the next one in line
    in_tmp(tmp_path, run)


def test_held_alpha_that_now_fails_a_check_is_dropped(tmp_path):
    def run():
        al.append_result(held_row("changed", 1.3))
        failing = QUALIFYING_CHECKS[:2] + [{"name": "HIGH_TURNOVER", "result": "FAIL"}] + QUALIFYING_CHECKS[3:]
        brain = FakeBrain({"changed": alpha_json(checks=failing)})
        assert run_held(brain) is None
        assert submitted_value("changed") == "NOT_SUBMITTED (no longer qualifies: failed HIGH_TURNOVER)"
        assert submit_posts(brain) == [] and al.submissions_today() == 0
        assert al.held_candidates() == []
    in_tmp(tmp_path, run)


def test_held_alpha_with_an_unsettled_check_stays_queued(tmp_path):
    def run():
        al.append_result(held_row("unsettled", 1.3))
        pending = QUALIFYING_CHECKS[:2] + [{"name": "HIGH_TURNOVER", "result": "PENDING"}] + QUALIFYING_CHECKS[3:]
        brain = FakeBrain({"unsettled": alpha_json(checks=pending)})
        assert run_held(brain) is None
        assert submitted_value("unsettled") == al.HELD_LABEL
        assert submit_posts(brain) == []
    in_tmp(tmp_path, run)


def test_held_alpha_stays_queued_when_the_submit_fails(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.3))
        brain = FakeBrain({"h1": alpha_json()}, submit_post=500)
        assert run_held(brain) is None
        assert submitted_value("h1") == al.HELD_LABEL
        assert al.submissions_today() == 0
    in_tmp(tmp_path, run)


def test_held_queue_leaves_everything_queued_when_brain_is_unreachable(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.3))
        brain = FakeBrain({"h1": alpha_json()})
        brain.session.get.side_effect = _requests.ConnectionError("down")
        assert run_held(brain) is None
        assert submitted_value("h1") == al.HELD_LABEL
        assert not brain.session.post.called
    in_tmp(tmp_path, run)


def test_locked_csv_falls_back_to_appending_the_outcome(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.3))
        al.append_result(held_row("h2", 1.1))
        brain = FakeBrain({"h1": alpha_json(), "h2": alpha_json()})
        with patch.object(al.os, "replace", side_effect=PermissionError("in use")):
            assert run_held(brain) == "YES"
        rows = csv_rows()
        assert [(r["alpha_id"], r["submitted"], r["round"]) for r in rows] == [
            ("h1", al.HELD_LABEL, "1"), ("h2", al.HELD_LABEL, "1"), ("h1", "YES", "held")]
        assert [r["alpha_id"] for r in al.held_candidates()] == ["h2"]   # h1 counts as done
        assert not Path(al.RESULTS_FILE + ".tmp").exists()
    in_tmp(tmp_path, run)


def test_a_broken_held_queue_never_stops_the_run(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.3))
        brain = FakeBrain({"x": calm()})
        with patch.object(al, "_submit_held", side_effect=KeyError("surprise")):
            rows = run_main_try(brain, ["--try", "rank(close)"])
        assert [r["expression"] for r in rows if r["round"] == "try"] == ["rank(close)"]
    in_tmp(tmp_path, run)


def test_held_from_yesterday_goes_first_on_the_next_day_in_try_mode(tmp_path):
    def run():
        day = {"d": _date(2026, 10, 8)}
        with patch.object(al, "brain_day", side_effect=lambda *a: day["d"]):
            al.record_submission()                                  # yesterday's slot already used
            brain = FakeBrain({"held1": alpha_json(sharpe=1.34, fitness=1.05, turnover=0.1059)})
            run_one_idea(brain, THREE[0])                           # qualifies, so it's held
            assert submitted_value("held1") == al.HELD_LABEL

            day["d"] = _date(2026, 10, 9)                           # BRAIN's next day: slot free
            brain = FakeBrain({"new": calm(), "held1": alpha_json(sharpe=1.34, fitness=1.05, turnover=0.1059)})
            run_main_try(brain, ["--try", "rank(close)"])
            urls = [c.args[0] for c in brain.session.post.call_args_list]
            first_sim = next(i for i, u in enumerate(urls) if u.endswith("/simulations"))
            assert urls.index("https://api.worldquantbrain.com/alphas/held1/submit") < first_sim
            assert submitted_value("held1") == "YES"
            assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_normal_run_submits_held_alpha_before_asking_gemini(tmp_path):
    def run():
        _write_operators()
        al.append_result(held_row("held1", 1.05, expression=THREE[0]))
        brain = FakeBrain({"new": calm(), "held1": alpha_json()})  # the one simulation gets "new"
        order = []
        real_submit_held = al.submit_held

        def gen(*a, **kw):
            order.append("gemini")
            return [("rank(close)", "r")]

        def held(session):
            order.append("held")
            return real_submit_held(session)

        with patch.object(al, "MAX_ROUNDS", 1), patch.object(al, "generate_with_fallback", side_effect=gen), \
             patch.object(al, "submit_held", side_effect=held), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])
        assert order == ["held", "gemini"]
        assert submitted_value("held1") == "YES"
        assert [(r["expression"], r["alpha_id"]) for r in csv_rows()] == [(THREE[0], "held1"), ("rank(close)", "new")]
    in_tmp(tmp_path, run)


# ---- Similarity checked before holding; held queue re-checked every run ----

def test_too_similar_alpha_is_not_held(tmp_path):
    def run():
        al.record_submission()                                    # today's slot already used
        brain = FakeBrain({"a1": alpha_json()}, corr=0.85)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "TOO_CORRELATED (max 0.85)"
        assert al.held_candidates() == []                         # never enters the queue
        assert not any(c.args[0].endswith("/submit") for c in brain.session.post.call_args_list)
    in_tmp(tmp_path, run)


def test_different_enough_alpha_is_held(tmp_path):
    def run():
        al.record_submission()
        brain = FakeBrain({"a1": alpha_json()}, corr=0.31)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == al.HELD_LABEL
        assert [r["alpha_id"] for r in al.held_candidates()] == ["a1"]
    in_tmp(tmp_path, run)


def test_unreadable_similarity_still_holds(tmp_path):
    def run():
        al.record_submission()
        brain = FakeBrain({"a1": alpha_json()}, corr={"a1": None})  # BRAIN gives no number
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == al.HELD_LABEL              # checked again when its turn comes
    in_tmp(tmp_path, run)


def test_held_queue_drops_too_similar_even_when_slot_used(tmp_path):
    def run():
        al.append_result(held_row("dup", 1.5))
        al.append_result(held_row("fresh", 1.2))
        al.record_submission()
        brain = FakeBrain({"dup": alpha_json(), "fresh": alpha_json()}, corr={"dup": 0.92, "fresh": 0.22})
        assert run_held(brain) is None
        assert submitted_value("dup") == "TOO_CORRELATED (max 0.92)"
        assert submitted_value("fresh") == al.HELD_LABEL
        assert submit_posts(brain) == []
        assert [r["alpha_id"] for r in al.held_candidates()] == ["fresh"]
    in_tmp(tmp_path, run)


def test_held_queue_fills_two_slots_best_first(tmp_path):
    def run():
        for alpha_id, fitness in (("third", 1.1), ("best", 1.6), ("second", 1.4)):
            al.append_result(held_row(alpha_id, fitness))
        brain = FakeBrain({a: alpha_json() for a in ("third", "best", "second")})
        brain.submit_polls = -10  # skip FakeBrain's one 'still working' reply per submit
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
            assert run_held(brain) == "YES"
            assert al.submissions_today() == 2
        assert submit_posts(brain) == ["best", "second"]
        assert submitted_value("third") == al.HELD_LABEL
    in_tmp(tmp_path, run)


def test_held_queue_stops_when_its_time_is_up(tmp_path):
    def run():
        al.append_result(held_row("h1", 1.3))
        brain = FakeBrain({"h1": alpha_json()})
        with patch.object(al, "HELD_CHECK_BUDGET_SECONDS", -1):
            assert run_held(brain) is None
        assert not brain.session.get.called and submitted_value("h1") == al.HELD_LABEL
    in_tmp(tmp_path, run)


# ---- Parallel simulations ----

class SlowBrain:
    """Simulations that take a few checks to finish, with BRAIN's limit on how many run at once.
    Each expression's result is fixed in advance, so mixed-up results would show."""
    def __init__(self, results, polls=None, limit=3, busy_first=0, expire_on=None, renew_ok=True,
                 corr=0.3, existing=None):
        self.results = results              # {expression: alpha json}
        self.corr = corr                    # similarity reading: a number, or {alpha_id: number}
        self.polls = polls or {}            # {expression: checks needed before it's done} (default 1)
        self.limit, self.busy_first = limit, busy_first
        self.expire_on, self.renew_ok = expire_on, renew_ok
        self.running, self.started, self.peak, self.sim = {}, [], 0, 0
        self.alpha_of = dict(existing or {})  # {alpha_id: expression}, incl. alphas from earlier runs
        self.session = MagicMock()
        self.session.get.side_effect = self.get
        self.session.post.side_effect = self.post

    def post(self, url, **kw):
        if url == be.AUTH_URL:
            return fake_response(201, {"user": {"id": "u"}}) if self.renew_ok else fake_response(401, text="no")
        if url.endswith("/simulations"):
            if self.busy_first > 0 or len(self.running) >= self.limit:
                self.busy_first -= 1
                return fake_response(429, {"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"},
                                     text='{"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"}')
            self.sim += 1
            expression = kw["json"]["regular"]
            self.running[f"sim-{self.sim}"] = [expression, self.polls.get(expression, 1)]
            self.started.append(expression)
            self.peak = max(self.peak, len(self.running))
            return fake_response(201, headers={"Location": f"sim-{self.sim}"})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, **kw):
        if url == be.AUTH_URL:
            return fake_response(200, {"token": {"expiry": 14400}})
        if url.endswith("/correlations/self"):
            alpha_id = url.split("/")[-3]
            corr = self.corr.get(alpha_id, 0.3) if isinstance(self.corr, dict) else self.corr
            return fake_response(200, {"max": corr}, text=json.dumps({"max": corr}))
        if url.startswith("sim-"):
            expression, left = self.running[url]
            if expression == self.expire_on:
                self.expire_on = None
                return fake_response(401, {"detail": "Incorrect authentication credentials."})
            if left > 1:
                self.running[url][1] -= 1
                return fake_response(200, {"progress": 0.5})
            del self.running[url]
            self.alpha_of[f"id{url[4:]}"] = expression
            return fake_response(200, {"status": "COMPLETE", "alpha": f"id{url[4:]}"})
        return fake_response(200, self.results[self.alpha_of[url.rsplit("/", 1)[-1]]])


def run_queue(brain, expressions, parallel=3):
    items = [(e, "r", "idea") for e in expressions]
    with patch("time.sleep"), patch.object(al, "PARALLEL_SIMULATIONS", parallel):
        return al.process_queue(brain.session, items, SETTINGS, "e@x.com", "pw", 1, set())


def test_runs_three_simulations_at_once_and_keeps_results_straight(tmp_path):
    def run():
        exprs = [f"rank(x{i})" for i in range(6)]
        brain = SlowBrain({e: calm(sharpe=0.1 * i) for i, e in enumerate(exprs)},
                          polls={e: 4 - i % 3 for i, e in enumerate(exprs)})
        rows = run_queue(brain, exprs)
        assert brain.peak == 3
        assert sorted(r["expression"] for r in rows) == exprs and all(r["status"] == "COMPLETE" for r in rows)
        for r in rows:  # each row carries its own expression's result
            assert r["sharpe"] == 0.1 * int(r["expression"][6:-1])
            assert brain.alpha_of[r["alpha_id"]] == r["expression"]
        assert len(csv_rows()) == 6
    in_tmp(tmp_path, run)


def test_quick_results_dont_wait_for_a_slow_one(tmp_path):
    def run():
        brain = SlowBrain({e: calm() for e in ("rank(slow)", "rank(a)", "rank(b)", "rank(c)")},
                          polls={"rank(slow)": 6})
        rows = run_queue(brain, ["rank(slow)", "rank(a)", "rank(b)", "rank(c)"])
        assert [r["expression"] for r in rows][-1] == "rank(slow)"
        assert brain.started == ["rank(slow)", "rank(a)", "rank(b)", "rank(c)"]
    in_tmp(tmp_path, run)


def test_brain_busy_drops_to_what_it_allows(tmp_path):
    def run():
        exprs = [f"rank(x{i})" for i in range(5)]
        brain = SlowBrain({e: calm() for e in exprs}, polls={e: 2 for e in exprs}, limit=2)
        rows = run_queue(brain, exprs)
        assert brain.peak == 2
        assert all(r["status"] == "COMPLETE" for r in rows) and len(rows) == 5   # nothing lost to the 429s
    in_tmp(tmp_path, run)


def test_busy_with_nothing_running_waits_then_starts(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()}, busy_first=2)
        with patch("time.sleep") as sleep, patch.object(al, "PARALLEL_SIMULATIONS", 3):
            rows = al.process_queue(brain.session, [("rank(a)", "r", "idea")], SETTINGS, "e@x.com", "pw", 1, set())
        assert [r["status"] for r in rows] == ["COMPLETE"]
        assert [c.args[0] for c in sleep.call_args_list].count(al.BUSY_WAIT_SECONDS) == 2
    in_tmp(tmp_path, run)


def test_busy_too_long_gives_up_on_that_idea_only(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()}, limit=0)
        with patch.object(al, "MAX_BUSY_WAIT_SECONDS", 0):
            rows = run_queue(brain, ["rank(a)"])
        assert rows[0]["status"] == "FAILED" and "stayed busy" in rows[0]["error"]
    in_tmp(tmp_path, run)


def test_followup_takes_the_next_free_slot(tmp_path):
    def run():
        results = {"rank(a)": alpha_json(sharpe=-1.5, checks=[]), "reverse(rank(a))": calm(),
                   "rank(b)": calm(), "rank(c)": calm(), "rank(d)": calm()}
        brain = SlowBrain(results, polls={"rank(b)": 5})
        run_queue(brain, ["rank(a)", "rank(b)", "rank(c)", "rank(d)"], parallel=2)
        assert brain.started[:3] == ["rank(a)", "rank(b)", "reverse(rank(a))"]   # ahead of c and d
    in_tmp(tmp_path, run)


def test_expired_session_while_checking_renews_and_carries_on(tmp_path):
    def run():
        brain = SlowBrain({e: calm() for e in ("rank(a)", "rank(b)")}, expire_on="rank(a)")
        rows = run_queue(brain, ["rank(a)", "rank(b)"])
        assert sorted(r["status"] for r in rows) == ["COMPLETE", "COMPLETE"]
    in_tmp(tmp_path, run)


def test_session_lost_while_checking_stops_and_records_every_running_idea(tmp_path):
    def run():
        brain = SlowBrain({e: calm() for e in ("rank(a)", "rank(b)", "rank(c)")},
                          polls={"rank(b)": 3, "rank(c)": 3}, expire_on="rank(a)", renew_ok=False)
        try:
            run_queue(brain, ["rank(a)", "rank(b)", "rank(c)"])
            assert False, "should have stopped"
        except al.StopLoop:
            pass
        saved = {r["expression"]: (r["status"], r["error"]) for r in csv_rows()}
        assert saved["rank(a)"] == ("STOPPED", "session expired and could not renew unattended")
        assert saved["rank(b)"][0] == saved["rank(c)"][0] == "STOPPED"
        assert "retry it with --try" in saved["rank(b)"][1]
    in_tmp(tmp_path, run)


def test_submissions_never_overlap_when_several_qualify(tmp_path):
    def run():
        good = alpha_json(sharpe=1.6, fitness=1.2, turnover=0.2)
        brain = SlowBrain({"rank(a)": good, "rank(b)": good, "rank(c)": calm()})
        submitted = []

        def fake_try_submit(session, alpha_id):
            submitted.append(alpha_id)
            al.record_submission()
            return "YES"

        with patch.object(al, "try_submit", side_effect=fake_try_submit):
            rows = run_queue(brain, ["rank(a)", "rank(b)", "rank(c)"])
        assert len(submitted) == 1                                     # daily cap still holds
        by_expr = {r["expression"]: r["submitted"] for r in rows}
        assert sorted(by_expr.values()) == ["NO", al.HELD_LABEL, "YES"]
    in_tmp(tmp_path, run)


# ---- Variety across data types ----

def write_cache(categories):
    Path(al.FIELDS_CACHE_FILE).write_text(json.dumps({"matrix_fields": [
        {"id": name, "category": {"id": cat[:3].lower(), "name": cat}} for name, cat in categories.items()]}),
        encoding="utf-8")


FOUR_TYPES = {"implied_volatility_call_60": "Option", "implied_volatility_put_90": "Option",
              "sales": "Fundamental", "debt": "Fundamental", "est_eps": "Analyst", "close": "Price Volume"}
FIELDS = [{"name": n, "description": "d [some dataset]"} for n in FOUR_TYPES] + \
         [{"name": "industry", "description": "group field - use as the group argument of group_* operators. [x]"}]


def test_field_types_prefers_brain_categories_then_dataset_tag(tmp_path):
    def run():
        fields = [{"name": "iv", "description": "implied vol [Volatility Data]"},
                  {"name": "sales", "description": "Sales [Fundamental Data]"},
                  {"name": "mystery", "description": "no tag here"},
                  {"name": "sector", "description": "group field - use as the group argument. [Univ]"}]
        assert al.field_types(fields) == {"iv": "Volatility Data", "sales": "Fundamental Data"}
        write_cache({"iv": "Option", "sales": "Fundamental"})
        assert al.field_types(fields) == {"iv": "Option", "sales": "Fundamental"}
        Path(al.FIELDS_CACHE_FILE).write_text("not json", encoding="utf-8")
        assert al.field_types(fields)["iv"] == "Volatility Data"         # broken cache: falls back
    in_tmp(tmp_path, run)


def test_crowded_types_come_from_submitted_queued_and_too_similar(tmp_path):
    def run():
        for expr, submitted in [("rank(implied_volatility_put_90)", "YES"),
                                ("rank(implied_volatility_call_60)", al.HELD_LABEL),
                                ("rank(est_eps)", "TOO_CORRELATED (max 0.81)"),
                                ("rank(sales)", "NO"), ("rank(debt)", "REJECTED: failed LOW_SHARPE")]:
            al.append_result(dict(_history_row(expr, 1.3, 1.0, 0.1), submitted=submitted))
        assert al.crowded_types(FOUR_TYPES) == Counter({"Option": 2, "Analyst": 1})
    in_tmp(tmp_path, run)


def test_pick_varied_caps_crowded_and_busy_types(tmp_path):
    ideas = [(f"rank(implied_volatility_call_60) + {i}", "r") for i in range(3)] + \
            [("rank(sales)", "r"), ("rank(debt)", "r"), ("rank(sales * debt)", "r"),
             ("rank(est_eps)", "r"), ("rank(close)", "r"), ("rank(volume_ratio)", "r")]
    kept = [e for e, _ in al.pick_varied(ideas, FOUR_TYPES, Counter({"Option": 1}), 8)]
    assert kept == ["rank(implied_volatility_call_60) + 0", "rank(sales)", "rank(debt)",
                    "rank(est_eps)", "rank(close)", "rank(volume_ratio)"]   # unknown fields aren't capped
    assert len(al.pick_varied(ideas, FOUR_TYPES, Counter(), 2)) == 2


def test_round_prompt_steers_away_from_submitted_data(tmp_path):
    def run():
        write_cache(FOUR_TYPES)
        al.append_result(dict(_history_row("rank(implied_volatility_put_90)", 1.63, 1.27, 0.16), submitted="YES"))
        returned = [{"expression": f"rank(ts_delta(implied_volatility_call_60, {i}))", "rationale": "r"} for i in range(5)] + \
                   [{"expression": "rank(sales)", "rationale": "r"}, {"expression": "rank(debt)", "rationale": "r"},
                    {"expression": "rank(sales / debt)", "rationale": "r"}, {"expression": "rank(est_eps)", "rationale": "r"}]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(returned)
        brain = SlowBrain({e["expression"]: calm() for e in returned})
        with patch("time.sleep"), patch.object(al, "MAX_IDEAS_PER_CROWDED_SHAPE", 99):  # data types only here
            rows = al.run_round(brain.session, client, [], FIELDS, SETTINGS, "e@x.com", "pw", 1,
                                al.build_feedback(), set())
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        assert "- implied_volatility_call_60: [type: Option]" in prompt
        assert "- industry: group field" in prompt                         # group fields unlabelled
        assert "already use: Option (1)" in prompt
        assert "favour these: Analyst, Fundamental, Price Volume" in prompt
        assert f"Propose {al.IDEAS_PER_ROUND + al.VARIETY_EXTRA_IDEAS} NEW" in prompt
        assert "rank(implied_volatility_put_90) -> sharpe=1.63" in prompt and "[submitted]" in prompt
        assert "keep the data" not in prompt
        assert sorted(r["expression"] for r in rows) == [
            "rank(debt)", "rank(est_eps)", "rank(sales)", "rank(ts_delta(implied_volatility_call_60, 0))"]
    in_tmp(tmp_path, run)


def test_no_variety_rule_without_data_types(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
        brain = SlowBrain({"rank(close)": calm()})
        with patch("time.sleep"):
            al.run_round(brain.session, client, [], [{"name": "close", "description": "d"}], SETTINGS,
                         "e@x.com", "pw", 1, "round 1", set())
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        assert "VARIETY RULE" not in prompt and f"Propose {al.IDEAS_PER_ROUND} NEW" in prompt
    in_tmp(tmp_path, run)


# ---- Auto-repair of strong alphas that fail a fixable check ----

DIVIDEND = "ts_decay_linear(group_rank(ts_delta(dividend, 180), market), 20)"
CW_FAIL = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "LOW_FITNESS", "result": "PASS"},
           {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]
SUB_FAIL = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]
RANKED = f"rank({DIVIDEND})"
BACKFILLED = "ts_decay_linear(group_rank(ts_delta(ts_backfill(dividend, 120), 180), market), 20)"
FINER = "ts_decay_linear(group_rank(ts_delta(dividend, 180), subindustry), 20)"


def near_miss(checks=None):
    return alpha_json(sharpe=1.35, fitness=1.34, turnover=0.13, checks=CW_FAIL if checks is None else checks)


def test_strong_alpha_failing_a_fixable_check_is_repaired_right_after(tmp_path):
    def run():
        brain = SlowBrain({DIVIDEND: near_miss(), RANKED: calm(), BACKFILLED: calm(), "rank(b)": calm()})
        rows = run_queue(brain, [DIVIDEND, "rank(b)"], parallel=1)
        assert brain.started == [DIVIDEND, RANKED, BACKFILLED, "rank(b)"]
        repair = next(r for r in rows if r["expression"] == RANKED)
        assert repair["rationale"].startswith(f"Auto-repair of {DIVIDEND}: failed CONCENTRATED_WEIGHT; rank()")
    in_tmp(tmp_path, run)


def test_a_repair_that_passes_is_submitted(tmp_path):
    def run():
        brain = SlowBrain({DIVIDEND: near_miss(SUB_FAIL), FINER: alpha_json(sharpe=1.5, fitness=1.3, turnover=0.12)})
        brain.session.post.side_effect = lambda url, **kw: (
            fake_response(201, {}, headers={}) if url.endswith("/submit") else brain.post(url, **kw))
        with patch.object(al, "submit_alpha", return_value=("YES", None)), \
             patch.object(al, "is_active_on_brain", return_value=True):
            rows = run_queue(brain, [DIVIDEND], parallel=1)
        assert {r["expression"]: r["submitted"] for r in rows} == {DIVIDEND: "NO", FINER: "YES"}
    in_tmp(tmp_path, run)


def test_no_repair_when_the_original_is_too_similar(tmp_path):
    def run():
        brain = SlowBrain({DIVIDEND: near_miss()}, corr=0.88)
        rows = run_queue(brain, [DIVIDEND], parallel=1)
        assert brain.started == [DIVIDEND] and len(rows) == 1
    in_tmp(tmp_path, run)


def test_no_repair_for_checks_we_cant_fix_or_weak_numbers(tmp_path):
    def run():
        unfixable = [{"name": "MATCHES_COMPETITION", "result": "FAIL"}] + CW_FAIL
        weak = alpha_json(sharpe=1.0, fitness=0.6, turnover=0.13, checks=CW_FAIL)
        brain = SlowBrain({"rank(a)": near_miss(unfixable), "rank(b)": weak})
        run_queue(brain, ["rank(a)", "rank(b)"], parallel=1)
        assert brain.started == ["rank(a)", "rank(b)"]
    in_tmp(tmp_path, run)


def test_repairs_are_not_repaired_again_and_are_capped(tmp_path):
    def run():
        exprs = [f"group_rank(ts_delta(dividend, {d}), market)" for d in (30, 60, 90)]
        results = {e: near_miss() for e in exprs}
        for e in exprs:   # every repair fails the same check again
            results[f"rank({e})"] = near_miss()
            results[e.replace("dividend", "ts_backfill(dividend, 120)")] = near_miss()
        brain = SlowBrain(results)
        rows = run_queue(brain, exprs, parallel=1)
        repairs = [r for r in rows if r["rationale"].startswith("Auto-repair")]
        assert len(repairs) == al.MAX_REPAIRS_PER_ROUND == 4
        assert not any(r["rationale"].startswith("Auto-repair of rank(") for r in repairs)
    in_tmp(tmp_path, run)


def test_earlier_near_miss_is_repaired_at_the_start_of_a_run(tmp_path):
    def run():
        _write_operators()
        al.append_result(dict(_history_row(DIVIDEND, 1.35, 1.34, 0.1349), alpha_id="div", submitted="NO",
                              timestamp="2026-10-10 12:14:01"))
        brain = SlowBrain({DIVIDEND: near_miss(), RANKED: calm(), BACKFILLED: calm(), "rank(close)": calm()},
                          existing={"div": DIVIDEND})
        gen = MagicMock(return_value=[("rank(close)", "r")])
        with patch.object(al, "MAX_ROUNDS", 1), patch.object(al, "generate_with_fallback", gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])
        assert brain.started[:2] == [RANKED, BACKFILLED]            # repairs before Gemini's ideas
        assert {r["expression"]: r["round"] for r in csv_rows()}[RANKED] == "repair"
        assert "rank(close)" in brain.started
    in_tmp(tmp_path, run)


def test_earlier_near_misses_are_only_looked_at_until_every_fix_was_tried(tmp_path):
    def run():
        al.append_result(dict(_history_row(DIVIDEND, 1.35, 1.34, 0.13), alpha_id="div", submitted="NO"))
        brain = SlowBrain({}, existing={"div": DIVIDEND})
        tried = set(al.all_repair_variants(DIVIDEND))
        with patch("time.sleep"):
            assert al.repair_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", tried) == []
        assert not brain.session.get.called                        # nothing left to try: BRAIN not asked
    in_tmp(tmp_path, run)


def test_near_miss_rows_skip_submitted_held_and_weak(tmp_path):
    def run():
        for alpha_id, submitted, sharpe, fitness in [("nm", "NO", 1.35, 1.34), ("sub", "YES", 1.6, 1.3),
                                                     ("held", al.HELD_LABEL, 1.5, 1.2), ("weak", "NO", 1.1, 0.7)]:
            al.append_result(dict(_history_row(f"rank({alpha_id})", sharpe, fitness, 0.1),
                                  alpha_id=alpha_id, submitted=submitted))
        assert [r["alpha_id"] for r in al.near_miss_rows()] == ["nm"]
    in_tmp(tmp_path, run)


def test_repair_helpers_read_expressions_carefully():
    expr = "rank(ts_regression(returns, rel_ret_comp, 60, lag=0, rettype=0)) * 1e-3"
    assert [n for _, _, n in al.data_fields_in(expr)] == ["returns", "rel_ret_comp"]
    grouped = "group_neutralize(rank(-returns), bucket(rank(cap), range='0, 1, 0.1'))"
    assert [n for _, _, n in al.data_fields_in(grouped)] == ["returns", "cap"]
    assert al.split_top_level("a, f(b, c), 'x, y'") == ["a", " f(b, c)", " 'x, y'"]
    assert al.outer_call("rank(x) + 1") is None and al.outer_call("rank(x)") == ("rank", ["x"])
    assert al.repair_candidates(DIVIDEND, ["CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"]) [1][0] == FINER
    assert al.repair_candidates(DIVIDEND, ["SELF_CORRELATION"]) == []


# ---- Variety of formula shape ----

TEMPLATE = [  # the shape every submitted alpha so far shares
    "ts_decay_linear(group_rank(ts_zscore(rel_ret_all, 60), sector), 10)",
    "ts_decay_linear(group_rank(ts_delta(dividend, 180), market), 20)",
    "ts_decay_linear(group_rank(ts_av_diff(unsystematic_risk_last_360_days, 60), subindustry), 10)",
]


def test_skeleton_ignores_data_windows_groups_and_which_ts_operator():
    assert {al.skeleton(e) for e in TEMPLATE} == {"group_*(ts_*(F,N),G)"}
    assert al.skeleton("rank(ts_regression(returns, rel_ret_comp, 60, lag=0, rettype=0))") == \
        "rank(ts_*(F,F,N,lag=N,rettype=N))"
    assert al.skeleton("group_neutralize(rank(-returns), bucket(rank(cap), range='0, 1, 0.1'))") == \
        "group_*(rank(-F),G)"
    assert al.skeleton("vwap/close") == "F/F"


def test_smoothing_is_not_part_of_the_shape():
    raw = "group_rank(ts_delta(close, 30), sector)"
    for smoothed in (f"ts_decay_linear({raw}, 20)", f"hump({raw}, hump=0.01)",
                     f"ts_decay_linear(ts_decay_linear({raw}, 10), 10)", f"ts_decay_linear(hump({raw}, hump=0.01), 5)"):
        assert al.skeleton(smoothed) == al.skeleton(raw) == "group_*(ts_*(F,N),G)"
    assert al.skeleton("rank(ts_decay_linear(close, 5))") != al.skeleton("rank(close)")   # inner decay is part of it
    rule = al.shape_rule({"group_*(ts_*(F,N),G)": TEMPLATE[0]}, [])
    assert "Smoothing is NOT part of the shape - keep using it" in rule and "hump(x, hump=0.01)" in rule


def test_over_the_limit_turnover_is_smoothed_from_sharpe_1():
    assert al.needs_smoothing(1.14, 0.37, 1.42) is True     # the flipped ts_regression from 10 Oct
    assert al.needs_smoothing(0.82, 0.25, 0.49) is False    # under the limit: the usual 1.25 bar applies
    assert al.needs_smoothing(0.58, 0.06, 1.16) is False    # too weak to be worth a simulation


def test_crowded_shapes_from_the_pool_and_from_recent_history(tmp_path):
    def run():
        assert al.crowded_shapes() == {}
        al.append_result(dict(_history_row(TEMPLATE[0], 1.86, 1.71, 0.13), submitted="YES"))
        assert list(al.crowded_shapes()) == ["group_*(ts_*(F,N),G)"]
        for i in range(12):   # a run of one other shape: 12 of 13 recent simulations
            al.append_result(_history_row(f"rank(field_{i})", 0.4, 0.2, 0.1))
        assert set(al.crowded_shapes()) == {"group_*(ts_*(F,N),G)", "rank(F)"}
    in_tmp(tmp_path, run)


def test_round_caps_the_overused_shape_and_suggests_others(tmp_path):
    def run():
        al.append_result(dict(_history_row(TEMPLATE[0], 1.86, 1.71, 0.13), submitted="YES"))
        returned = [{"expression": f"ts_decay_linear(group_rank(ts_delta(field_{i}, 30), sector), 10)",
                     "rationale": "r"} for i in range(5)] + \
                   [{"expression": "rank(divide(sales, debt))", "rationale": "r"},
                    {"expression": "-ts_corr(rank(close), rank(volume), 20)", "rationale": "r"}]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(returned)
        brain = SlowBrain({e["expression"]: calm() for e in returned})
        operators = [{"name": n, "usage": "u"} for n in ("rank", "divide", "ts_corr", "ts_delta", "group_rank",
                                                         "ts_decay_linear")]
        with patch("time.sleep"):
            rows = al.run_round(brain.session, client, operators, [], SETTINGS, "e@x.com", "pw", 1,
                                al.build_feedback(), set())
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        assert "SHAPE RULE" in prompt and TEMPLATE[0] in prompt
        assert "rank(divide(x, y))" in prompt and "-ts_corr(rank(x), rank(y), d)" in prompt
        assert "ts_regression(y, x, d)" not in prompt          # operator not on this account's list
        assert f"Propose {al.IDEAS_PER_ROUND + al.VARIETY_EXTRA_IDEAS} NEW" in prompt
        shapes = Counter(al.skeleton(r["expression"]) for r in rows)
        assert shapes["group_*(ts_*(F,N),G)"] == al.MAX_IDEAS_PER_CROWDED_SHAPE == 2
        assert len(rows) == 4                                   # 2 of the template + the 2 new shapes
    in_tmp(tmp_path, run)


def test_no_shape_rule_without_an_overused_shape(tmp_path):
    def run():
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
        brain = SlowBrain({"rank(close)": calm()})
        with patch("time.sleep"):
            al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert "SHAPE RULE" not in client.models.generate_content.call_args.kwargs["contents"]
    in_tmp(tmp_path, run)


# ---- Daily scheduled run: one run at a time, log to a file ----

def test_only_one_run_at_a_time(tmp_path):
    def run():
        first = al.acquire_run_lock()
        assert first is not None
        assert al.acquire_run_lock() is None          # a second run can't get it
        first.close()                                 # the first run ends (or crashes)
        again = al.acquire_run_lock()
        assert again is not None
        again.close()
    in_tmp(tmp_path, run)


def test_a_second_run_exits_without_touching_brain(tmp_path):
    def run():
        held = al.acquire_run_lock()
        try:
            with patch.object(al, "get_authenticated_session", side_effect=AssertionError("must not log in")):
                al.main(["--try", "rank(close)"])
        finally:
            held.close()
        assert not Path(al.RESULTS_FILE).exists()
    in_tmp(tmp_path, run)


def test_windows_lock_uses_msvcrt(tmp_path):
    def run():
        fake = MagicMock()
        fake.LK_NBLCK = 2
        with patch.dict(sys.modules, {"msvcrt": fake}), patch.object(al.os, "name", "nt"):
            handle = al.acquire_run_lock()
            fake.locking.assert_called_once_with(handle.fileno(), 2, 1)
            handle.close()
            fake.locking.side_effect = OSError(13, "Permission denied")  # another run holds it
            assert al.acquire_run_lock() is None
    in_tmp(tmp_path, run)


def test_log_dir_keeps_a_dated_utf8_log(tmp_path):
    def run():
        import logging, time as _time
        handlers_before = list(logging.getLogger().handlers)
        brain = FakeBrain({"a": calm()})
        with patch.object(al, "get_authenticated_session", return_value=brain.session), patch("time.sleep"):
            al.main(["--try", "rank(close)", "--log-dir", "logs"])
        log = Path("logs") / f"alpha_loop_{_time.strftime('%Y-%m-%d')}.log"
        text = log.read_text(encoding="utf-8")
        assert "Trying 1 expression(s) you gave." in text and "Done: 1 simulation(s)" in text
        assert logging.getLogger().handlers == handlers_before    # file handler removed afterwards
    in_tmp(tmp_path, run)


# ---- Staying logged in (the expired-login bug from the 10 Oct log) ----

def login_session(auth_reply, alpha_reply=None):
    """A session whose GET /authentication answers auth_reply() and alpha lookups alpha_reply."""
    state = {"logged_in": False}
    session = MagicMock()

    def get(url, **kw):
        if url == be.AUTH_URL:
            return fake_response(200, {"token": {"expiry": 14400}}) if state["logged_in"] else auth_reply()
        return alpha_reply or fake_response(404, {"detail": "Not found."})

    def authenticate(sess, email, password):
        state["logged_in"] = True
        return True

    session.get.side_effect = get
    return session, authenticate


def test_saved_login_reused_only_with_enough_time_left(tmp_path):
    def run():
        Path(al.SESSION_FILE).write_text('{"sid": "saved"}', encoding="utf-8")
        for expiry, expect_login in ((7200, False), (600, True)):
            session, authenticate = login_session(lambda: fake_response(200, {"token": {"expiry": expiry}}))
            with patch.object(al.requests, "Session", return_value=session), \
                 patch.object(be, "authenticate", side_effect=authenticate) as auth:
                assert al.get_authenticated_session("e@x.com", "pw") is session
            assert auth.called == expect_login
            assert abs(session.brain_expires_at - (_time.time() + (14400 if expect_login else 7200))) < 60
    in_tmp(tmp_path, run)


def test_the_expired_login_from_the_log_is_caught(tmp_path):
    def run():
        # What fooled the old check: logged out, BRAIN answered alpha lookups with "not found".
        Path(al.SESSION_FILE).write_text('{"sid": "stale"}', encoding="utf-8")
        session, authenticate = login_session(
            lambda: fake_response(401, {"detail": "Incorrect authentication credentials."}))
        with patch.object(al.requests, "Session", return_value=session), \
             patch.object(be, "authenticate", side_effect=authenticate) as auth:
            al.get_authenticated_session("e@x.com", "pw")
        auth.assert_called_once()                                  # logged in fresh, not reused
    in_tmp(tmp_path, run)


def test_a_login_brain_cant_confirm_is_renewed(tmp_path):
    def run():
        Path(al.SESSION_FILE).write_text('{"sid": "saved"}', encoding="utf-8")
        session, authenticate = login_session(lambda: fake_response(200, {"something": "else"}))
        with patch.object(al.requests, "Session", return_value=session), \
             patch.object(be, "authenticate", side_effect=authenticate) as auth:
            al.get_authenticated_session("e@x.com", "pw")
        auth.assert_called_once()
    in_tmp(tmp_path, run)


def test_login_renewed_before_it_runs_out_mid_run(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm(), "rank(b)": calm()})
        brain.session.brain_expires_at = _time.time() + 60          # under 5 minutes left
        rows = run_queue(brain, ["rank(a)"])
        posts = [c.args[0] for c in brain.session.post.call_args_list]
        assert posts[0] == be.AUTH_URL and posts[1].endswith("/simulations")   # logged in first
        assert brain.session.brain_expires_at > _time.time() + 3600
        assert rows[0]["status"] == "COMPLETE"
        brain.session.post.reset_mock()
        run_queue(brain, ["rank(b)"])                               # plenty of time left now
        assert be.AUTH_URL not in [c.args[0] for c in brain.session.post.call_args_list]
    in_tmp(tmp_path, run)


def test_expired_login_while_reading_a_result_is_renewed_and_retried(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm(sharpe=0.7)})
        real_get, state = brain.get, {"first": True}

        def get(url, **kw):
            if "/alphas/" in url and state["first"]:
                state["first"] = False
                return fake_response(401, {"detail": "Incorrect authentication credentials."})
            return real_get(url, **kw)

        brain.session.get.side_effect = get
        rows = run_queue(brain, ["rank(a)"])
        assert rows[0]["status"] == "COMPLETE" and rows[0]["sharpe"] == 0.7
        assert [c.args[0] for c in brain.session.post.call_args_list].count(be.AUTH_URL) == 1
    in_tmp(tmp_path, run)


def test_a_finished_simulation_without_a_result_is_failed_not_blank(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()})
        real_get = brain.get
        brain.session.get.side_effect = lambda url, **kw: (
            fake_response(404, {"detail": "Not found."}) if "/alphas/" in url else real_get(url, **kw))
        rows = run_queue(brain, ["rank(a)"])
        assert rows[0]["status"] == "FAILED" and "didn't send its result" in rows[0]["error"]
        assert rows[0]["alpha_link"].endswith(rows[0]["alpha_id"])
    in_tmp(tmp_path, run)


def test_identity_check_in_an_unattended_run_stops_cleanly(tmp_path):
    def run():
        offline = MagicMock()
        offline.get.side_effect = _requests.ConnectionError("no answer")
        with patch.object(be, "authenticate", side_effect=EOFError("no keyboard")), \
             patch.object(al.requests, "Session", return_value=offline):
            al.main(["--try", "rank(close)"])                       # returns, doesn't crash
        assert not Path(al.RESULTS_FILE).exists()
    in_tmp(tmp_path, run)


def test_a_crash_is_written_to_the_log_file(tmp_path):
    def run():
        with patch.object(al, "run", side_effect=ValueError("boom")):
            try:
                al.main(["--log-dir", "logs"])
                assert False, "should have raised"
            except ValueError:
                pass
        text = next(Path("logs").glob("*.log")).read_text(encoding="utf-8")
        assert "The run crashed" in text and "ValueError: boom" in text
    in_tmp(tmp_path, run)


# ---- Learning what doesn't work: rejected names and dead-end fields ----

def failed_row(expression, error):
    return dict(_history_row(expression, "", "", "", status="FAILED"), error=error)


def test_names_brain_rejected_are_read_from_the_errors(tmp_path):
    def run():
        al.append_result(failed_row("rank(divide(net_income, assets))",
                                    'Attempted to use unknown variable "net_income". <linkToCommonErrorMessages>'))
        al.append_result(failed_row("trade_when(gt(news_max_up_ret, 0.05), rank(close), -1)",
                                    'Attempted to use inaccessible or unknown operator "gt". <link>'))
        al.append_result(failed_row("rank(close)", "timed out after 1200s"))
        assert al.rejected_names() == {"net_income", "gt"}
    in_tmp(tmp_path, run)


def test_dead_end_fields_need_enough_tries_and_spare_the_basics(tmp_path):
    def run():
        for i in range(8):
            al.append_result(_history_row(f"rank(ts_delta(news_max_up_ret, {i + 2}))", 0.1, 0.18 - i / 100, 0.3))
            al.append_result(_history_row(f"rank(volume) * {i}", 0.1, 0.1, 0.3))            # basic: exempt
            al.append_result(_history_row(f"rank(snt_buzz) + {i}", 1.0, 0.6 if i == 3 else 0.2, 0.3))  # one decent result
        for i in range(7):
            al.append_result(_history_row(f"rank(adv_thing_{0}) + {i}", 0.1, 0.1, 0.3))     # only 7 tries
        assert al.dead_end_fields() == {"news_max_up_ret": (8, 0.18)}
    in_tmp(tmp_path, run)


def test_round_leaves_out_rejected_names_and_dead_ends(tmp_path):
    def run():
        al.append_result(failed_row("trade_when(gt(sales, 0), rank(sales), -1)",
                                    'Attempted to use inaccessible or unknown operator "gt".'))
        for i in range(8):
            al.append_result(_history_row(f"ts_rank(news_max_up_ret, {10 * i + 10})", -0.3, 0.05, 0.25))
        returned = [{"expression": "trade_when(gt(volume, adv20), rank(sales), -1)", "rationale": "r"},
                    {"expression": "rank(ts_delta(news_max_up_ret, 5))", "rationale": "r"},
                    {"expression": "rank(divide(sales, assets))", "rationale": "r"}]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(returned)
        brain = SlowBrain({"rank(divide(sales, assets))": calm()})
        operators = [{"name": n, "usage": f"{n}(...)"} for n in ("gt", "rank", "divide", "trade_when")]
        fields = [{"name": n, "description": "d"} for n in ("news_max_up_ret", "sales", "assets")]
        with patch("time.sleep"):
            rows = al.run_round(brain.session, client, operators, fields, SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert [r["expression"] for r in rows] == ["rank(divide(sales, assets))"]   # the other two cost nothing
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        assert "never use them: gt." in prompt and "- gt: gt(...)" not in prompt
        assert "news_max_up_ret (8 tries, best 0.05)" in prompt and "- news_max_up_ret: d" not in prompt
    in_tmp(tmp_path, run)


# ---- Auto-sharpen slow near-misses; log the checks we can't fix ----

import contextlib
import logging as _logging

SLOW = "ts_decay_linear(group_rank(ts_delta(anl4_ebit_value, 60), market), 10)"   # 1.50 / 0.98 / 4.9% on 10 Oct
SHARPER = "group_rank(ts_delta(anl4_ebit_value, 60), market)"
FITNESS_FAIL = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "LOW_FITNESS", "result": "FAIL"},
                {"name": "SELF_CORRELATION", "result": "PENDING"}]


def slow_near_miss():
    return alpha_json(sharpe=1.5, fitness=0.98, turnover=0.0489, checks=FITNESS_FAIL)


@contextlib.contextmanager
def captured_logs():
    lines = []

    class Grab(_logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Grab()
    al.logger.addHandler(handler)
    try:
        yield lines
    finally:
        al.logger.removeHandler(handler)


def test_slow_alpha_with_good_sharpe_is_tested_without_its_smoothing(tmp_path):
    def run():
        brain = SlowBrain({SLOW: slow_near_miss(), SHARPER: calm(), "rank(b)": calm()})
        rows = run_queue(brain, [SLOW, "rank(b)"], parallel=1)
        assert brain.started == [SLOW, SHARPER, "rank(b)"]
        sharpened = next(r for r in rows if r["expression"] == SHARPER)
        assert sharpened["rationale"].startswith(f"Auto-sharpen of {SLOW}: Sharpe 1.50 but fitness 0.98 at 4.9%")
    in_tmp(tmp_path, run)


def test_a_sharpened_result_is_not_smoothed_back(tmp_path):
    def run():
        slow20 = SLOW.replace(", 10)", ", 20)")          # its 10-day smoothing would be a new expression
        brain = SlowBrain({slow20: slow_near_miss(),
                           SHARPER: alpha_json(sharpe=1.6, fitness=0.9, turnover=0.3, checks=FITNESS_FAIL)})
        run_queue(brain, [slow20], parallel=1)
        assert brain.started == [slow20, SHARPER]
    in_tmp(tmp_path, run)


def test_no_sharpening_when_too_similar_too_fast_or_nothing_to_remove(tmp_path):
    def run():
        brain = SlowBrain({SLOW: slow_near_miss()}, corr=0.9)
        run_queue(brain, [SLOW])
        assert brain.started == [SLOW]                             # too similar to submitted alphas
    in_tmp(tmp_path, run)
    assert not al.needs_sharpening({"expression": SLOW, "sharpe": 1.5, "fitness": 0.9, "turnover": 0.2})
    assert not al.needs_sharpening({"expression": SHARPER, "sharpe": 1.5, "fitness": 0.9, "turnover": 0.05})
    assert not al.needs_sharpening({"expression": SLOW, "sharpe": 1.1, "fitness": 0.9, "turnover": 0.05})
    assert al.needs_sharpening({"expression": SLOW, "sharpe": 1.5, "fitness": 0.98, "turnover": 0.0489})


def test_earlier_slow_near_misses_are_sharpened_at_the_start_of_a_run(tmp_path):
    def run():
        _write_operators()
        other = "ts_decay_linear(group_rank(ts_zscore(anl4_ebit_value, 120), sector), 20)"
        al.append_result(dict(_history_row(SLOW, 1.5, 0.98, 0.0489), alpha_id="s1", timestamp="2026-10-10 12:07:05"))
        al.append_result(dict(_history_row(other, 1.38, 0.89, 0.0406), alpha_id="s2", timestamp="2026-10-10 11:54:35"))
        stripped = "group_rank(ts_zscore(anl4_ebit_value, 120), sector)"
        brain = SlowBrain({SHARPER: calm(), stripped: calm(), "rank(close)": calm()},
                          existing={"s1": SLOW, "s2": other})
        gen = MagicMock(return_value=[("rank(close)", "r")])
        with patch.object(al, "MAX_ROUNDS", 1), patch.object(al, "generate_with_fallback", gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])
        assert brain.started[:2] == [SHARPER, stripped] and brain.started[2] == "rank(close)"
        assert {r["expression"]: r["round"] for r in csv_rows()}[SHARPER] == "repair"
    in_tmp(tmp_path, run)


def test_a_check_we_cant_fix_is_named_in_the_log(tmp_path):
    def run():
        unfixable = [{"name": "MATCHES_COMPETITION", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]
        brain = SlowBrain({DIVIDEND: near_miss(unfixable)})
        with captured_logs() as lines:
            run_queue(brain, [DIVIDEND])
        assert "  Strong numbers but failed MATCHES_COMPETITION - no automatic fix for that check." in lines
    in_tmp(tmp_path, run)


def test_earlier_near_misses_always_say_why_they_werent_repaired(tmp_path):
    def run():
        al.append_result(dict(_history_row(DIVIDEND, 1.35, 1.34, 0.1349), alpha_id="div", submitted="NO"))
        cases = [([{"name": "MATCHES_COMPETITION", "result": "FAIL"}], "fails MATCHES_COMPETITION - no automatic fix"),
                 ([{"name": "SELF_CORRELATION", "result": "PENDING"}], "BRAIN lists no failed check")]
        for checks, expected in cases:
            brain = SlowBrain({DIVIDEND: near_miss(checks)}, existing={"div": DIVIDEND})
            with captured_logs() as lines, patch("time.sleep"):
                assert al.repair_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", set()) == []
            assert any(expected in line for line in lines), lines
        empty = SlowBrain({}, existing={})
        empty.session.get.side_effect = lambda url, **kw: fake_response(404, {"detail": "Not found."})
        with captured_logs() as lines, patch("time.sleep"):
            al.repair_past_near_misses(empty.session, SETTINGS, "e@x.com", "pw", set())
        assert any("couldn't read BRAIN's record of it" in line for line in lines), lines
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