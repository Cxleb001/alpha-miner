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

# Most tests here are about other things than the strong-alpha rule, so in this file every alpha that clears
# BRAIN's bar counts as strong. The strong-alpha tests set the real thresholds themselves (real_strong_bar).
al.STRONG_MIN_SHARPE, al.STRONG_MIN_FITNESS = al.SUBMIT_MIN_SHARPE, al.SUBMIT_MIN_FITNESS
al.STRONG_MIN_RETURNS_TO_DRAWDOWN = 0.0
# The test world never touches the real internet or the computer's power settings: BRAIN always answers the
# "are you there?" probe and the power state is unknown. BRAIN's day never ends in the middle of a test either.
# The tests about lost connections, laptops and the end of the day set these themselves.
REAL_BRAIN_REACHABLE, REAL_ON_MAINS_POWER = al.brain_reachable, al.on_mains_power
al.brain_reachable = lambda: True
al.on_mains_power = lambda: None
al.SETTLE_BEFORE_DAY_END_MINUTES = 0
# The portfolio check reads PnL the fake BRAINs here don't serve; its own tests switch it on (portfolio_on).
al.PORTFOLIO_CHECK = False


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
    al.RESERVE = al.Reserve()  # each test starts with an empty reserve, like each run does
    al.PORTFOLIO = al.Portfolio()
    al.ACTIVE_COMPETITION = ""
    al.BEFORE_AFTER_REFUSED = False
    al.SETTLING = False
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
            fake_response(200, BEFORE_AFTER_REPLY, text="{}"),         # before/after check, recorded for later
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
             patch.object(al, "MAX_EXTRA_ROUNDS", 0), \
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
             patch.object(al, "MAX_EXTRA_ROUNDS", 0), \
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


def alpha_json(sharpe=1.6, fitness=1.2, turnover=0.3, checks=None, status="UNSUBMITTED", returns=None, drawdown=None):
    stats = {"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
             "checks": QUALIFYING_CHECKS if checks is None else checks}
    stats.update({k: v for k, v in (("returns", returns), ("drawdown", drawdown)) if v is not None})
    return {"status": status, "is": stats}


BEFORE_AFTER_REPLY = {"stats": {"before": {"sharpe": 2.1, "fitness": 1.5}, "after": {"sharpe": 2.14, "fitness": 1.52}},
                      "score": {"before": 5627, "after": 5710}}


class FakeBrain:
    """Answers by URL, the way community clients recorded BRAIN behaving."""
    def __init__(self, alphas, corr=0.3, submit_post=503, verdict=None, verdicts=None):
        self.alphas = alphas                # {alpha_id: alpha json}; simulations hand them out in order
        self.order = list(alphas)
        self.corr = corr
        self.submit_post = submit_post
        self.verdict = verdict if verdict is not None else {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]}}
        self.verdicts = verdicts or {}      # {alpha_id: verdict} - overrides `verdict` for that alpha
        self.submitted = set()
        self.before_after = []              # alpha ids BRAIN's before-and-after check was asked about
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
        if url.endswith("/before-and-after-performance"):
            self.before_after.append(url.split("/")[-2])
            return fake_response(200, BEFORE_AFTER_REPLY, text=json.dumps(BEFORE_AFTER_REPLY))
        if url.endswith("/submit"):
            self.submit_polls += 1
            if self.submit_polls == 1:  # first poll: still working
                return fake_response(200, None, headers={"Retry-After": "2.5"}, text="")
            alpha_id = url.split("/")[-2]
            verdict = self.verdicts.get(alpha_id, self.verdict)
            if not failing_names(verdict):
                self.submitted.add(alpha_id)
            return fake_response(200, verdict, text=json.dumps(verdict))
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


def test_rejected_verdict_is_recorded_and_doesnt_count_toward_today(tmp_path):
    def run():
        verdict = {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]}}
        brain = FakeBrain({"a1": alpha_json()}, verdict=verdict)
        rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "REJECTED: failed SELF_CORRELATION"
        assert al.submissions_today() == 0   # nothing went in, so the day isn't covered - the run keeps trying
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


def test_held_queue_fills_a_second_slot_only_with_a_full_reserve(tmp_path):
    def run():
        names = ("best", "second", "third", "fourth", "fifth")   # rank(<name>): five different kinds of data
        for alpha_id, fitness in zip(names, (1.6, 1.5, 1.4, 1.3, 1.2)):
            al.append_result(held_row(alpha_id, fitness))
        brain = FakeBrain({a: alpha_json() for a in names})
        brain.submit_polls = -10  # skip FakeBrain's one 'still working' reply per submit
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
            assert run_held(brain) == "YES"
            assert al.submissions_today() == 2
        assert submit_posts(brain) == ["best", "second"]          # third, fourth, fifth stay as the reserve
        assert [submitted_value(a) for a in names[2:]] == [al.HELD_LABEL] * 3
        assert al.RESERVE.kinds() == 3
    in_tmp(tmp_path, run)


def test_a_second_held_alpha_is_rechecked_after_todays_submission(tmp_path):
    def run():
        names = ("best", "second", "third", "fourth", "fifth", "sixth")
        for alpha_id, fitness in zip(names, (1.6, 1.5, 1.4, 1.3, 1.2, 1.1)):
            al.append_result(held_row(alpha_id, fitness))
        readings = iter([0.3, 0.85])               # second: fine before today's submission, too similar after
        brain = FakeBrain({a: alpha_json() for a in names})
        real_get = brain.get

        def get(url, **kw):
            if url.endswith("/second/correlations/self"):
                return fake_response(200, {"max": next(readings)})
            return real_get(url, **kw)

        brain.session.get.side_effect = get
        brain.submit_polls = -10
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
            assert run_held(brain) == "YES"
        assert submitted_value("second") == "TOO_CORRELATED (max 0.85)"
        assert submit_posts(brain) == ["best", "third"]
    in_tmp(tmp_path, run)


def test_held_queue_keeps_the_rest_back_when_the_reserve_is_thin(tmp_path):
    def run():
        for alpha_id, fitness in (("third", 1.1), ("best", 1.6), ("second", 1.4)):
            al.append_result(held_row(alpha_id, fitness))
        brain = FakeBrain({a: alpha_json() for a in ("third", "best", "second")})
        brain.submit_polls = -10
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
            assert run_held(brain) == "YES"
            assert al.submissions_today() == 1                     # a second now would leave 1 kind waiting
        assert submit_posts(brain) == ["best"]
        assert submitted_value("second") == submitted_value("third") == al.HELD_LABEL
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
        if url.endswith("/before-and-after-performance"):
            return fake_response(200, BEFORE_AFTER_REPLY, text=json.dumps(BEFORE_AFTER_REPLY))
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

        def fake_try_submit(session, alpha_id, corr=None):  # the caller now passes the similarity it just read
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


def test_pick_varied_caps_crowded_and_busy_types_then_fills_the_round(tmp_path):
    ideas = [(f"rank(implied_volatility_call_60) + {i}", "r") for i in range(3)] + \
            [("rank(sales)", "r"), ("rank(debt)", "r"), ("rank(sales * debt)", "r"),
             ("rank(est_eps)", "r"), ("rank(close)", "r"), ("rank(volume_ratio)", "r")]
    first_pass = ["rank(implied_volatility_call_60) + 0", "rank(sales)", "rank(debt)",
                  "rank(est_eps)", "rank(close)", "rank(volume_ratio)"]      # unknown fields aren't capped
    assert [e for e, _ in al.pick_varied(ideas, FOUR_TYPES, Counter({"Option": 1}), 6)] == first_pass
    with captured_logs() as lines:
        kept = [e for e, _ in al.pick_varied(ideas, FOUR_TYPES, Counter({"Option": 1}), 8)]
    assert kept == first_pass + ["rank(implied_volatility_call_60) + 1", "rank(sales * debt)"]  # each up to 2x
    assert any("(2 more let in to fill the round)" in line for line in lines), lines
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
        assert ("Write exactly this many ideas per data type: Fundamental 4, Analyst 2, Price Volume 2, Option 1, "
                "and 5 on any type. Put your best idea for each type first: only the first 2 of a type (1 for the "
                "types your alphas already use) are sure to be tested") in prompt
        assert sorted(r["expression"] for r in rows) == [
            "rank(debt)", "rank(est_eps)", "rank(sales / debt)", "rank(sales)",
            "rank(ts_delta(implied_volatility_call_60, 0))", "rank(ts_delta(implied_volatility_call_60, 1))"]
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


# ---- Strong alphas that fail a check ----

DIVIDEND = "ts_decay_linear(group_rank(ts_delta(dividend, 180), market), 20)"
CW_FAIL = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "LOW_FITNESS", "result": "PASS"},
           {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]
SUB_FAIL = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]


def near_miss(checks=None):
    return alpha_json(sharpe=1.35, fitness=1.34, turnover=0.13, checks=CW_FAIL if checks is None else checks)


def test_expression_helpers_read_expressions_carefully():
    expr = "rank(ts_regression(returns, rel_ret_comp, 60, lag=0, rettype=0)) * 1e-3"
    assert [n for _, _, n in al.data_fields_in(expr)] == ["returns", "rel_ret_comp"]
    grouped = "group_neutralize(rank(-returns), bucket(rank(cap), range='0, 1, 0.1'))"
    assert [n for _, _, n in al.data_fields_in(grouped)] == ["returns", "cap"]
    assert al.split_top_level("a, f(b, c), 'x, y'") == ["a", " f(b, c)", " 'x, y'"]
    assert al.outer_call("rank(x) + 1") is None and al.outer_call("rank(x)") == ("rank", ["x"])


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


def test_ideas_using_fields_off_your_list_are_skipped(tmp_path):
    def run():
        Path(al.FIELDS_CACHE_FILE).write_text(json.dumps({"matrix_fields": [
            {"id": name, "dataset": {"id": dataset}} for name, dataset in
            (("sales", "fundamental6"), ("assets", "fundamental6"), ("implied_volatility_call_90", "option8"),
             ("debt", "fundamental6"), ("cap", "pv1"))]}), encoding="utf-8")
        returned = [{"expression": "rank(divide(implied_volatility_call_90, assets))", "rationale": "r"},  # remembered
                    {"expression": "rank(divide(sales, assets))", "rationale": "r"},
                    {"expression": "group_rank(divide(sales, cap), sector)", "rationale": "r"},     # cap: basic data
                    {"expression": "rank(divide(debt, assets))", "rationale": "r"}]  # unlisted, but a listed dataset
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(returned)
        brain = SlowBrain({"rank(divide(sales, assets))": calm(), "group_rank(divide(sales, cap), sector)": calm(),
                           "rank(divide(debt, assets))": calm()})
        operators = [{"name": n, "usage": f"{n}(...)"} for n in ("rank", "divide", "group_rank")]
        fields = [{"name": n, "description": "d"} for n in ("sales", "assets")]
        with patch("time.sleep"), captured_logs() as lines:
            rows = al.run_round(brain.session, client, operators, fields, SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert [r["expression"] for r in rows] == ["rank(divide(sales, assets))", "group_rank(divide(sales, cap), sector)",
                                                   "rank(divide(debt, assets))"]
        assert any("Skipped 1 idea(s) using data your field list leaves out (implied_volatility_call_90)" in l
                   for l in lines), lines
        Path(al.FIELDS_CACHE_FILE).unlink()          # without BRAIN's field list nothing can be told apart
        assert al.off_list_fields("rank(divide(implied_volatility_call_90, assets))", {"sales"}) == set()
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
        assert {r["expression"]: r["round"] for r in csv_rows()}[SHARPER] == "follow-up"
    in_tmp(tmp_path, run)


def test_a_check_that_didnt_pass_is_named_in_the_log(tmp_path):
    def run():
        unfixable = [{"name": "MATCHES_COMPETITION", "result": "FAIL"}, {"name": "SELF_CORRELATION", "result": "PENDING"}]
        brain = SlowBrain({DIVIDEND: near_miss(unfixable)})
        with captured_logs() as lines:
            run_queue(brain, [DIVIDEND])
        assert "  Strong numbers but failed MATCHES_COMPETITION." in lines
        assert brain.started == [DIVIDEND]                         # nothing else is tried on it
    in_tmp(tmp_path, run)


# ---- Never miss a day: extra rounds, the reserve, the backup run ----

def write_dataset_cache(datasets):
    Path(al.FIELDS_CACHE_FILE).write_text(json.dumps({"matrix_fields": [
        {"id": name, "dataset": {"id": ds, "name": ds.upper()}, "category": {"id": "x", "name": "X"}}
        for name, ds in datasets.items()]}), encoding="utf-8")


def run_main_rounds(brain, argv=(), **settings):
    """main() with Gemini replaced: round n proposes rank(idea<n>). Returns how many rounds asked for ideas."""
    _write_operators()
    calls = {"n": 0}

    def gen(*a, **kw):
        calls["n"] += 1
        return [(f"rank(idea{calls['n']})", "r")]

    with contextlib.ExitStack() as stack:
        for name, value in dict({"IDEAS_PER_ROUND": 1}, **settings).items():
            stack.enter_context(patch.object(al, name, value))
        stack.enter_context(patch.object(al, "generate_with_fallback", side_effect=gen))
        stack.enter_context(patch.object(al, "get_authenticated_session", return_value=brain.session))
        stack.enter_context(patch("google.genai.Client"))
        stack.enter_context(patch("time.sleep"))
        al.main(list(argv))
    return calls["n"]


def test_rounds_go_on_past_max_rounds_until_todays_submission_lands(tmp_path):
    def run():
        brain = FakeBrain({"r1": calm(), "r2": calm(), "r3": alpha_json(), "r4": calm()})
        with captured_logs() as lines:
            asked = run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=5)
        assert asked == 3                                          # stopped as soon as r3 went in
        assert submitted_value("r3") == "YES" and al.submissions_today() == 1
        assert any("carrying on past MAX_ROUNDS" in line for line in lines), lines
        assert any("(2 of them extra, to get today's submission in)" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_extra_rounds_stop_at_their_cap(tmp_path):
    def run():
        brain = FakeBrain({f"r{i}": calm() for i in range(1, 9)})
        with captured_logs() as lines:
            asked = run_main_rounds(brain, MAX_ROUNDS=2, MAX_EXTRA_ROUNDS=3)
        assert asked == 5 and al.submissions_today() == 0
        assert any("Still nothing submitted for today after 3 extra round(s)" in line for line in lines), lines
        assert any("Nothing was submitted for BRAIN day" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_no_extra_rounds_when_today_is_already_in(tmp_path):
    def run():
        al.record_submission()
        brain = FakeBrain({f"r{i}": calm() for i in range(1, 9)})
        assert run_main_rounds(brain, MAX_ROUNDS=2, MAX_EXTRA_ROUNDS=3) == 2
    in_tmp(tmp_path, run)


def test_an_extra_round_first_gives_held_alphas_another_try(tmp_path):
    def run():
        al.append_result(held_row("held", 1.3))
        brain = FakeBrain({"r1": calm(), "held": alpha_json()}, submit_post=500)  # submits refused at first
        real_post = brain.post

        def post(url, **kw):
            if url.endswith("/simulations"):
                brain.submit_post = 503                            # ...and taken again later in the run
            return real_post(url, **kw)

        brain.session.post.side_effect = post
        assert run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=5) == 1   # no extra round needed
        assert submit_posts(brain) == ["held", "held"] and submitted_value("held") == "YES"
    in_tmp(tmp_path, run)


def test_alpha_kind_reads_datasets_and_ignores_basic_price_data(tmp_path):
    def run():
        write_dataset_cache({"implied_volatility_put_90": "option8", "implied_volatility_put_180": "option8",
                             "unsystematic_risk_last_90_days": "model16", "close": "pv1", "adv60": "pv1"})
        put90 = "ts_decay_linear(group_rank(ts_delta(implied_volatility_put_90, 30), industry), 10)"
        assert al.alpha_kind(put90) == {"option8"}
        assert al.alpha_kind("rank(implied_volatility_put_180 / close)") == {"option8"}   # close is an ingredient
        assert al.alpha_kind("rank(adv60) - rank(close)") == {al.PRICE_VOLUME}            # only basic data
        assert al.alpha_kind("rank(unsystematic_risk_last_90_days) + rank(implied_volatility_put_90)") == \
            {"model16", "option8"}
        assert al.alpha_kind("rank(snt_buzz_ret_30)") == {"snt_buzz_ret"}                 # not in the cache
    in_tmp(tmp_path, run)


def test_reserve_counts_kinds_not_alphas(tmp_path):
    def run():
        reserve = al.Reserve()
        for alpha_id, expression in (("u90", "rank(unsystematic_risk_last_90_days)"),
                                     ("u360", "ts_zscore(unsystematic_risk_last_360_days, 90)"),
                                     ("iv", "rank(implied_volatility_call_90)"),
                                     ("ebit", "ts_delta(anl4_ebit_value, 180)")):
            reserve.keep(alpha_id, expression)
        assert reserve.kinds() == 3                                # the two unsystematic-risk ones count once
        assert reserve.kinds(besides="rank(implied_volatility_call_60)") == 2
        reserve.note_landed("ebit", "ts_delta(anl4_ebit_value, 180)")
        assert reserve.kinds() == 2
        reserve.note_landed("x", "rank(unsystematic_risk_last_30_days)")   # today's submission shares their data
        assert reserve.kinds() == 1 and len(reserve.usable()) == 1
        assert al.distinct_kinds([{"a"}, {"b"}, {"a", "c"}, {"c", "d"}]) == 2
    in_tmp(tmp_path, run)


def test_todays_first_submission_always_goes_in_a_second_needs_the_reserve(tmp_path):
    def run():
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
            assert run_one_idea(FakeBrain({"a1": alpha_json()}), "rank(first_field)")[0]["submitted"] == "YES"
            with captured_logs() as lines:
                second = run_one_idea(FakeBrain({"a2": alpha_json()}), "rank(second_field)")
            assert second[0]["submitted"] == al.HELD_LABEL        # would leave nothing for the coming days
            assert any("holding it for a coming day" in line and "only 0 other kind(s)" in line
                       for line in lines), lines
            assert list(al.RESERVE.waiting) == ["a2"]              # held, and it counts toward the reserve
            for name in ("alpha_one", "alpha_two"):                # with a2, that's 3 other kinds waiting
                al.RESERVE.keep(name, f"rank({name}_field)")
            third = run_one_idea(FakeBrain({"a3": alpha_json()}), "rank(third_field)")
            assert third[0]["submitted"] == "YES" and al.submissions_today() == 2
            fourth = run_one_idea(FakeBrain({"a4": alpha_json()}), "rank(fourth_field)")
            assert fourth[0]["submitted"] == al.HELD_LABEL        # the daily limit still holds
    in_tmp(tmp_path, run)


def test_reserve_target_zero_turns_the_rule_off(tmp_path):
    def run():
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2), patch.object(al, "RESERVE_TARGET", 0):
            al.record_submission()
            assert al.may_submit("rank(anything)") is True
            al.record_submission()
            assert al.may_submit("rank(anything)") is False
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 0):
            assert al.may_submit("rank(anything)") is False        # submissions switched off
    in_tmp(tmp_path, run)


def test_after_a_rejection_the_next_good_alpha_still_goes_in(tmp_path):
    def run():
        al.append_result(held_row("first", 1.5))
        al.append_result(held_row("second", 1.2))
        reject = {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]}}
        brain = FakeBrain({"first": alpha_json(), "second": alpha_json()}, verdicts={"first": reject})
        brain.submit_polls = -10
        assert run_held(brain) == "YES"
        assert submitted_value("first") == "REJECTED: failed SELF_CORRELATION"
        assert submitted_value("second") == "YES"
        assert submit_posts(brain) == ["first", "second"] and al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_a_submit_that_doesnt_go_through_is_held_not_lost(tmp_path):
    def run():
        rows = run_one_idea(FakeBrain({"a1": alpha_json()}, submit_post=500))
        assert rows[0]["submitted"] == al.HELD_RETRY_LABEL
        assert [r["alpha_id"] for r in al.held_candidates()] == ["a1"] and al.submissions_today() == 0
    in_tmp(tmp_path, run)


def test_backup_run_stops_when_today_is_in_and_the_reserve_is_full(tmp_path):
    def run():
        al.record_submission()
        for name, fitness in (("aaa", 1.3), ("bbb", 1.2), ("ccc", 1.1)):
            al.append_result(held_row(name, fitness))
        brain = FakeBrain({"aaa": alpha_json(), "bbb": alpha_json(), "ccc": alpha_json()})
        with captured_logs() as lines:
            asked = run_main_rounds(brain, ["--backup"], MAX_ROUNDS=2)
        assert asked == 0 and submit_posts(brain) == []
        assert any("Backup run: today's submission is in and 3 kinds" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_backup_run_mines_when_the_reserve_is_thin(tmp_path):
    def run():
        al.record_submission()
        al.append_result(held_row("aaa", 1.3))
        brain = FakeBrain({"r1": calm(), "r2": calm(), "aaa": alpha_json()})
        with captured_logs() as lines:
            asked = run_main_rounds(brain, ["--backup"], MAX_ROUNDS=2)
        assert asked == 2
        assert any("only 1 kind(s) of good alpha are waiting (aiming for 3) - mining" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_backup_run_is_a_normal_run_when_nothing_is_in_today(tmp_path):
    def run():
        brain = FakeBrain({"r1": calm(), "r2": alpha_json()})
        assert run_main_rounds(brain, ["--backup"], MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=3) == 2
        assert submitted_value("r2") == "YES"
    in_tmp(tmp_path, run)


# ---- BRAIN's before-and-after check: recorded, not acted on yet ----

def test_before_after_check_is_recorded_before_submitting(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()})
        with captured_logs() as lines:
            rows = run_one_idea(brain)
        assert rows[0]["submitted"] == "YES" and brain.before_after == ["a1"]
        order = [c.args[0] for c in brain.session.mock_calls if c[0] in ("get", "post")]
        assert order.index(al.BEFORE_AFTER_URL.format(alpha_id="a1")) < \
            order.index("https://api.worldquantbrain.com/alphas/a1/submit")
        record = json.loads(Path(al.BEFORE_AFTER_FILE).read_text(encoding="utf-8").splitlines()[0])
        assert (record["alpha_id"], record["when"], record["status"]) == ("a1", "before submitting", 200)
        assert record["reply"] == BEFORE_AFTER_REPLY
        assert any("score 5627 -> 5710 (+83); sharpe 2.1 -> 2.14; fitness 1.5 -> 1.52" in line
                   for line in lines), lines
    in_tmp(tmp_path, run)


def test_held_alpha_gets_a_before_after_check_too(tmp_path):
    def run():
        al.record_submission()
        brain = FakeBrain({"a1": alpha_json()})
        assert run_one_idea(brain)[0]["submitted"] == al.HELD_LABEL and brain.before_after == ["a1"]
        record = json.loads(Path(al.BEFORE_AFTER_FILE).read_text(encoding="utf-8").splitlines()[0])
        assert record["when"] == "held for a coming day"
    in_tmp(tmp_path, run)


def test_before_after_waits_while_brain_works_it_out():
    session = MagicMock()
    session.get.side_effect = [fake_response(200, None, headers={"Retry-After": "3"}, text=""),
                               fake_response(200, {"stats": {}}, text="{}")]
    with patch("time.sleep") as sleep:
        assert al.before_after(session, "a1") == (200, {"stats": {}})
    assert [c.args[0] for c in sleep.call_args_list] == [3.0]
    assert session.get.call_args_list[0].args[0].endswith("/users/self/alphas/a1/before-and-after-performance")


def test_before_after_problems_never_stop_a_submission(tmp_path):
    def run():
        assert al.before_after_summary({"something": 1, "else": {}}) == "reply saved - its fields: else, something"
        assert al.before_after_summary([1, 2]) == "reply saved (list)"
        for trouble in (_requests.ConnectionError("dropped"), KeyError("odd reply")):
            brain = FakeBrain({"a1": alpha_json()})
            real_get = brain.get

            def get(url, trouble=trouble, real_get=real_get, **kw):
                if url.endswith("/before-and-after-performance"):
                    raise trouble
                return real_get(url, **kw)

            brain.session.get.side_effect = get
            al.RESERVE = al.Reserve()
            Path(al.SUBMISSION_LOG_FILE).unlink(missing_ok=True)
            assert run_one_idea(brain)[0]["submitted"] == "YES"
        assert not Path(al.BEFORE_AFTER_FILE).exists()
    in_tmp(tmp_path, run)


# ---- Full rounds: the variety limits no longer leave rounds half-empty ----

SIX_TYPES = {"news_a": "News", "news_b": "News", "news_c": "News", "scl12_buzz": "Social Media",
             "implied_volatility_call_30": "Option", "unsystematic_risk_last_90_days": "Model",
             "anl4_ebit_value": "Analyst", "close": "Price Volume", "returns": "Price Volume", "adv60": "Price Volume"}
CROWDED_THREE = Counter({"Option": 2, "Model": 2, "Analyst": 1})


def test_basic_price_data_only_counts_when_nothing_else_is_used():
    assert al.expression_types("ts_corr(scl12_buzz, returns, 30)", SIX_TYPES) == {"Social Media"}
    assert al.expression_types("rank(anl4_ebit_value / adv60)", SIX_TYPES) == {"Analyst"}  # any price/volume field
    assert al.expression_types("ts_corr(close, adv60, 20)", SIX_TYPES) == {"Price Volume"}
    assert al.expression_types("rank(mystery_field)", SIX_TYPES) == set()


def test_type_quota_spreads_the_ideas_over_every_type():
    quota = al.type_quota(SIX_TYPES, CROWDED_THREE, 14)
    assert quota == {"News": 5, "Price Volume": 4, "Social Media": 2, "Analyst": 1, "Model": 1, "Option": 1}
    assert list(quota) == ["News", "Price Volume", "Social Media", "Analyst", "Model", "Option"]
    assert sum(al.type_quota(SIX_TYPES, Counter(), 14).values()) == 14
    assert al.type_quota({"scl12_buzz": "Social Media", "iv": "Option"}, Counter({"Option": 1}), 14) == \
        {"Social Media": 2, "Option": 1}                                    # never more than the fields allow


def test_the_live_log_round_isnt_left_half_empty(tmp_path):
    # 11 Oct: Gemini put all 14 ideas on Social Media, so only 2 were tested per round.
    ideas = [(f"ts_decay_linear(group_rank(ts_delta(scl12_buzz, {d}), sector), 10)", "r") for d in range(14)]
    with captured_logs() as lines:
        kept = al.pick_varied(ideas, SIX_TYPES, CROWDED_THREE, 8)
    assert len(kept) == 4                                   # 2 as before, plus 2 let in - still varied
    assert any("skipped 10 idea(s)" in line and "(2 more let in to fill the round)" in line for line in lines), lines
    overused = {al.skeleton(ideas[0][0])}                   # filling up never breaks the shape limit
    assert len(al.pick_varied(ideas, SIX_TYPES, CROWDED_THREE, 8, overused)) == al.MAX_IDEAS_PER_CROWDED_SHAPE == 2
    spread = [f"rank(news_a + {i})" for i in range(2)] + [f"rank(news_b * {i})" for i in range(2)] + \
             ["rank(scl12_buzz)", "ts_rank(scl12_buzz, 20)", "rank(implied_volatility_call_30)",
              "rank(unsystematic_risk_last_90_days)", "rank(anl4_ebit_value)", "ts_corr(close, adv60, 20)"]
    assert len(al.pick_varied([(e, "r") for e in spread], SIX_TYPES, CROWDED_THREE, 8)) == 8


def test_a_refused_before_after_check_isnt_asked_again_this_run(tmp_path):
    def run():
        session = MagicMock()
        session.get.return_value = fake_response(403, {"detail": "You do not have permission to perform this action."})
        with captured_logs() as lines:
            assert al.record_before_after(session, "a1", "before submitting") is None
            assert al.record_before_after(session, "a2", "held for a coming day") is None
        assert session.get.call_count == 1
        assert any("isn't open to your account (You do not have permission" in line for line in lines), lines
        brain = FakeBrain({"r1": alpha_json()})
        run_main_rounds(brain, MAX_ROUNDS=1)                 # a new run asks again
        assert brain.before_after == ["r1"]
    in_tmp(tmp_path, run)


# ---- Strong alphas first, without losing a day ----

@contextlib.contextmanager
def real_strong_bar():
    with patch.object(al, "STRONG_MIN_SHARPE", 1.6), patch.object(al, "STRONG_MIN_FITNESS", 1.3), \
         patch.object(al, "STRONG_MIN_RETURNS_TO_DRAWDOWN", 1.0):
        yield


def weak_json(**kw):     # clears BRAIN's bar, but isn't strong
    return alpha_json(**dict(dict(sharpe=1.4, fitness=1.1, returns=0.06, drawdown=0.03), **kw))


def strong_json(**kw):
    return alpha_json(**dict(dict(sharpe=1.8, fitness=1.5, returns=0.0913, drawdown=0.0405), **kw))


def test_strong_needs_sharpe_fitness_and_returns_above_drawdown():
    with real_strong_bar():
        assert al.is_strong(1.73, 1.48, 0.0913, 0.0405)           # the 11 Oct alpha in your screenshot
        assert not al.is_strong(1.73, 1.48, 0.03, 0.0405)         # lost more in its worst stretch than it made
        assert not al.is_strong(1.55, 1.48, 0.0913, 0.0405)       # Sharpe under 1.6
        assert not al.is_strong(1.73, 1.25, 0.0913, 0.0405)       # fitness under 1.3
        assert not al.is_strong(1.73, 1.48, None, 0.0405)         # BRAIN didn't say: never strong
        assert al.is_strong("1.73", "1.48", "0.05", "-0.04")      # as read back from the CSV; negative drawdown
        assert al.strong_note() == "Sharpe 1.6+, fitness 1.3+, returns above drawdown"
        with patch.object(al, "STRONG_MIN_RETURNS_TO_DRAWDOWN", 0):
            assert al.is_strong(1.73, 1.48)                        # 0 turns the returns part off


def test_returns_and_drawdown_are_saved_and_shown_to_the_generator(tmp_path):
    def run():
        rows = run_one_idea(FakeBrain({"a1": alpha_json(returns=0.0913, drawdown=0.0405)}))
        saved = csv_rows()[0]
        assert (saved["returns"], saved["drawdown"]) == ("0.0913", "0.0405")
        assert "returns=0.0913 drawdown=0.0405" in al.build_feedback(rows)
        al.append_result(dict(_history_row("rank(bumpy)", 1.5, 1.2, 0.1), returns="0.02", drawdown="0.05"))
        assert "RETURNS BELOW DRAWDOWN" in al.build_feedback()
    in_tmp(tmp_path, run)


def test_an_older_results_file_gets_the_new_columns_once(tmp_path):
    def run():
        newer = ("returns", "drawdown", "sub_universe_sharpe", "sub_universe_limit")   # an older file has none
        old_header = [c for c in al.RESULTS_HEADER if c not in newer]
        with open(al.RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_header)
            writer.writeheader()
            writer.writerow(dict({k: "" for k in old_header}, expression="rank(old)", alpha_id="old1"))
        with patch.object(al.os, "replace", side_effect=PermissionError("open in Excel")), patch("time.sleep"):
            al.upgrade_results_file()                              # can't be swapped in: left as it was
        al.append_result(dict(_history_row("rank(new)", 1.3, 1.1, 0.1), returns="0.05", drawdown="0.02"))
        assert al.results_columns() == old_header                  # the new row still lines up
        with open(al.RESULTS_FILE, encoding="utf-8", newline="") as f:
            assert {len(r) for r in csv.reader(f)} == {len(old_header)}
        assert [r["expression"] for r in csv_rows()] == ["rank(old)", "rank(new)"]
        assert not Path(al.RESULTS_FILE + ".tmp").exists()
        al.upgrade_results_file()
        assert al.results_columns() == al.RESULTS_HEADER
        assert [(r["expression"], r["returns"]) for r in csv_rows()] == [("rank(old)", ""), ("rank(new)", "")]
        al.append_result(dict(_history_row("rank(newer)", 1.3, 1.1, 0.1), returns="0.05", drawdown="0.02"))
        assert csv_rows()[-1]["returns"] == "0.05"
    in_tmp(tmp_path, run)


def test_a_strong_alpha_goes_straight_in_a_weak_one_waits_and_extras_must_be_strong(tmp_path):
    def run():
        with real_strong_bar():
            weak = run_one_idea(FakeBrain({"weak": weak_json()}), "rank(weak_field)")
            assert weak[0]["submitted"] == al.HELD_FOR_STRONGER_LABEL and al.submissions_today() == 0
            strong = run_one_idea(FakeBrain({"strong": strong_json()}), "rank(strong_field)")
            assert strong[0]["submitted"] == "YES" and al.submissions_today() == 1
            with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2):
                for name in ("alpha_one", "alpha_two", "alpha_three"):
                    al.RESERVE.keep(name, f"rank({name}_field)")     # a full reserve
                with captured_logs() as lines:
                    bumpy = run_one_idea(FakeBrain({"bumpy": strong_json(returns=0.02)}), "rank(bumpy_field)")
                assert bumpy[0]["submitted"] == al.HELD_LABEL
                assert any("only strong alphas" in line for line in lines), lines
                extra = run_one_idea(FakeBrain({"extra": strong_json(sharpe=1.9)}), "rank(extra_field)")
                assert extra[0]["submitted"] == "YES" and al.submissions_today() == 2
    in_tmp(tmp_path, run)


def test_with_no_strong_alpha_the_best_held_one_goes_in_after_the_usual_rounds(tmp_path):
    def run():
        brain = FakeBrain({"r1": weak_json(), "r2": weak_json(fitness=1.2), "r3": calm()})
        with real_strong_bar(), captured_logs() as lines:
            asked = run_main_rounds(brain, MAX_ROUNDS=2, MAX_EXTRA_ROUNDS=5)
        assert asked == 2                                          # settled - no extra rounds needed
        assert submitted_value("r2") == "YES"                      # the better of the two
        assert submitted_value("r1") == al.HELD_FOR_STRONGER_LABEL
        assert any("No strong alpha has gone in today" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_a_strong_alpha_later_in_the_run_beats_an_earlier_weak_one(tmp_path):
    def run():
        brain = FakeBrain({"r1": weak_json(), "r2": strong_json()})
        with real_strong_bar():
            run_main_rounds(brain, MAX_ROUNDS=2)
        assert submit_posts(brain) == ["r2"]
        assert submitted_value("r1") == al.HELD_FOR_STRONGER_LABEL     # kept for a coming day
    in_tmp(tmp_path, run)


def test_held_queue_puts_a_strong_alpha_ahead_of_a_fitter_bumpy_one(tmp_path):
    def run():
        al.append_result(held_row("bumpy", 1.6))
        al.append_result(held_row("steady", 1.35))
        brain = FakeBrain({"bumpy": strong_json(fitness=1.6, returns=0.02, drawdown=0.05),
                           "steady": strong_json(fitness=1.35)})
        al.SETTLING = True                                         # either may go in: BRAIN's latest numbers decide
        with real_strong_bar():
            assert run_held(brain) == "YES"
        assert submit_posts(brain) == ["steady"] and submitted_value("bumpy") == al.HELD_LABEL
    in_tmp(tmp_path, run)


def test_held_candidates_list_strong_ones_first(tmp_path):
    def run():
        al.append_result(held_row("fitter", 1.6))                                       # no returns saved
        al.append_result(dict(held_row("strong", 1.35, sharpe=1.7), returns="0.08", drawdown="0.03"))
        with real_strong_bar():
            assert [r["alpha_id"] for r in al.held_candidates()] == ["strong", "fitter"]
        assert [r["alpha_id"] for r in al.held_candidates()] == ["fitter", "strong"]   # both strong: by fitness
    in_tmp(tmp_path, run)


def test_weak_held_alphas_wait_until_the_run_settles(tmp_path):
    def run():
        al.append_result(held_row("weak", 1.2))
        brain = FakeBrain({"weak": weak_json(fitness=1.2)})
        with real_strong_bar(), patch("time.sleep"):
            with captured_logs() as lines:
                assert run_held(brain) is None
            assert any("None of the held alphas is strong" in line for line in lines), lines
            assert submit_posts(brain) == []
            al.settle_for_today(brain.session)
        assert submit_posts(brain) == ["weak"] and submitted_value("weak") == "YES"
    in_tmp(tmp_path, run)


def test_try_mode_settles_at_the_end(tmp_path):
    def run():
        brain = FakeBrain({"t1": weak_json()})
        with real_strong_bar():
            run_main_try(brain, ["--try", "rank(my_idea)"])
        assert submitted_value("t1") == "YES"
    in_tmp(tmp_path, run)


def test_a_run_that_stops_early_still_settles(tmp_path):
    def run():
        _write_operators()
        al.append_result(held_row("weak", 1.2))
        brain = FakeBrain({"weak": weak_json(fitness=1.2)})
        with real_strong_bar(), patch.object(al, "generate_with_fallback", side_effect=busy()), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])                                            # Gemini down all run
        assert submitted_value("weak") == "YES"
    in_tmp(tmp_path, run)


def test_the_generator_is_told_the_strong_targets():
    client = MagicMock()
    client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
    with real_strong_bar():
        al.generate_ideas(client, "m", [], [], SETTINGS, "feedback", set(), 3)
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "aim for Sharpe 1.6+, fitness 1.3+, returns above drawdown" in prompt


# ---- Near-copies: stricter than BRAIN for every submission but the end-of-day pick ----

def test_a_strong_alpha_somewhat_similar_to_yours_is_held_not_submitted(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()}, corr=0.62)               # under BRAIN's 0.7, not under 0.5
        with captured_logs() as lines:
            rows = run_one_idea(brain)
        assert rows[0]["submitted"] == al.HELD_SIMILAR_LABEL
        assert al.submissions_today() == 0 and submit_posts(brain) == []
        assert [r["alpha_id"] for r in al.held_candidates()] == ["a1"]   # queued for the end-of-day pick
        assert al.RESERVE.kinds() == 1 and brain.before_after == ["a1"]  # counted, and checked like any held one
        assert any("self-correlation is 0.62" in l and "under 0.5 go in before the end-of-day pick" in l
                   for l in lines), lines
    in_tmp(tmp_path, run)


def test_exactly_at_the_daytime_limit_is_held_just_under_goes_in(tmp_path):
    def run():
        assert run_one_idea(FakeBrain({"at": alpha_json()}, corr=0.5))[0]["submitted"] == al.HELD_SIMILAR_LABEL
        assert run_one_idea(FakeBrain({"under": alpha_json()}, corr=0.49))[0]["submitted"] == "YES"
    in_tmp(tmp_path, run)


def test_a_daytime_limit_of_0_7_turns_the_rule_off(tmp_path):
    def run():
        with patch.object(al, "DAYTIME_MAX_SELF_CORRELATION", 0.7):
            unread = run_one_idea(FakeBrain({"u1": alpha_json()}, corr={"u1": None}))   # as before: in it goes
            assert unread[0]["submitted"] == "YES"
            Path(al.SUBMISSION_LOG_FILE).unlink()                                     # a fresh day for the next one
            assert run_one_idea(FakeBrain({"a1": alpha_json()}, corr=0.62))[0]["submitted"] == "YES"
            assert al.daytime_note() == ""
    in_tmp(tmp_path, run)


def test_during_the_day_held_alphas_go_by_brains_latest_numbers(tmp_path):
    def run():
        al.append_result(held_row("was_best", 1.6))                      # fittest when it was found...
        al.append_result(held_row("now_best", 1.35))
        brain = FakeBrain({"was_best": alpha_json(fitness=1.3), "now_best": alpha_json(fitness=1.7)})
        assert run_held(brain) == "YES"                                  # ...but BRAIN's latest numbers decide
        assert submit_posts(brain) == ["now_best"]
    in_tmp(tmp_path, run)


def test_once_the_run_settles_a_new_alpha_up_to_0_7_goes_straight_in(tmp_path):
    def run():
        al.SETTLING = True                                               # the next good one is the end-of-day pick
        rows = run_one_idea(FakeBrain({"a1": alpha_json()}, corr=0.66))
        assert rows[0]["submitted"] == "YES" and al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_an_extra_submission_must_be_under_the_daytime_limit_even_after_settling(tmp_path):
    def run():
        al.record_submission()                                           # today's first is in
        al.SETTLING = True                                               # only the day's first is the end-of-day pick
        brain = FakeBrain({"a1": alpha_json()}, corr=0.55)
        with patch.object(al, "MAX_SUBMISSIONS_PER_DAY", 2), patch.object(al, "RESERVE_TARGET", 0):
            rows = run_one_idea(brain)
        assert rows[0]["submitted"] == al.HELD_SIMILAR_LABEL and submit_posts(brain) == []
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_unreadable_similarity_waits_for_the_end_of_day_pick(tmp_path):
    def run():
        brain = FakeBrain({"a1": alpha_json()}, corr={"a1": None})       # BRAIN gives no number
        assert run_one_idea(brain)[0]["submitted"] == al.HELD_SIMILAR_LABEL
        assert submit_posts(brain) == []                                 # can't be shown to be under 0.5
        with patch("time.sleep"):
            al.settle_for_today(brain.session)                           # still no number: in it goes, as before
        assert submit_posts(brain) == ["a1"] and submitted_value("a1") == "YES"
    in_tmp(tmp_path, run)


def test_held_alphas_need_the_daytime_limit_until_the_end_of_day_pick(tmp_path):
    def run():
        al.append_result(held_row("similar", 1.5, sharpe=1.8, submitted=al.HELD_SIMILAR_LABEL))
        brain = FakeBrain({"similar": alpha_json(sharpe=1.8, fitness=1.5)}, corr=0.61)
        with captured_logs() as lines:
            assert run_held(brain) is None                               # strong, but not under 0.5: waits
        assert submit_posts(brain) == [] and submitted_value("similar") == al.HELD_SIMILAR_LABEL
        assert any("self-correlation is 0.61" in l and "stays queued" in l for l in lines), lines
        al.SETTLING = True
        assert run_held(brain) == "YES"                                  # the end-of-day pick allows up to 0.7
        assert submit_posts(brain) == ["similar"]
    in_tmp(tmp_path, run)


def test_the_end_of_day_pick_goes_least_similar_first(tmp_path):
    def run():
        held = {"strongest": (1.6, 0.64), "unique": (1.1, 0.18), "unknown": (1.4, None), "middling": (1.2, 0.4)}
        for alpha_id, (fitness, _) in held.items():
            al.append_result(held_row(alpha_id, fitness))
        brain = FakeBrain({a: alpha_json(fitness=f) for a, (f, _) in held.items()},
                          corr={a: c for a, (_, c) in held.items()})
        al.SETTLING = True
        assert run_held(brain) == "YES"
        assert submit_posts(brain) == ["unique"]                         # least similar, though the least fit
        assert {a: submitted_value(a) for a in ("strongest", "unknown", "middling")} == \
               dict.fromkeys(("strongest", "unknown", "middling"), al.HELD_LABEL)
    in_tmp(tmp_path, run)


def test_end_of_day_order_least_similar_then_stronger_unreadable_last():
    items = [("x", 0.64, {"sharpe": "1.9", "fitness": "1.6"}), ("y", 0.18, {"sharpe": "1.3", "fitness": "1.05"}),
             ("z", None, {"sharpe": "2.0", "fitness": "1.8"}), ("w", 0.18, {"sharpe": "1.4", "fitness": "1.2"})]
    assert [i[0] for i in sorted(items, key=al.end_of_day_order)] == ["w", "y", "x", "z"]


def test_a_day_with_only_somewhat_similar_alphas_still_gets_its_submission(tmp_path):
    def run():
        brain = FakeBrain({"r1": alpha_json(), "r2": calm()}, corr={"r1": 0.6})
        with captured_logs() as lines:
            asked = run_main_rounds(brain, MAX_ROUNDS=2, MAX_EXTRA_ROUNDS=5)
        assert asked == 2                                                # no extra rounds needed
        assert submitted_value("r1") == "YES" and al.submissions_today() == 1
        assert any("with self-correlation under 0.5 go in first" in l for l in lines), lines   # said at the start
    in_tmp(tmp_path, run)


# ---- Portfolio check: an alpha goes in only if it raises your submitted alphas' combined Sharpe ----

import math as _math
import random as _random
from datetime import date as _d, timedelta as _td


def pnl_series(seed, sharpe, on_factor=0.0, days=1500):
    """Daily PnL with roughly this annual Sharpe; on_factor ties it to one shared market factor (so alphas built that
    way move together, like near-copies do)."""
    rng, factor = _random.Random(seed), _random.Random(99)
    shared = [factor.gauss(0, 1) for _ in range(days)]
    mean = sharpe / _math.sqrt(252)
    rest = _math.sqrt(1 - on_factor ** 2)
    return [mean + on_factor * shared[t] + rest * rng.gauss(0, 1) for t in range(days)]


def pnl_recordset(daily):
    """BRAIN's PnL recordset for these daily values: a running total, one row per day."""
    total, records = 0.0, [[_d(2019, 1, 1).isoformat(), 0.0]]
    for t, value in enumerate(daily, start=1):
        total += value
        records.append([(_d(2019, 1, 1) + _td(t)).isoformat(), total])
    return {"schema": {"name": "pnl", "title": "PnL", "properties": [{"name": "date", "title": "Date", "type": "date"},
                                                                     {"name": "pnl", "title": "PnL", "type": "amount"}]},
            "records": records}


# Three submitted alphas built on the same market factor; COPY is one more like them, only weaker; FRESH is unrelated;
# MEH is unrelated too, but weaker than FRESH - it still raises the combined Sharpe, just by less.
MINE = {f"mine{i}": pnl_series(i, 1.5, on_factor=0.8) for i in (1, 2, 3)}
COPY, FRESH, MEH = pnl_series(11, 0.8, on_factor=0.95), pnl_series(12, 2.0), pnl_series(13, 1.3)


class PortfolioBrain(FakeBrain):
    """FakeBrain that also lists your submitted alphas, serves each alpha's PnL and lists your competitions."""
    def __init__(self, alphas, pnl, listed=(), competitions=None, pnl_status=200, **kw):
        super().__init__(alphas, **kw)
        self.pnl, self.listed, self.competitions, self.pnl_status = pnl, list(listed), competitions, pnl_status
        self.pnl_asked, self.list_params = [], []

    def get(self, url, **kw):
        if url == al.USER_ALPHAS_URL:
            params = kw.get("params") or {}
            self.list_params.append(dict(params))
            mine = self.listed + sorted(self.submitted - set(self.listed))
            everyone = [{"id": a, "status": "ACTIVE"} for a in mine] + [{"id": "draft", "status": "UNSUBMITTED"}]
            start = params.get("offset", 0)
            return fake_response(200, {"count": len(everyone), "results": everyone[start:start + params.get("limit")]})
        if url.endswith("/recordsets/pnl"):
            alpha_id = url.split("/")[-3]
            self.pnl_asked.append(alpha_id)
            if self.pnl_status != 200 or alpha_id not in self.pnl:
                return fake_response(self.pnl_status if self.pnl_status != 200 else 404, {"detail": "no"}, text="no")
            return fake_response(200, pnl_recordset(self.pnl[alpha_id]), text="pnl")
        if url == al.COMPETITIONS_URL:
            if self.competitions is None:
                return fake_response(404, {"detail": "no"}, text="no")
            return fake_response(200, {"count": len(self.competitions), "next": None, "results": self.competitions})
        return super().get(url, **kw)


@contextlib.contextmanager
def portfolio_on(**settings):
    with patch.object(al, "PORTFOLIO_CHECK", True), contextlib.ExitStack() as stack:
        for name, value in settings.items():
            stack.enter_context(patch.object(al, name, value))
        yield


def mine_and(candidates, corr=0.3, **kw):
    """A PortfolioBrain where MINE are your submitted alphas and `candidates` ({id: daily PnL}) are new ones."""
    return PortfolioBrain({a: alpha_json() for a in candidates}, dict(MINE, **candidates), listed=MINE, corr=corr, **kw)


def test_daily_pnl_turns_brains_running_total_into_daily_changes():
    data = {"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
            "records": [["2019-01-03", 150.0], ["2019-01-02", 100.0], ["2019-01-04", None], ["2019-01-07", 120.0]]}
    assert al.daily_pnl(data) == {"2019-01-03": 50.0, "2019-01-07": -30.0}    # sorted by date; a blank day skipped
    flipped = {"schema": {"properties": {"pnl": {}, "date": {}}}, "records": [[100.0, "2019-01-02"], [90.0, "2019-01-03"]]}
    assert al.daily_pnl(flipped) == {"2019-01-03": -10.0}                     # columns found by name
    assert al.daily_pnl({}) == {} and al.daily_pnl("junk") == {}


def test_fetching_a_pnl_waits_while_brain_prepares_it_and_gives_up_cleanly():
    session = MagicMock()
    session.get.side_effect = [fake_response(200, None, headers={"Retry-After": "1.5"}, text=""),
                               fake_response(429, {"detail": "slow down"}, headers={"Retry-After": "2.0"}, text="x"),
                               fake_response(200, pnl_recordset([5.0, -2.0]), text="pnl")]
    with patch("time.sleep") as sleep:
        assert al.fetch_daily_pnl(session, "a1") == {"2019-01-02": 5.0, "2019-01-03": -2.0}
    assert [c.args[0] for c in sleep.call_args_list] == [1.5, 2.0]
    assert session.get.call_args.args[0] == "https://api.worldquantbrain.com/alphas/a1/recordsets/pnl"
    session.get.side_effect = [fake_response(500, {"detail": "oops"}, text="oops")]
    assert al.fetch_daily_pnl(session, "a1") is None


def test_your_submitted_alphas_are_listed_page_by_page_or_read_from_the_results_file(tmp_path):
    def run():
        brain = PortfolioBrain({}, {}, listed=[f"s{i}" for i in range(150)])
        with patch("time.sleep"):
            ids, source = al.submitted_alpha_ids(brain.session)
        assert (len(ids), source, ids[0], "draft" in ids) == (150, "BRAIN", "s0", False)   # only ACTIVE ones
        assert [(p["status"], p["offset"]) for p in brain.list_params] == [("ACTIVE", 0), ("ACTIVE", 100)]
        al.append_result(held_row("in1", 1.3, submitted="YES"))
        al.append_result(held_row("held1", 1.3))
        broken = MagicMock()
        broken.get.return_value = fake_response(500, {"detail": "down"}, text="down")
        with patch("time.sleep"):
            assert al.submitted_alpha_ids(broken) == (["in1"], "the results file")
    in_tmp(tmp_path, run)


def test_combined_effect_compares_your_portfolio_with_and_without_the_alpha():
    mine = [dict(zip(map(str, range(300)), MINE["mine1"]))]
    same = al.combined_effect(mine, mine[0])
    assert abs(same.after - same.before) < 1e-9 and abs(same.corr - 1) < 1e-9 and (same.days, same.alphas) == (300, 1)
    first = al.combined_effect([], dict(zip(map(str, range(300)), FRESH)))
    assert first.before is None and first.after > 0
    assert al.combined_effect(mine, {str(i): 1.0 for i in range(200)}) is None      # too few shared days
    days = lambda s: {(_d(2019, 1, 1) + _td(t + 1)).isoformat(): v for t, v in enumerate(s)}
    assert al.combined_effect([days(s) for s in MINE.values()], days(COPY)).after < 1.7 < \
           al.combined_effect([days(s) for s in MINE.values()], days(FRESH)).after


def test_an_alpha_that_would_lower_your_combined_portfolio_is_held(tmp_path):
    def run():
        brain = mine_and({"copy": COPY})
        with portfolio_on(), captured_logs() as lines:
            rows = run_one_idea(brain)
        assert rows[0]["submitted"] == al.HELD_PORTFOLIO_LABEL and submit_posts(brain) == []
        assert al.submissions_today() == 0 and al.RESERVE.kinds() == 0              # not counted as a reserve
        assert al.PORTFOLIO.blocked == {"copy"} and brain.before_after == ["copy"]
        assert any("3 submitted alpha(s), listed from BRAIN, combined Sharpe 1.80" in l for l in lines), lines
        assert any("Sharpe 1.80, 1.64 with it (-8.7%); its correlation with them is 0.87" in l
                   and "would lower your combined portfolio" in l for l in lines), lines
    in_tmp(tmp_path, run)


def test_an_alpha_that_raises_it_goes_in_and_joins_the_portfolio(tmp_path):
    def run():
        brain = mine_and({"fresh": FRESH, "copy": COPY})
        with portfolio_on(), captured_logs() as lines:
            assert run_one_idea(brain)[0]["submitted"] == "YES"
            assert set(al.PORTFOLIO.daily) == {"mine1", "mine2", "mine3", "fresh"}     # part of it from now on
            asked = len(brain.pnl_asked)
            effect = al.portfolio_effect(brain.session, "copy")
        assert any("Portfolio check on fresh: your 3 submitted alpha(s) combined have Sharpe 1.80, 2.50" in l
                   for l in lines), lines
        assert effect.alphas == 4 and brain.pnl_asked[asked:] == ["copy"]              # the portfolio isn't re-read
    in_tmp(tmp_path, run)


def test_the_portfolio_check_steps_aside_when_brain_wont_send_a_pnl(tmp_path):
    def run():
        brain = mine_and({"copy": COPY}, pnl_status=500)
        with portfolio_on(), captured_logs() as lines:
            assert run_one_idea(brain)[0]["submitted"] == "YES"                         # the usual rules decide
        assert any("Couldn't read the PnL of your submitted alpha mine1" in l for l in lines), lines
        assert al.PORTFOLIO.state == "unavailable"
    in_tmp(tmp_path, run)


def test_your_first_alpha_passes_the_portfolio_check(tmp_path):
    def run():
        brain = PortfolioBrain({"first": alpha_json()}, {"first": COPY}, listed=[])
        with portfolio_on():
            assert run_one_idea(brain)[0]["submitted"] == "YES"
    in_tmp(tmp_path, run)


def test_with_the_portfolio_check_off_no_pnl_is_read(tmp_path):
    def run():
        brain = mine_and({"copy": COPY})
        assert run_one_idea(brain)[0]["submitted"] == "YES" and brain.pnl_asked == []
    in_tmp(tmp_path, run)


def test_the_end_of_day_pick_goes_best_for_the_portfolio_first(tmp_path):
    def run():
        al.append_result(held_row("meh", 1.4))
        al.append_result(held_row("fresh", 1.2))
        brain = mine_and({"meh": MEH, "fresh": FRESH}, corr={"meh": 0.2, "fresh": 0.3})   # both raise it; meh less
        al.SETTLING = True
        with portfolio_on():
            assert run_held(brain) == "YES"
        assert submit_posts(brain) == ["fresh"] and submitted_value("meh") == al.HELD_LABEL
    in_tmp(tmp_path, run)


def test_an_alpha_whose_pnl_couldnt_be_read_goes_in_and_the_portfolio_is_read_again(tmp_path):
    def run():
        brain = PortfolioBrain({"nopnl": alpha_json(), "copy": alpha_json()}, dict(MINE, copy=COPY), listed=MINE)
        with portfolio_on():
            assert run_one_idea(brain)[0]["submitted"] == "YES"                 # the check couldn't be made
            assert al.PORTFOLIO.state == "unread"                               # so it's read again next time...
            listed = len(brain.list_params)
            al.portfolio_effect(brain.session, "copy")
        assert len(brain.list_params) > listed and "nopnl" in brain.pnl_asked[-2:]   # ...now with nopnl in it
    in_tmp(tmp_path, run)


def test_the_end_of_day_pick_lowers_the_portfolio_only_as_far_as_allowed(tmp_path):
    def run():
        al.append_result(held_row("copy", 1.4))
        brain = mine_and({"copy": COPY})
        al.SETTLING = True
        with portfolio_on(), captured_logs() as lines:
            assert run_held(brain) is None                                  # -8.7%: no submission rather than that
        assert submit_posts(brain) == [] and al.PORTFOLIO.blocked == {"copy"}
        assert any("stays queued" in l and "would lower your combined portfolio" in l for l in lines), lines
        with portfolio_on(END_OF_DAY_MAX_PORTFOLIO_DROP=0.05), captured_logs() as lines:
            assert run_held(brain) is None                                  # more than the 5% allowed
        assert any("more than the 5% drop the end-of-day pick may make" in l for l in lines), lines
        with portfolio_on(END_OF_DAY_MAX_PORTFOLIO_DROP=0.1):
            assert run_held(brain) == "YES"                                 # within 10%: the streak goes on
        assert submit_posts(brain) == ["copy"]
    in_tmp(tmp_path, run)


def test_during_the_day_the_end_of_day_allowance_doesnt_apply(tmp_path):
    def run():
        brain = mine_and({"copy": COPY})
        with portfolio_on(END_OF_DAY_MAX_PORTFOLIO_DROP=0.5):
            assert run_one_idea(brain)[0]["submitted"] == al.HELD_PORTFOLIO_LABEL
    in_tmp(tmp_path, run)


def test_held_alphas_that_would_lower_the_portfolio_dont_count_as_a_reserve(tmp_path):
    def run():
        al.append_result(held_row("copy", 1.4, submitted=al.HELD_PORTFOLIO_LABEL))
        al.append_result(held_row("other", 1.3))
        al.record_submission()                                              # today's is in: just re-checked
        with portfolio_on():
            run_held(mine_and({"copy": COPY, "other": FRESH}))
        assert list(al.RESERVE.waiting) == ["other"]
    in_tmp(tmp_path, run)


def test_a_day_the_portfolio_kept_out_ends_with_a_hint(tmp_path):
    def run():
        brain = PortfolioBrain({"r1": alpha_json(), "r2": calm()}, dict(MINE, r1=COPY), listed=MINE)
        with portfolio_on(), captured_logs() as lines:
            run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=1)
        assert al.submissions_today() == 0 and submitted_value("r1") == al.HELD_PORTFOLIO_LABEL
        assert any("1 good alpha(s) were kept out" in l and "END_OF_DAY_MAX_PORTFOLIO_DROP" in l for l in lines), lines
    in_tmp(tmp_path, run)


def test_competitions_are_logged_and_the_before_after_check_asks_for_yours(tmp_path):
    def run():
        challenge = {"id": "challenge", "name": "Challenge", "leaderboard": {"rank": 1484, "score": 0.27, "user": "me"}}
        brain = PortfolioBrain({"a1": alpha_json()}, {}, competitions=[challenge])
        with captured_logs() as lines:
            al.note_competitions(brain.session)
        assert al.ACTIVE_COMPETITION == "challenge"
        assert any("Competition challenge (Challenge) - you: rank 1484, score 0.27" in l for l in lines), lines
        with patch("time.sleep"):
            al.record_before_after(brain.session, "a1", "test")
        asked = [c.args[0] for c in brain.session.get.call_args_list if c.args[0].endswith("before-and-after-performance")]
        assert asked == ["https://api.worldquantbrain.com/competitions/challenge/alphas/a1/before-and-after-performance"]
        assert json.loads(Path(al.BEFORE_AFTER_FILE).read_text().splitlines()[-1])["competition"] == "challenge"
        two = PortfolioBrain({}, {}, competitions=[challenge, {"id": "IQC2026", "name": "IQC"}])
        with captured_logs() as lines:
            al.note_competitions(two.session)
        assert al.ACTIVE_COMPETITION == "" and any("Set BRAIN_COMPETITION" in l for l in lines), lines
        with patch.object(al, "BRAIN_COMPETITION", "IQC2026"):
            al.note_competitions(two.session)
        assert al.ACTIVE_COMPETITION == "IQC2026"
        al.note_competitions(PortfolioBrain({}, {}).session)              # BRAIN won't say: the account's own check
        assert al.ACTIVE_COMPETITION == ""
    in_tmp(tmp_path, run)


# ---- Riding out a lost connection: wifi dropping, a laptop waking up ----

class Outage:
    """The connection is down until BRAIN's "are you there?" probe has been asked `probes` times - that probe, and
    every one after it, gets an answer. A huge number means it never comes back."""
    def __init__(self, probes=3):
        self.down, self.left, self.asked = True, probes, 0

    def reachable(self):
        self.asked += 1
        self.left -= 1
        if self.left <= 0:
            self.down = False
        return not self.down


def cut_off(brain, outage, gets=lambda url: True, posts=lambda url: False):
    """While the outage lasts, requests the two filters pick fail like a dropped connection."""
    real_get, real_post = brain.get, brain.post

    def get(url, **kw):
        if outage.down and gets(url):
            raise _requests.ConnectionError("wifi dropped")
        return real_get(url, **kw)

    def post(url, **kw):
        if outage.down and posts(url):
            raise _requests.ConnectionError("wifi dropped")
        return real_post(url, **kw)

    brain.session.get.side_effect = get
    brain.session.post.side_effect = post


def test_brain_counts_as_reachable_on_any_reply_but_not_a_failed_connection():
    with patch.object(al.requests, "get", return_value=fake_response(401, {"detail": "no"})) as get:
        assert REAL_BRAIN_REACHABLE() is True                       # "not logged in" is still an answer
    assert get.call_args.args[0] == be.AUTH_URL and get.call_args.kwargs["timeout"] == al.CONNECTION_PROBE_TIMEOUT
    with patch.object(al.requests, "get", side_effect=_requests.ConnectionError("no wifi")):
        assert REAL_BRAIN_REACHABLE() is False


def test_waiting_for_the_connection_checks_every_minute_until_brain_answers():
    outage = Outage(probes=3)
    before = al.OUTAGE_SECONDS
    with patch.object(al, "brain_reachable", outage.reachable), patch("time.sleep") as sleep, \
         captured_logs() as lines:
        assert al.wait_for_connection("while testing") is True
    assert [c.args[0] for c in sleep.call_args_list] == [60, 60] and outage.asked == 3
    assert any("Lost the connection to BRAIN while testing" in line for line in lines)
    assert any("The connection is back" in line for line in lines)
    assert al.OUTAGE_SECONDS >= before                                # the wait is set aside from the runtime cap
    with patch.object(al, "brain_reachable", lambda: True), patch("time.sleep") as sleep:
        assert al.wait_for_connection("x") is True                    # already connected: no wait at all
    sleep.assert_not_called()


def test_waiting_for_the_connection_gives_up_after_the_limit():
    outage = Outage(probes=10 ** 6)
    with patch.object(al, "brain_reachable", outage.reachable), patch.object(al, "WAIT_FOR_CONNECTION_MINUTES", 3), \
         patch("time.sleep") as sleep:
        assert al.wait_for_connection("x") is False
    assert sleep.call_count == 3 and outage.asked == 4                # one look, then one every minute for 3 minutes


def test_lost_connection_while_simulations_run_waits_and_keeps_them(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm(), "rank(b)": calm()})
        outage = Outage(probes=3)
        cut_off(brain, outage, gets=lambda url: url.startswith("sim-"))   # checking on simulations fails
        with patch.object(al, "brain_reachable", outage.reachable), captured_logs() as lines:
            rows = run_queue(brain, ["rank(a)", "rank(b)"])
        assert sorted((r["expression"], r["status"]) for r in rows) == [("rank(a)", "COMPLETE"), ("rank(b)", "COMPLETE")]
        assert brain.started == ["rank(a)", "rank(b)"]                # nothing dropped, nothing simulated twice
        assert [r["status"] for r in csv_rows()] == ["COMPLETE", "COMPLETE"]
        assert any("while checking on simulations" in line for line in lines)
    in_tmp(tmp_path, run)


def test_lost_connection_while_starting_keeps_the_idea_first_in_line(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()})
        outage = Outage(probes=3)
        cut_off(brain, outage, gets=lambda url: False, posts=lambda url: url.endswith("/simulations"))
        with patch.object(al, "brain_reachable", outage.reachable):
            rows = run_queue(brain, ["rank(a)"])
        assert [(r["expression"], r["status"]) for r in rows] == [("rank(a)", "COMPLETE")]
        assert brain.started == ["rank(a)"] and [r["status"] for r in csv_rows()] == ["COMPLETE"]
    in_tmp(tmp_path, run)


def test_lost_connection_while_reading_a_result_reads_it_again(tmp_path):
    def run():
        brain = FakeBrain({"a1": calm()})
        outage = Outage(probes=3)
        cut_off(brain, outage, gets=lambda url: url.endswith("/alphas/a1"))
        with patch.object(al, "brain_reachable", outage.reachable):
            rows = run_one_idea(brain)
        assert [(r["alpha_id"], r["status"]) for r in rows] == [("a1", "COMPLETE")]
        assert [r["status"] for r in csv_rows()] == ["COMPLETE"]
    in_tmp(tmp_path, run)


def test_a_connection_that_never_comes_back_stops_cleanly_and_keeps_the_records(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()})
        outage = Outage(probes=10 ** 6)
        cut_off(brain, outage, gets=lambda url: url.startswith("sim-"))
        with patch.object(al, "brain_reachable", outage.reachable), \
             patch.object(al, "WAIT_FOR_CONNECTION_MINUTES", 2):
            try:
                run_queue(brain, ["rank(a)"])
                assert False, "should have stopped"
            except al.StopLoop as e:
                assert "No connection to BRAIN for 2 minutes" in str(e)
        assert [(r["expression"], r["status"]) for r in csv_rows()] == [("rank(a)", "STOPPED")]
    in_tmp(tmp_path, run)


def test_a_run_waits_for_the_connection_before_logging_in(tmp_path):
    def run():
        outage = Outage(probes=3)
        brain = FakeBrain({"i1": calm()})
        with patch.object(al, "brain_reachable", outage.reachable):
            rounds = run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=0)
        assert rounds == 1 and outage.asked >= 3
    in_tmp(tmp_path, run)


def test_no_connection_at_all_stops_before_logging_in(tmp_path):
    def run():
        _write_operators()
        outage = Outage(probes=10 ** 6)
        with patch.object(al, "brain_reachable", outage.reachable), patch.object(al, "WAIT_FOR_CONNECTION_MINUTES", 2), \
             patch.object(al, "get_authenticated_session", side_effect=AssertionError("must not log in")), \
             patch("google.genai.Client"), patch("time.sleep"), captured_logs() as lines:
            al.main([])
        assert any("No connection to BRAIN for 2 minutes - stopping before using any quota" in line for line in lines)
        assert not Path(al.RESULTS_FILE).exists()
    in_tmp(tmp_path, run)


def test_a_login_lost_to_the_connection_waits_and_tries_again(tmp_path):
    def run():
        session = MagicMock()
        outage = Outage(probes=3)
        logins = iter([False, True])                                  # the first try had no connection
        with patch.object(al, "brain_reachable", outage.reachable), patch("time.sleep"), \
             patch.object(be, "authenticate", side_effect=lambda *a: next(logins)) as auth, \
             patch.object(al, "note_expiry"), patch.object(al, "save_session"):
            assert al.log_in_again(session, "e@x.com", "pw") is True
        assert auth.call_count == 2
    in_tmp(tmp_path, run)


def test_a_refused_login_with_brain_answering_is_still_reported_as_verification(tmp_path):
    def run():
        session = MagicMock()
        session.brain_expires_at = _time.time() + 60
        with patch.object(be, "authenticate", return_value=False), patch("time.sleep") as sleep:
            try:
                al.keep_logged_in(session, "e@x.com", "pw")
                assert False, "should have stopped"
            except al.StopLoop as e:
                assert "identity verification" in str(e)
        sleep.assert_not_called()                                     # BRAIN answered: nothing to wait for
    in_tmp(tmp_path, run)


def test_session_renewal_waits_for_a_lost_connection(tmp_path):
    def run():
        brain = SlowBrain({"rank(a)": calm()}, expire_on="rank(a)")
        outage = Outage(probes=3)
        cut_off(brain, outage, gets=lambda url: False, posts=lambda url: url == be.AUTH_URL)
        with patch.object(al, "brain_reachable", outage.reachable):
            rows = run_queue(brain, ["rank(a)"])
        assert [(r["expression"], r["status"]) for r in rows] == [("rank(a)", "COMPLETE")]
        assert "STOPPED" not in [r["status"] for r in csv_rows()]
    in_tmp(tmp_path, run)


def test_gemini_failing_for_lack_of_a_connection_waits_without_counting_failures(tmp_path):
    def run():
        _write_operators()
        outage = Outage(probes=0)
        outage.down = False                                           # connected when the run starts...
        calls = {"n": 0}

        def gen(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:                                       # ...and the wifi drops as Gemini is asked
                outage.down, outage.left = True, 3
            if outage.down:
                raise _httpx.ConnectError("no wifi")
            return [(f"rank(idea{calls['n']})", "r")]

        brain = FakeBrain({"i1": calm(), "i2": calm()})
        with patch.object(al, "brain_reachable", outage.reachable), patch.object(al, "MAX_ROUNDS", 2), \
             patch.object(al, "MAX_EXTRA_ROUNDS", 0), \
             patch.object(al, "MAX_GENERATION_FAILURES", 1), patch.object(al, "IDEAS_PER_ROUND", 1), \
             patch.object(al, "generate_with_fallback", side_effect=gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"), captured_logs() as lines:
            al.main([])
        assert [r["round"] for r in csv_rows()] == ["1", "2"]         # one failure allowed, yet both rounds ran
        assert calls["n"] == 3 and outage.asked == 4                  # login, the failure, then a minute's wait
        assert any("Lost the connection to BRAIN while asking Gemini for ideas" in line for line in lines)
    in_tmp(tmp_path, run)


def test_the_computer_sleeping_mid_run_is_noticed():
    clock = {"t": 1000.0}

    def sleep(seconds):
        clock["t"] += seconds + 3 * 3600                              # lid closed for three hours

    before = al.OUTAGE_SECONDS
    with patch("time.time", lambda: clock["t"]), patch("time.sleep", sleep), captured_logs() as lines:
        assert al.nap(5) == 3 * 3600
    assert al.OUTAGE_SECONDS - before == 3 * 3600
    assert any("asleep for about 180 minute(s)" in line for line in lines)
    with patch("time.sleep"):
        assert al.nap(5) == 0.0                                       # an ordinary pause


def test_simulations_still_running_after_a_long_outage_are_not_timed_out(tmp_path):
    def run():
        clock = {"t": 1_000_000.0}

        def sleep(seconds):
            clock["t"] += seconds

        brain = SlowBrain({"rank(a)": calm()}, polls={"rank(a)": 3})      # still running when the wifi comes back
        outage = Outage(probes=60)                                         # about an hour without a connection
        cut_off(brain, outage, gets=lambda url: url.startswith("sim-"))
        with patch.object(al, "brain_reachable", outage.reachable), patch("time.time", lambda: clock["t"]), \
             patch("time.sleep", sleep), patch.object(al, "PARALLEL_SIMULATIONS", 1):
            rows = al.process_queue(brain.session, [("rank(a)", "r", "idea")], SETTINGS, "e@x.com", "pw", 1, set())
        assert [(r["expression"], r["status"]) for r in rows] == [("rank(a)", "COMPLETE")]
    in_tmp(tmp_path, run)


def test_simulations_running_while_the_computer_slept_are_not_timed_out(tmp_path):
    def run():
        clock = {"t": 1_000_000.0, "naps": 0}

        def sleep(seconds):
            clock["naps"] += 1
            clock["t"] += seconds + (2 * 3600 if clock["naps"] == 1 else 0)   # asleep for 2 hours mid-simulation

        brain = SlowBrain({"rank(a)": calm()}, polls={"rank(a)": 3})
        with patch("time.time", lambda: clock["t"]), patch("time.sleep", sleep), \
             patch.object(al, "PARALLEL_SIMULATIONS", 1):
            rows = al.process_queue(brain.session, [("rank(a)", "r", "idea")], SETTINGS, "e@x.com", "pw", 1, set())
        assert [(r["expression"], r["status"]) for r in rows] == [("rank(a)", "COMPLETE")]
    in_tmp(tmp_path, run)


def test_time_away_doesnt_count_against_the_runtime_cap(tmp_path):
    def run():
        _write_operators()
        clock = {"t": 1_000_000.0, "away": False}

        def sleep(seconds):
            clock["t"] += seconds + (3 * 3600 if clock["away"] else 0)
            clock["away"] = False

        def gen(*a, **kw):
            clock["away"] = True                                      # each round, the laptop sleeps for 3 hours
            return [(f"rank(idea{len(csv_rows()) if Path(al.RESULTS_FILE).exists() else 0})", "r")]

        brain = FakeBrain({f"i{n}": calm() for n in range(3)})
        with patch.object(al, "MAX_ROUNDS", 3), patch.object(al, "MAX_RUNTIME_SECONDS", 3600), \
             patch.object(al, "MAX_EXTRA_ROUNDS", 0), \
             patch.object(al, "IDEAS_PER_ROUND", 1), patch.object(al, "generate_with_fallback", side_effect=gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.time", lambda: clock["t"]), patch("time.sleep", sleep):
            al.main([])
        assert [r["round"] for r in csv_rows()] == ["1", "2", "3"]   # 9 hours asleep, 0 counted
    in_tmp(tmp_path, run)


# ---- Laptops: stay awake while plugged in ----

def test_the_computer_is_kept_awake_only_while_plugged_in():
    flags = []
    al.AWAKE = False
    with patch.object(al, "set_execution_state", side_effect=lambda f: flags.append(f) or True), \
         patch.object(al, "on_mains_power", side_effect=[True, True, False, None]):
        al.keep_awake()                                               # plugged in: stay awake
        al.keep_awake()                                               # no change: nothing asked
        al.keep_awake()                                               # unplugged: sleep as usual
        al.keep_awake()                                               # can't tell: same as on battery
    assert flags == [al.ES_CONTINUOUS | al.ES_SYSTEM_REQUIRED, al.ES_CONTINUOUS] and al.AWAKE is False


def test_a_run_lets_the_computer_sleep_again_when_it_ends(tmp_path):
    def run():
        flags = []
        al.AWAKE = False
        brain = FakeBrain({"i1": calm()})
        with patch.object(al, "set_execution_state", side_effect=lambda f: flags.append(f) or True), \
             patch.object(al, "on_mains_power", return_value=True):
            run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=0)
        assert flags == [al.ES_CONTINUOUS | al.ES_SYSTEM_REQUIRED, al.ES_CONTINUOUS] and al.AWAKE is False
    in_tmp(tmp_path, run)


def test_plugging_in_mid_run_is_noticed(tmp_path):
    def run():
        flags = []
        al.AWAKE = False
        power = iter([False, False] + [True] * 50)                    # on battery when the run starts
        brain = SlowBrain({"rank(idea1)": calm()}, polls={"rank(idea1)": 3})
        with patch.object(al, "set_execution_state", side_effect=lambda f: flags.append(f) or True), \
             patch.object(al, "on_mains_power", side_effect=lambda: next(power)), captured_logs() as lines:
            run_main_rounds(brain, MAX_ROUNDS=1, MAX_EXTRA_ROUNDS=0)
        assert flags[0] == al.ES_CONTINUOUS | al.ES_SYSTEM_REQUIRED and al.AWAKE is False
        plugged = next(i for i, line in enumerate(lines) if line.startswith("Plugged in"))
        assert plugged < next(i for i, line in enumerate(lines) if "rank(idea1) -> sharpe" in line)  # mid-round
    in_tmp(tmp_path, run)


def test_power_helpers_do_nothing_off_windows():
    if os.name != "nt":
        assert REAL_ON_MAINS_POWER() is None and al.set_execution_state(al.ES_CONTINUOUS) is False


# ---- --catch-up: extra triggers that only act when today isn't covered ----

def test_catch_up_does_nothing_when_today_is_in(tmp_path):
    def run():
        al.record_submission()
        with patch.object(al, "get_authenticated_session", side_effect=AssertionError("must not log in")), \
             captured_logs() as lines:
            al.main(["--catch-up"])
        assert any("today's submission is already in - nothing to do" in line for line in lines)
    in_tmp(tmp_path, run)


def test_catch_up_steps_aside_quietly_for_a_run_already_going(tmp_path):
    def run():
        held = al.acquire_run_lock()
        records = []

        class Grab(_logging.Handler):
            def emit(self, record):
                records.append(record)

        grab = Grab()
        al.logger.addHandler(grab)
        try:
            with patch.object(al, "get_authenticated_session", side_effect=AssertionError("must not log in")):
                al.main(["--catch-up"])
        finally:
            al.logger.removeHandler(grab)
            held.close()
        assert [r.getMessage() for r in records] == ["Catch-up run: another run is going - leaving today to it."]
        assert all(r.levelno < _logging.WARNING for r in records)     # expected, so not an error in the log
    in_tmp(tmp_path, run)


def test_catch_up_stops_as_soon_as_todays_submission_lands(tmp_path):
    def run():
        brain = FakeBrain({"i1": alpha_json(), "i2": calm(), "i3": calm()})   # round 1's idea qualifies
        assert run_main_rounds(brain, ["--catch-up"], MAX_ROUNDS=3) == 1
        assert al.submissions_today() == 1
    in_tmp(tmp_path, run)


def test_catch_up_stops_when_a_held_alpha_covers_today(tmp_path):
    def run():
        al.append_result(held_row("held1", 1.3))
        brain = FakeBrain({"held1": alpha_json(sharpe=1.6, fitness=1.3)})
        with captured_logs() as lines:
            assert run_main_rounds(brain, ["--catch-up"], MAX_ROUNDS=3) == 0
        assert submitted_value("held1") == "YES"
        assert "Catch-up run: a held alpha covered today - nothing more to do." in lines
    in_tmp(tmp_path, run)


def test_a_normal_run_still_mines_after_todays_submission(tmp_path):
    def run():
        brain = FakeBrain({"i1": alpha_json(), "i2": calm(), "i3": calm()})
        assert run_main_rounds(brain, MAX_ROUNDS=3) == 3
    in_tmp(tmp_path, run)


# ---- The end of BRAIN's day ----

def test_brain_day_ends_at_3am_us_eastern():
    assert al.brain_day_ends_in(_dt(2026, 10, 14, 6, 30, tzinfo=_tz.utc)).total_seconds() == 30 * 60   # summer time
    assert al.brain_day_ends_in(_dt(2026, 10, 14, 7, 0, tzinfo=_tz.utc)).total_seconds() == 24 * 3600
    assert al.brain_day_ends_in(_dt(2026, 12, 1, 7, 30, tzinfo=_tz.utc)).total_seconds() == 30 * 60    # winter time
    assert al.brain_day_ends_in(_dt(2026, 11, 1, 7, 30, tzinfo=_tz.utc)).total_seconds() == 30 * 60    # clocks go back
    assert al.brain_day_ends_in(_dt(2026, 3, 8, 6, 30, tzinfo=_tz.utc)).total_seconds() == 30 * 60     # clocks go forward


def test_a_run_settles_straight_away_when_brains_day_is_about_to_end(tmp_path):
    def run():
        al.append_result(held_row("weak", 1.2))
        brain = FakeBrain({"weak": weak_json(fitness=1.2), "i1": calm()})
        from datetime import timedelta as _td
        with real_strong_bar(), patch.object(al, "SETTLE_BEFORE_DAY_END_MINUTES", 60), \
             patch.object(al, "brain_day_ends_in", return_value=_td(minutes=30)), captured_logs() as lines:
            run_main_rounds(brain, MAX_ROUNDS=1)
        assert submitted_value("weak") == "YES"
        settled = next(i for i, line in enumerate(lines) if "settling now rather than waiting" in line)
        assert settled < next(i for i, line in enumerate(lines) if "=== Round 1" in line)   # before any round
    in_tmp(tmp_path, run)


def test_a_new_brain_day_mid_run_goes_back_to_strong_first(tmp_path):
    def run():
        from datetime import timedelta as _td
        state = {"day": _date(2026, 10, 14)}

        def gen(*a, **kw):
            n = len(csv_rows()) if Path(al.RESULTS_FILE).exists() else 0
            state["day"] = _date(2026, 10, 15)                        # BRAIN's day rolls over during round 1
            return [(f"rank(idea{n})", "r")]

        brain = FakeBrain({"i1": calm(), "i2": weak_json()})          # round 2 brings a weak alpha
        _write_operators()
        with real_strong_bar(), patch.object(al, "SETTLE_BEFORE_DAY_END_MINUTES", 60), \
             patch.object(al, "brain_day", side_effect=lambda *a: state["day"]), \
             patch.object(al, "brain_day_ends_in",
                          side_effect=lambda *a: _td(minutes=30) if state["day"].day == 14 else _td(hours=20)), \
             patch.object(al, "MAX_ROUNDS", 2), patch.object(al, "MAX_EXTRA_ROUNDS", 0), \
             patch.object(al, "IDEAS_PER_ROUND", 1), patch.object(al, "generate_with_fallback", side_effect=gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"), captured_logs() as lines:
            al.main([])
        assert any("BRAIN day 2026-10-15 has started" in line for line in lines)
        # In the new day the weak alpha is held for a strong one first (not rushed in); the usual end-of-rounds
        # settling then sends it, as on any day.
        assert any("Alpha i2 qualifies" in line and "holding it for now" in line for line in lines)
        assert submitted_value("i2") == "YES"
    in_tmp(tmp_path, run)


# ---- Padding: add(x, 0.0001) ----

IV_RATIO = ("ts_decay_linear(group_rank(divide(implied_volatility_call_180, add(implied_volatility_put_180, 0.0001)), "
            "sector), 10)")                                            # E5VW6LeK: Sharpe 2.02, failed the sub-universe check
IV_RATIO_BARE = "ts_decay_linear(group_rank(divide(implied_volatility_call_180, implied_volatility_put_180), sector), 10)"


def test_padding_is_taken_out_and_nothing_else():
    cases = {
        IV_RATIO: IV_RATIO_BARE,
        "ts_delta(iv, 30) / add(ts_std_dev(iv, 60), 0.0001)": "ts_delta(iv, 30) / ts_std_dev(iv, 60)",
        "add(0.0001, x)": "x",
        "add(add(x, 0.0001), 1e-4)": "x",
        "rank(x / (ts_std_dev(x, 20) + 0.0001))": "rank(x / (ts_std_dev(x, 20)))",
        "rank((1e-4 + x) * y)": "rank((x) * y)",
        "ts_mean(x+0.0001, 20)": "ts_mean(x, 20)",
        "x * 1e+05 + 0.0001": "x * 1e+05",
        "group_neutralize(add(x, 0.0001), bucket(rank(cap), range='0, 1, 0.1'))":
            "group_neutralize(x, bucket(rank(cap), range='0, 1, 0.1'))",
    }
    for padded, bare in cases.items():
        assert al.strip_padding(padded) == bare, padded
    for untouched in ("add(x, 0.5)", "add(x, y)", "x + 0.0001 * y", "add(x, 0.0001, filter=true)", "x + 0",
                      "add(x, 0.0001, 0.0001)",
                      "rank(close - 0.0001)", "group_neutralize(x, bucket(rank(cap), range='0.0001, 1, 0.1'))"):
        assert al.strip_padding(untouched) == untouched, untouched


def test_gemini_ideas_are_tested_without_their_padding(tmp_path):
    def run():
        brain = SlowBrain({"rank(divide(a, b))": calm()})
        with captured_logs() as lines:
            run_one_idea(brain, "rank(divide(a, add(b, 0.0001)))")
        assert brain.started == ["rank(divide(a, b))"]
        assert any("Took the add(x, 0.0001)-style padding out of 1 idea(s)" in line for line in lines)
    in_tmp(tmp_path, run)


def test_an_idea_differing_only_by_padding_counts_as_already_tried(tmp_path):
    def run():
        al.append_result(dict(_history_row("rank(divide(a, add(b, 0.0001)))", 0.5, 0.2, 0.1), alpha_id="old"))
        assert "rank(divide(a, b))" in al.already_tried()
        avoid = al.already_tried()
        brain = SlowBrain({})
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(
            [{"expression": "rank(divide(a, b))", "rationale": "r"}])
        with patch("time.sleep"):
            al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", avoid)
        assert brain.started == []
    in_tmp(tmp_path, run)


def test_the_generator_is_told_not_to_pad_and_what_the_checks_mean():
    client = MagicMock()
    client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
    al.generate_ideas(client, "m", [], [], SETTINGS, "feedback", set(), 3)
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "never pad a division with a tiny number like add(x, 0.0001)" in prompt
    assert "LOW_SUB_UNIVERSE_SHARPE - it must also work among the largest, most liquid stocks" in prompt


# ---- Smoothing toward the strong bar ----

def test_a_strong_sharpe_held_back_by_turnover_is_smoothed_toward_the_strong_bar(tmp_path):
    def run():
        # Your mLYwwez5's numbers: Sharpe 2.02, fitness 1.29 at 17.6% turnover - at 12.5% it'd be about 1.53.
        # Tested as given (like --try), padding and all: the smoothed version is built without the padding.
        results = {IV_RATIO: alpha_json(sharpe=2.02, fitness=1.29, turnover=0.1764, returns=0.0715, drawdown=0.0391),
                   f"ts_decay_linear({IV_RATIO_BARE}, 10)": strong_json(turnover=0.12)}
        brain = SlowBrain(results)
        brain.session.post.side_effect = lambda url, **kw: (
            fake_response(201, {}, headers={}) if url.endswith("/submit") else brain.post(url, **kw))
        with real_strong_bar(), patch.object(al, "submit_alpha", return_value=("YES", None)), \
             patch.object(al, "is_active_on_brain", return_value=True):
            rows = run_queue(brain, [IV_RATIO], parallel=1)
        assert [r["expression"] for r in rows] == [IV_RATIO, f"ts_decay_linear({IV_RATIO_BARE}, 10)"]
        assert "misses the strong bar (1.3)" in rows[1]["rationale"]
        assert rows[0]["submitted"].startswith(al.HELD_PREFIX) and rows[1]["submitted"] == "YES"
    in_tmp(tmp_path, run)


def test_a_flip_of_a_padded_expression_is_built_without_the_padding(tmp_path):
    def run():
        padded = "rank(divide(a, add(b, 0.0001)))"
        brain = SlowBrain({padded: alpha_json(sharpe=-1.5, checks=[]), "reverse(rank(divide(a, b)))": calm()})
        rows = run_queue(brain, [padded], parallel=1)
        assert [r["expression"] for r in rows] == [padded, "reverse(rank(divide(a, b)))"]
    in_tmp(tmp_path, run)


def test_tried_expressions_are_remembered_with_and_without_padding():
    avoid = set()
    al.note_tried(avoid, "rank(divide(a, add(b, 0.0001)))")
    assert avoid == {"rank(divide(a, add(b, 0.0001)))", "rank(divide(a, b))"}


def test_smoothing_toward_strong_only_when_it_could_get_there():
    def row(sharpe=2.02, fitness=1.29, turnover=0.1764, **kw):
        return dict(dict(sharpe=str(sharpe), fitness=str(fitness), turnover=str(turnover), submitted="NO"), **kw)

    with real_strong_bar():
        assert al.smoothing_could_make_it_strong(row())
        assert al.smoothing_could_make_it_strong(row(submitted=al.HELD_FOR_STRONGER_LABEL))
        assert not al.smoothing_could_make_it_strong(row(sharpe=1.8, fitness=1.0, turnover=0.18))     # 1.2 at best
        assert not al.smoothing_could_make_it_strong(row(turnover=0.15))                     # trades little already
        assert not al.smoothing_could_make_it_strong(row(sharpe=1.59))                       # Sharpe isn't strong
        assert not al.smoothing_could_make_it_strong(row(fitness=1.3))                       # already strong
        assert not al.smoothing_could_make_it_strong(row(failed_checks=["LOW_SUB_UNIVERSE_SHARPE"]))  # smoothing won't fix that
        for done in ("YES", "VERDICT_PENDING (check with BRAIN in the dashboard)", "TOO_CORRELATED (max 0.91)",
                     "REJECTED: failed SELF_CORRELATION"):
            assert not al.smoothing_could_make_it_strong(row(submitted=done)), done
        assert not al.smoothing_could_make_it_strong(row(sharpe=""))


# ---- BRAIN's value and limit for a check that didn't pass ----

LSUS_WITH_NUMBERS = {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.6213, "limit": 0.8746}


def test_checks_that_didnt_pass_show_brains_value_and_limit():
    assert al.check_note(LSUS_WITH_NUMBERS) == "LOW_SUB_UNIVERSE_SHARPE:FAIL (0.62 vs limit 0.87)"
    assert al.check_note({"name": "SELF_CORRELATION", "result": "PENDING"}) == "SELF_CORRELATION:PENDING"
    assert al.failed_text([LSUS_WITH_NUMBERS], ["LOW_SUB_UNIVERSE_SHARPE", "OTHER"]) == \
        "LOW_SUB_UNIVERSE_SHARPE (0.62 vs limit 0.87), OTHER"


def test_the_log_and_the_generator_see_the_value_and_limit(tmp_path):
    def run():
        checks = [LSUS_WITH_NUMBERS, {"name": "SELF_CORRELATION", "result": "PENDING"}]
        brain = SlowBrain({DIVIDEND: near_miss(checks)})
        with captured_logs() as lines:
            rows = run_queue(brain, [DIVIDEND], parallel=1)
        assert "  Strong numbers but failed LOW_SUB_UNIVERSE_SHARPE (0.62 vs limit 0.87)." in lines, lines
        assert "LOW_SUB_UNIVERSE_SHARPE:FAIL (0.62 vs limit 0.87)" in al.summarize_round(rows)
    in_tmp(tmp_path, run)


# ---- The big-stocks check: no follow-ups, and the generator hears which ideas fail it ----

FAR_GAP = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.10, "limit": 0.68},   # your seven
           {"name": "SELF_CORRELATION", "result": "PENDING"}]
NEAR_GAP = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.61, "limit": 0.93},  # the call/put ratio
            {"name": "SELF_CORRELATION", "result": "PENDING"}]


def test_a_strong_alpha_failing_the_big_stocks_check_gets_no_follow_up(tmp_path):
    def run():
        brain = SlowBrain({DIVIDEND: near_miss(FAR_GAP)})
        with captured_logs() as lines:
            rows = run_queue(brain, [DIVIDEND], parallel=1)
        assert brain.started == [DIVIDEND] and len(rows) == 1
        assert "  Strong numbers but failed LOW_SUB_UNIVERSE_SHARPE (0.10 vs limit 0.68)." in lines, lines
    in_tmp(tmp_path, run)


def test_big_stocks_numbers_are_saved_with_each_result(tmp_path):
    def run():
        brain = SlowBrain({DIVIDEND: near_miss(NEAR_GAP), "rank(close)": calm()})
        run_queue(brain, [DIVIDEND, "rank(close)"], parallel=1)
        saved = {r["expression"]: (r["sub_universe_sharpe"], r["sub_universe_limit"]) for r in csv_rows()}
        assert saved[DIVIDEND] == ("0.61", "0.93") and saved["rank(close)"] == ("", "")  # calm() has no such check
    in_tmp(tmp_path, run)


def test_filling_in_the_numbers_leaves_a_locked_file_alone(tmp_path):
    def run():
        al.append_result(dict(_history_row(DIVIDEND, 1.35, 1.34, 0.13), alpha_id="div"))
        with patch.object(al.os, "replace", side_effect=PermissionError("open in Excel")), patch("time.sleep"):
            al.record_sub_universe("div", 0.1, 0.68)
        assert csv_rows()[0]["sub_universe_sharpe"] == "" and not Path(al.RESULTS_FILE + ".tmp").exists()
        al.record_sub_universe("div", 0.1, 0.68)
        al.record_sub_universe("div", 0.5, 0.9)                   # numbers already there are kept
        assert (csv_rows()[0]["sub_universe_sharpe"], csv_rows()[0]["sub_universe_limit"]) == ("0.1", "0.68")
    in_tmp(tmp_path, run)


def test_every_version_of_an_idea_looks_alike():
    core = "group_rank(divide(implied_volatility_call_90, implied_volatility_put_90), sector)"
    versions = [
        f"ts_decay_linear({core}, 10)",
        "ts_decay_linear(group_rank(divide(implied_volatility_call_90, add(implied_volatility_put_90, 0.0001)), "
        "sector), 10)",
        f"rank(group_neutralize(ts_decay_linear({core}, 10), bucket(rank(cap), range='0, 1, 0.1')))",
        f"ts_decay_linear(group_neutralize(ts_decay_linear({core}, 10), bucket(rank(cap), range='0, 1, 0.1')), 15) "
        f"* rank(ts_mean(volume * close, 20)) + ts_decay_linear(group_neutralize(ts_decay_linear({core}, 10), "
        f"bucket(rank(cap), range='0, 1, 0.1')), 30) * (1 - rank(ts_mean(volume * close, 20)))",
    ]
    assert {al.idea_core(v) for v in versions} == {core}
    assert al.idea_core("ts_delta(dividend, 90)") == "ts_delta(dividend, 90)"


def test_the_generator_is_told_which_ideas_only_work_in_small_stocks(tmp_path):
    def run():
        def saved(expression, sharpe, value, limit, when):
            al.append_result(dict(_history_row(expression, sharpe, 1.5, 0.15), timestamp=when,
                                  sub_universe_sharpe=value, sub_universe_limit=limit))

        saved("ts_decay_linear(group_rank(divide(implied_volatility_call_150, implied_volatility_put_150), "
              "sector), 10)", 2.04, "0.61", "0.93", "2026-10-15 11:14:16")
        saved("group_neutralize(ts_decay_linear(group_rank(divide(implied_volatility_call_90, "
              "implied_volatility_put_90), subindustry), 10), bucket(rank(cap), range='0, 1, 0.1'))",
              2.13, "0.57", "0.93", "2026-10-15 11:25:27")             # same idea, other maturity and group
        saved("ts_decay_linear(group_rank(ts_delta(dividend, 90), sector), 20)", 1.4, "0.10", "0.68",
              "2026-10-15 11:06:49")
        saved("ts_decay_linear(group_rank(ts_zscore(rel_ret_all, 60), sector), 10)", 1.86, "0.95", "0.80",
              "2026-10-15 11:30:00")                                   # passed the check: not listed
        saved("rank(close)", 0.5, "", "", "2026-10-15 11:31:00")       # no numbers: not listed
        rule = al.small_stocks_rule()
        assert rule.startswith("ONLY WORKS IN SMALL STOCKS - don't write new versions of these ideas.")
        assert rule.splitlines()[1:] == [
            "- group_rank(divide(implied_volatility_call_90, implied_volatility_put_90), subindustry) "
            "(Sharpe 2.13 overall, 0.57 among the biggest stocks - needs 0.93)",
            "- group_rank(ts_delta(dividend, 90), sector) (Sharpe 1.4 overall, 0.10 among the biggest stocks - "
            "needs 0.68)"]
        feedback = al.build_feedback()  # the memory of best results leaves the small-stock ones out
        assert "implied_volatility" not in feedback and "dividend" not in feedback
        assert "ts_decay_linear(group_rank(ts_zscore(rel_ret_all, 60), sector), 10)" in feedback
        assert "ones that only worked in small stocks, or use data left off your field list, are left out" in feedback
        write_dataset_cache({"rel_ret_all": "pv13", "sales": "fundamental6"})   # pv13 left off the field list
        Path(al.OPERATORS_FILE).write_text(json.dumps({"operators": [], "fields": [{"name": "sales", "description": "d"}]}))
        assert "rel_ret_all" not in al.build_feedback()
        assert "rel_ret_all, 60), sector), 10) -> sharpe=1.86 fitness=1.5 turnover=0.15\n" in feedback + "\n"
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
        with patch("time.sleep"):
            al.run_round(SlowBrain({"rank(close)": calm()}).session, client, [], [], SETTINGS, "e@x.com", "pw", 1,
                         "round 1", {"rank(close)"})
        assert "ONLY WORKS IN SMALL STOCKS" in client.models.generate_content.call_args.kwargs["contents"]
    in_tmp(tmp_path, run)


def test_no_small_stocks_rule_until_an_idea_fails_the_check(tmp_path):
    def run():
        assert al.small_stocks_rule() == ""
        al.append_result(dict(_history_row("rank(close)", 1.4, 1.1, 0.1), sub_universe_sharpe="0.7",
                              sub_universe_limit="0.6"))
        assert al.small_stocks_rule() == "" and "big-stocks" not in al.build_feedback()
    in_tmp(tmp_path, run)


def test_only_the_most_recent_examples_are_shown(tmp_path):
    def run():
        for n in range(al.SMALL_STOCKS_EXAMPLES + 3):
            al.append_result(dict(_history_row(f"rank(ts_delta(field{chr(97 + n)}, 5))", 1.4, 1.1, 0.1),
                                  timestamp=f"2026-10-15 11:{n:02d}:00", sub_universe_sharpe="0.1",
                                  sub_universe_limit="0.6"))
        lines = al.small_stocks_rule().splitlines()[1:]
        assert len(lines) == al.SMALL_STOCKS_EXAMPLES and lines[0].startswith("- ts_delta(fieldk, 5) (")   # newest
    in_tmp(tmp_path, run)


# ---- A time limit on every Gemini request ----

def test_gemini_requests_get_a_time_limit():
    with patch("google.genai.Client") as client:
        al.gemini_client()
    assert client.call_args.kwargs["http_options"].timeout == al.GEMINI_TIMEOUT_SECONDS * 1000 == 300_000


def test_a_run_asks_gemini_with_the_time_limit(tmp_path):
    def run():
        _write_operators()
        brain = FakeBrain({})
        with patch.object(al, "MAX_ROUNDS", 0), patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client") as client, patch("time.sleep"):
            al.main([])
        assert client.call_args.kwargs["http_options"].timeout == 300_000
    in_tmp(tmp_path, run)


def test_an_older_gemini_library_still_works_without_a_time_limit():
    plain = MagicMock()
    with patch("google.genai.Client", side_effect=[TypeError("unexpected keyword"), plain]), \
         captured_logs() as lines:
        assert al.gemini_client() is plain
    assert any("can't put a time limit on Gemini requests" in line for line in lines)

# ---- Auto-blend: a good Sharpe that misses only on fitness, added to your best slow alphas ----

REV = ("reverse(ts_decay_linear(group_neutralize(group_rank(divide(close, vwap), sector), "
       "bucket(rank(cap), range='0, 1, 0.1')), 10))")
EBIT = ("ts_decay_linear(group_neutralize(group_rank(ts_delta(anl4_ebit_value, 90) / ts_std_dev(anl4_ebit_value, 180), "
        "industry), bucket(rank(cap), range='0, 1, 0.1')), 20)")
UNSYS = "ts_decay_linear(group_rank(ts_zscore(unsystematic_risk_last_90_days, 60), sector), 15)"
BLEND_DATASETS = {"anl4_ebit_value": "analyst4", "sales_estimate_median_value": "analyst4",
                  "unsystematic_risk_last_90_days": "model51", "implied_volatility_put_90": "option8",
                  "snt_buzz": "socialmedia12", "close": "pv1", "vwap": "pv1", "cap": "pv1"}
FAST_CHECKS = [{"name": "LOW_FITNESS", "result": "FAIL"},
               {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": 1.09, "limit": 0.86},
               {"name": "SELF_CORRELATION", "result": "PENDING"}]
BLEND_EBIT, BLEND_UNSYS = f"rank({REV}) + rank({EBIT})", f"rank({REV}) + rank({UNSYS})"
SMOOTHED_REV = f"ts_decay_linear({REV}, 10)"


def fast_json(checks=None, **kw):
    """Your close/vwap reversal: Sharpe 1.99, fitness 0.99 at 55% turnover, big-stocks check passed."""
    stats = dict(sharpe=1.99, fitness=0.99, turnover=0.5546, checks=FAST_CHECKS if checks is None else checks)
    stats.update(kw)
    return alpha_json(**stats)


def saved_row(expression, sharpe, turnover, fitness=1.4, big="0.9", limit="0.8", alpha_id="", submitted="NO",
              passed="7", total="8", rationale="r"):
    row = _history_row(expression, sharpe, fitness, turnover)
    row.update(alpha_id=alpha_id, submitted=submitted, sub_universe_sharpe=big, sub_universe_limit=limit,
               checks_passed=passed, checks_total=total, rationale=rationale)
    return row


def save_partners():
    """Your slow alphas, as the results file has them, plus ones that mustn't be picked as partners."""
    write_dataset_cache(BLEND_DATASETS)
    for row in [
        saved_row(EBIT, 1.96, 0.0416, big="0.85", limit="0.85", submitted="TOO_CORRELATED (max 0.81)"),  # best
        saved_row(UNSYS, 1.7, 0.096, big="0.9", limit="0.74", submitted="YES"),                       # other data
        saved_row("ts_decay_linear(group_rank(ts_delta(anl4_ebit_value, 60), sector), 20)", 1.5, 0.05),  # same data
        saved_row("rank(ts_delta(close, 5))", 2.0, 0.3),                                              # not slow
        saved_row("ts_decay_linear(group_rank(implied_volatility_put_90, sector), 20)", 2.1, 0.06,
                  big="0.5", limit="0.9"),                                                             # small stocks
        saved_row("ts_decay_linear(group_rank(sales_estimate_median_value, sector), 20)", 1.9, 0.05,
                  fitness=0.8),                                                                         # under the bar
        saved_row("ts_decay_linear(group_rank(snt_buzz, sector), 20)", 1.95, 0.05, passed="5"),      # a check failed
        saved_row("ts_decay_linear(group_rank(ts_delta(snt_buzz, 5), sector), 20)", 1.94, 0.05, big="", limit=""),
    ]:
        al.append_result(row)


def test_operator_count_matches_brains_count():
    assert al.operator_count("rank(a) + ts_mean(b, 5) * 2") == 4
    assert al.operator_count("group_neutralize(x, bucket(rank(cap), range='0, 1, 0.1'))") == 3  # quotes don't count
    assert al.operator_count("ts_decay_linear(rank(close > open), 5) - 1") == 4
    sixty_four = "rank(" * 64 + "close" + ")" * 64
    assert not al.too_long(sixty_four) and al.too_long(f"rank({sixty_four})")


def test_expressions_over_brains_operator_limit_are_never_sent(tmp_path):
    def run():
        long_one = "rank(" * 65 + "close" + ")" * 65
        brain = SlowBrain({"rank(close)": calm()})
        with captured_logs() as lines:
            rows = run_queue(brain, [long_one, "rank(close)"], parallel=1)
        assert brain.started == ["rank(close)"] and len(rows) == 1
        assert any("it has 65 operators, and BRAIN refuses more than 64" in line for line in lines), lines
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(
            [{"expression": long_one, "rationale": "r"}, {"expression": "rank(open)", "rationale": "r"}])
        brain = SlowBrain({"rank(open)": calm()})
        with patch("time.sleep"), captured_logs() as lines:
            al.run_round(brain.session, client, [], [], SETTINGS, "e@x.com", "pw", 1, "round 1", set())
        assert brain.started == ["rank(open)"]
        assert any("Skipped 1 idea(s) with more than 64 operators" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_which_alphas_are_worth_a_blend():
    good = {"sharpe": "1.99", "fitness": "0.99", "turnover": "0.5546", "sub_universe_sharpe": "1.09",
            "sub_universe_limit": "0.86"}
    assert al.blend_worthy(good)
    assert al.blend_worthy(dict(good, sub_universe_sharpe="", sub_universe_limit=""))       # numbers unknown
    assert al.blend_worthy(dict(good, failed_checks=["LOW_FITNESS", "HIGH_TURNOVER"]))
    assert not al.blend_worthy(dict(good, failed_checks=["LOW_FITNESS", "CONCENTRATED_WEIGHT"]))
    assert not al.blend_worthy(dict(good, sub_universe_sharpe="0.5"))                      # small stocks only
    assert not al.blend_worthy(dict(good, sharpe="1.2"))                                   # Sharpe isn't there
    assert not al.blend_worthy(dict(good, fitness="1.0"))                                  # nothing to fix
    assert not al.blend_worthy(dict(good, turnover="1.0"))                                 # still over 70% blended
    assert al.blend_worthy(dict(good, turnover="0.98"))
    assert not al.blend_worthy(dict(good, sharpe=""))


def test_partners_are_your_best_slow_alphas_on_other_data(tmp_path):
    def run():
        save_partners()
        assert [p for p, _ in al.blend_partners(REV)] == [EBIT, UNSYS]
        assert [p for p, _ in al.blend_partners("rank(anl4_ebit_value)")] == [UNSYS]          # not its own data
        # Datasets left off your field list stay out of blends too.
        Path(al.OPERATORS_FILE).write_text(json.dumps({"operators": [], "fields": [
            {"name": "sales_estimate_median_value", "description": "d"}, {"name": "close", "description": "d"}]}))
        assert [p for p, _ in al.blend_partners(REV)] == [EBIT]
        assert al.listed_datasets() == {"analyst4"}
        # Too long to add: the blend would pass BRAIN's operator limit.
        assert al.blend_partners("rank(" * 60 + "close" + ")" * 60) == []
        # A blend is never a partner: it already carries a fast signal.
        Path(al.OPERATORS_FILE).unlink()
        al.append_result(saved_row(f"{REV} + {EBIT}", 2.76, 0.128, big="1.33", limit="1.2"))
        al.append_result(saved_row("rank(ts_zscore(anl4_ebit_value, 90))", 2.5, 0.05, rationale="Auto-blend of x: y."))
        assert [p for p, _ in al.blend_partners(REV)] == [EBIT, UNSYS]
    in_tmp(tmp_path, run)


def test_a_fast_alpha_missing_only_fitness_is_blended_right_after(tmp_path):
    def run():
        save_partners()
        brain = SlowBrain({REV: fast_json(), BLEND_EBIT: calm(), BLEND_UNSYS: calm(), SMOOTHED_REV: calm(),
                           "rank(b)": calm()})
        with captured_logs() as lines:
            rows = run_queue(brain, [REV, "rank(b)"], parallel=1)
        assert brain.started == [REV, BLEND_EBIT, BLEND_UNSYS, SMOOTHED_REV, "rank(b)"]
        blend = next(r for r in rows if r["expression"] == BLEND_EBIT)
        assert blend["rationale"].startswith(f"Auto-blend of {REV}: added to your slow alpha {EBIT} (Sharpe 1.96, "
                                             f"turnover 4.2%)")
        assert any("testing it added to 2 of your best slow alphas next" in line for line in lines), lines
    in_tmp(tmp_path, run)


def test_a_blend_that_clears_the_bar_is_submitted(tmp_path):
    def run():
        save_partners()
        brain = SlowBrain({REV: fast_json(), BLEND_EBIT: alpha_json(sharpe=2.6, fitness=1.6, turnover=0.39),
                           BLEND_UNSYS: calm(), SMOOTHED_REV: calm()})
        with patch.object(al, "submit_alpha", return_value=("YES", None)), \
             patch.object(al, "is_active_on_brain", return_value=True):
            rows = run_queue(brain, [REV], parallel=1)
        assert {r["expression"]: r["submitted"] for r in rows}[BLEND_EBIT] == "YES"
    in_tmp(tmp_path, run)


def test_blends_are_capped_and_never_blended_again(tmp_path):
    def run():
        save_partners()
        fast = [REV.replace("10))", f"{d}))") for d in (10, 12, 14)]
        results = {e: fast_json() for e in fast}
        for e in fast:  # every blend misses on fitness again
            results[f"rank({e}) + rank({EBIT})"] = fast_json()
            results[f"rank({e}) + rank({UNSYS})"] = fast_json()
            results[f"ts_decay_linear({e}, 10)"] = calm()
        brain = SlowBrain(results)
        rows = run_queue(brain, fast, parallel=1)
        blends = [r for r in rows if r["rationale"].startswith("Auto-blend")]
        assert len(blends) == al.MAX_BLENDS_PER_ROUND == 4
        assert not any(r["expression"].startswith("rank(rank(") for r in rows)
    in_tmp(tmp_path, run)


def test_at_most_two_blends_per_alpha_and_the_round_cap_holds_exactly(tmp_path):
    def run():
        save_partners()
        write_dataset_cache(dict(BLEND_DATASETS, rel_ret_all="pv13"))
        al.append_result(saved_row("ts_decay_linear(group_rank(ts_zscore(rel_ret_all, 60), sector), 10)", 1.65, 0.13))
        assert len(al.blend_partners(REV)) == 3 and len(al.blend_candidates(REV)) == 2
        fast = [REV.replace("10))", f"{d}))") for d in (10, 12)]
        results = {e: fast_json() for e in fast}
        for e in fast:
            for partner in (EBIT, UNSYS):
                results[f"rank({e}) + rank({partner})"] = calm()
            results[f"ts_decay_linear({e}, 10)"] = calm()
        brain = SlowBrain(results)
        with patch.object(al, "MAX_BLENDS_PER_ROUND", 3):
            rows = run_queue(brain, fast, parallel=1)
        assert sum(r["rationale"].startswith("Auto-blend") for r in rows) == 3   # 2 for the first, 1 left for the next
    in_tmp(tmp_path, run)


def test_an_earlier_near_miss_too_similar_to_your_alphas_is_not_blended(tmp_path):
    def run():
        save_partners()
        al.append_result(saved_row(REV, 1.99, 0.5546, fitness=0.99, big="1.09", limit="0.86", alpha_id="rev"))
        brain = SlowBrain({}, corr=0.9, existing={"rev": REV})
        with patch("time.sleep"), captured_logs() as lines:
            assert al.follow_up_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", set()) == []
        assert brain.started == [] and any("too similar to your submitted alphas" in line for line in lines)
    in_tmp(tmp_path, run)

def test_no_blend_when_it_cant_help(tmp_path):
    def run():
        save_partners()
        cases = {
            "too similar already": (fast_json(), 0.88),
            "small stocks only": (fast_json(checks=[{"name": "LOW_FITNESS", "result": "FAIL"}, {
                "name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.4, "limit": 0.86}]), 0.3),
            "another check failed": (fast_json(checks=[{"name": "LOW_FITNESS", "result": "FAIL"},
                                                       {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}]), 0.3),
            "fitness is fine": (fast_json(fitness=1.2, checks=QUALIFYING_CHECKS), 0.3),
        }
        for why, (result, corr) in cases.items():
            brain = SlowBrain({REV: result, SMOOTHED_REV: calm()}, corr=corr)
            with patch.object(al, "PORTFOLIO_CHECK", False), patch.object(al, "submit_alpha", return_value=("YES", None)), \
                 patch.object(al, "is_active_on_brain", return_value=True):
                run_queue(brain, [REV], parallel=1)
            assert not any(e.startswith("rank(") for e in brain.started), why
            Path(al.RESULTS_FILE).unlink()
            save_partners()
        Path(al.RESULTS_FILE).unlink()                              # no slow alpha to add: nothing to blend
        brain = SlowBrain({REV: fast_json(), SMOOTHED_REV: calm()})
        run_queue(brain, [REV], parallel=1)
        assert brain.started == [REV, SMOOTHED_REV]
    in_tmp(tmp_path, run)


def test_an_earlier_near_miss_is_blended_at_the_start_of_a_run(tmp_path):
    def run():
        _write_operators()
        save_partners()
        al.append_result(saved_row(REV, 1.99, 0.5546, fitness=0.99, big="1.09", limit="0.86", alpha_id="rev",
                                   passed="6"))
        brain = SlowBrain({BLEND_EBIT: calm(), BLEND_UNSYS: calm(), "rank(close)": calm()}, existing={"rev": REV})
        gen = MagicMock(return_value=[("rank(close)", "r")])
        with patch.object(al, "MAX_ROUNDS", 1), patch.object(al, "generate_with_fallback", gen), \
             patch.object(al, "get_authenticated_session", return_value=brain.session), \
             patch("google.genai.Client"), patch("time.sleep"):
            al.main([])
        assert brain.started[:3] == [BLEND_EBIT, BLEND_UNSYS, "rank(close)"]    # before Gemini's ideas
        assert {r["expression"]: r["round"] for r in csv_rows()}[BLEND_EBIT] == "follow-up"
        # Next run: both blends were tried, so it's left alone - BRAIN isn't even asked about it.
        brain.session.get.reset_mock()
        with patch("time.sleep"):
            assert al.follow_up_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", al.already_tried()) == []
        assert not brain.session.get.called
    in_tmp(tmp_path, run)


def test_an_earlier_near_miss_without_big_stocks_numbers_gets_them_read_once(tmp_path):
    def run():
        save_partners()
        al.append_result(saved_row(REV, 1.99, 0.5546, fitness=0.99, big="", limit="", alpha_id="rev"))
        failing = fast_json(checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.4, "limit": 0.86}])
        brain = SlowBrain({REV: failing}, existing={"rev": REV})
        with patch("time.sleep"), captured_logs() as lines:
            assert al.follow_up_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", set()) == []
        assert any("fails the big-stocks check (0.40 vs limit 0.86) - not blending it" in line for line in lines)
        saved = {r["alpha_id"]: (r["sub_universe_sharpe"], r["sub_universe_limit"]) for r in csv_rows()}
        assert saved["rev"] == ("0.4", "0.86")
        brain.session.get.reset_mock()
        with patch("time.sleep"):
            al.follow_up_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", set())
        assert not brain.session.get.called                        # known now: not asked again, not blended
        # BRAIN has no such numbers: saved as "-", and the near-miss is blended.
        Path(al.RESULTS_FILE).unlink()
        save_partners()
        al.append_result(saved_row(REV, 1.99, 0.5546, fitness=0.99, big="", limit="", alpha_id="rev"))
        brain = SlowBrain({REV: fast_json(checks=[]), BLEND_EBIT: calm(), BLEND_UNSYS: calm()}, existing={"rev": REV})
        with patch("time.sleep"):
            rows = al.follow_up_past_near_misses(brain.session, SETTINGS, "e@x.com", "pw", set())
        assert [r["expression"] for r in rows] == [BLEND_EBIT, BLEND_UNSYS]
        assert {r["alpha_id"]: r["sub_universe_sharpe"] for r in csv_rows()}["rev"] == "-"
    in_tmp(tmp_path, run)


def test_blend_rows_pick_unsubmitted_originals_on_your_data_best_first(tmp_path):
    def run():
        write_dataset_cache(BLEND_DATASETS)
        rows = [saved_row(REV, 1.99, 0.55, fitness=0.99, alpha_id="a"),
                saved_row(REV.replace("10))", "15))"), 1.87, 0.46, fitness=0.99, alpha_id="b"),     # same idea
                saved_row("rank(ts_zscore(unsystematic_risk_last_90_days, 20))", 2.1, 0.5, fitness=0.9, alpha_id="c"),
                saved_row("rank(implied_volatility_put_90)", 2.2, 0.5, fitness=0.9, alpha_id="d"),
                saved_row("rank(snt_buzz)", 2.3, 0.5, fitness=0.9, alpha_id="e", submitted="YES"),
                saved_row(BLEND_EBIT, 2.4, 0.4, fitness=0.95, alpha_id="f", rationale="Auto-blend of x: y."),
                saved_row("rank(ts_delta(close, 3))", 1.6, 0.5, fitness=0.9, alpha_id="g", big="0.3", limit="0.7")]
        for r in rows:
            al.append_result(r)
        assert [r["alpha_id"] for r in al.blend_rows()] == ["d", "c", "a"]
        Path(al.OPERATORS_FILE).write_text(json.dumps({"operators": [], "fields": [
            {"name": "unsystematic_risk_last_90_days", "description": "d"}]}))
        assert [r["alpha_id"] for r in al.blend_rows()] == ["c", "a"]           # options left off the list
    in_tmp(tmp_path, run)


def test_fields_whose_good_scores_all_failed_the_big_stocks_check_are_dead_ends(tmp_path):
    def run():
        for i in range(8):
            al.append_result(saved_row(f"rank(ts_delta(snt_buzz, {i + 1}))", 0.3, 0.1, fitness=0.2, big="", limit=""))
            al.append_result(saved_row(f"rank(ts_delta(sales_est, {i + 1}))", 0.3, 0.1, fitness=0.2, big="", limit=""))
        al.append_result(saved_row("rank(snt_buzz)", 1.8, 0.04, fitness=2.2, big="-0.6", limit="0.78"))  # small stocks
        al.append_result(saved_row("rank(sales_est)", 1.3, 0.04, fitness=0.9, big="0.6", limit="0.5"))   # big ones too
        dead = al.dead_end_fields()
        assert dead == {"snt_buzz": (9, 0.2)} and "sales_est" not in dead
        al.append_result(saved_row("rank(only_small)", 1.8, 0.04, fitness=2.2, big="-0.6", limit="0.78"))
        with patch.object(al, "DEAD_END_MIN_TRIES", 1):
            assert al.dead_end_fields()["only_small"] == (1, None)
            note = al.avoid_note(set(), al.dead_end_fields())
        assert "never reached fitness 0.5 outside small stocks" in note and "only_small (1 tries, only small-stock " \
                                                                            "results)" in note
    in_tmp(tmp_path, run)


def test_the_generator_is_shown_what_works_on_your_account(tmp_path):
    def run():
        write_dataset_cache({"anl4_ebit_value": "analyst4", "snt_buzz": "socialmedia12", "rel_ret_all": "pv13"})
        for i in range(10):
            submitted = {0: "YES", 1: "TOO_CORRELATED (max 0.8)"}.get(i, "NO")
            al.append_result(saved_row(f"rank(ts_delta(anl4_ebit_value, {i + 1}))", 1.0, 0.1, submitted=submitted))
            al.append_result(saved_row(f"rank(ts_delta(snt_buzz, {i + 1}))", 0.3, 0.1))
            al.append_result(saved_row(f"rank(ts_delta(close, {i + 1}))", 0.3, 0.1))
        al.append_result(saved_row("rank(snt_buzz)", 0.3, 0.1))
        for i in range(3):                                                   # too few tests to score
            al.append_result(saved_row(f"rank(ts_delta(rel_ret_all, {i + 1}))", 1.5, 0.1, submitted="YES"))
        card = al.data_scorecard()
        assert card.splitlines() == [
            "Data that has produced alphas passing every check here (passed / tested):",
            "- ANALYST4 (analyst4): 2 of 10 (e.g. anl4_ebit_value)",
            "Data that never has, after 10+ tests - only use it with a clear reason it would also work in large stocks:",
            "- SOCIALMEDIA12 (socialmedia12): 0 of 11 (e.g. snt_buzz)",
            "- price/volume data only: 0 of 10"]
        client = MagicMock()
        client.models.generate_content.return_value = fake_gemini_response(ONE_IDEA)
        brain = SlowBrain({"rank(close)": calm()})
        with patch("time.sleep"):
            al.run_round(brain.session, client, [], [{"name": "close", "description": "d"}], SETTINGS,
                         "e@x.com", "pw", 1, "round 1", set())
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        assert "WHAT HAS WORKED ON THIS ACCOUNT" in prompt and "ts_delta(x, 90) / ts_std_dev(x, 180)" in prompt
        assert "bucket(rank(cap), range='0, 1, 0.1')" in prompt and "rank(fast) + rank(slow)" in prompt
        assert "- ANALYST4 (analyst4): 2 of 10" in prompt
        assert "re-tested added to your best slow alphas" in prompt
        # With a field list, only its datasets are scored (price/volume always is).
        Path(al.OPERATORS_FILE).write_text(json.dumps({"operators": [], "fields": [
            {"name": "anl4_ebit_value", "description": "d"}]}))
        assert "SOCIALMEDIA12" not in al.data_scorecard() and "ANALYST4" in al.data_scorecard()
        assert "price/volume data only" in al.data_scorecard()
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
