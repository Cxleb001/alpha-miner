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
        if url.startswith("sim-"):
            return fake_response(200, {"status": "COMPLETE", "alpha": url[4:]})
        if url.endswith("/correlations/self"):
            return fake_response(200, {"max": self.corr})
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
        with patch("time.sleep"):
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