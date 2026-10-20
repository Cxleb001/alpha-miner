"""Tests for fetch_fields.py. Run: python test_fetch_fields.py  (no network - BRAIN is mocked)."""
import json
import os
import shutil
import sys
import tempfile
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
import fetch_fields as ff


def field(fid, users=0, alphas=0, coverage=1.0, ftype="MATRIX", desc="d", dataset="pv1"):
    # Same shape as a real captured BRAIN /data-fields result.
    return {"id": fid, "region": "USA", "universe": "TOP3000", "delay": 1, "type": ftype,
            "category": {"id": "pv", "name": "Price Volume"}, "dataset": {"id": dataset, "name": dataset.upper()},
            "description": desc, "coverage": coverage, "alphaCount": alphas, "userCount": users}


def reply(status, payload=None, headers=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.text = text or json.dumps(payload or {})
    r.json.return_value = payload if payload is not None else {}
    return r


def paged_session(all_fields, page_size=50):
    session = MagicMock()

    def get(url, params=None, timeout=None):
        matching = [f for f in all_fields if f["type"] == params["type"]]
        start = params["offset"]
        return reply(200, {"count": len(matching), "results": matching[start:start + page_size]})

    session.get.side_effect = get
    return session


def test_fetch_all_pages_through_everything_with_the_right_params():
    fields = [field(f"f{i}") for i in range(120)]
    session = paged_session(fields)
    with patch("time.sleep"):
        got = ff.fetch_all(session, "MATRIX")
    assert [f["id"] for f in got] == [f"f{i}" for i in range(120)]
    calls = session.get.call_args_list
    assert [c.kwargs["params"]["offset"] for c in calls] == [0, 50, 100]
    p = calls[0].kwargs["params"]
    assert (p["instrumentType"], p["region"], p["universe"], p["delay"], p["type"], p["limit"]) == \
           ("EQUITY", "USA", "TOP3000", 1, "MATRIX", 50)
    assert calls[0].args[0] == "https://api.worldquantbrain.com/data-fields"


def test_fetch_all_drops_other_types_if_brain_ignores_the_filter():
    session = MagicMock()
    session.get.return_value = reply(200, {"count": 2, "results": [field("close"), field("vec1", ftype="VECTOR")]})
    assert [f["id"] for f in ff.fetch_all(session, "MATRIX")] == ["close"]


def test_rate_limit_waits_the_float_retry_after():
    session = MagicMock()
    session.get.side_effect = [reply(429, {"detail": "slow down"}, headers={"Retry-After": "2.5"}),
                               reply(200, {"count": 1, "results": [field("close")]})]
    with patch("time.sleep") as sleep:
        data = ff.fetch_page(session, "MATRIX", 0)
    sleep.assert_called_once_with(2.5)
    assert data["results"][0]["id"] == "close"


def test_brain_error_message_is_shown_verbatim():
    session = MagicMock()
    session.get.return_value = reply(400, {"region": ["This field is required."]},
                                     text='{"region": ["This field is required."]}')
    try:
        ff.fetch_page(session, "MATRIX", 0)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "This field is required" in str(e) and "400" in str(e)


def test_expired_session_raises_permission_error():
    session = MagicMock()
    session.get.return_value = reply(401, {"detail": "Incorrect authentication credentials."})
    try:
        ff.fetch_page(session, "MATRIX", 0)
        assert False, "should have raised"
    except PermissionError:
        pass


def test_select_fields_ranks_by_usage_filters_coverage_and_keeps_basics():
    matrix = [field("popular", users=900), field("thin", users=999, coverage=0.3),
              field("middling", users=50), field("close", users=1), field("volume", users=2)]
    groups = [field("industry", users=500, ftype="GROUP"), field("sector", users=400, ftype="GROUP")]
    entries = ff.select_fields(matrix, groups, top=2, min_coverage=0.8)
    names = [e["name"] for e in entries]
    assert names[:2] == ["popular", "middling"]          # most used first, low coverage skipped
    assert "thin" not in names
    assert "close" in names and "volume" in names         # basics kept despite low rank
    assert names[-2:] == ["industry", "sector"]           # group fields appended
    assert entries[-1]["description"].startswith("group field")
    assert entries[0]["description"] == "d [PV1]"


def test_select_fields_can_skip_the_most_used_ones():
    matrix = [field(f"rank{i}", users=100 - i) for i in range(1, 7)] + [field("close", users=500)]
    names = [e["name"] for e in ff.select_fields(matrix, [], top=2, min_coverage=0.8, skip=3)]
    assert names == ["rank3", "rank4", "close"]           # close (most used) and rank1-2 skipped; basics kept
    everyone = [e["name"] for e in ff.select_fields(matrix, [], top=3, min_coverage=0.8)]
    assert not set(everyone) - {"close"} & set(names)     # no signal field in both selections


def test_least_used_takes_the_rarest_fields_a_few_per_dataset():
    matrix = ([field(f"a{i}", users=10 + i, dataset="analyst4") for i in range(6)]       # the rarest, all one dataset
              + [field(f"o{i}", users=200 + i, dataset="option9") for i in range(3)]
              + [field("m0", users=900, dataset="model16"), field("thin", users=11, coverage=0.3, dataset="news18")]
              + [field("close", users=60000), field("volume", users=50000)])
    groups = [field("industry", users=500, ftype="GROUP"), field("sector", users=400, ftype="GROUP")]
    names = [e["name"] for e in ff.select_fields(matrix, groups, top=5, min_coverage=0.8, least_used=True,
                                                 per_dataset=2)]
    assert names[:5] == ["a0", "a1", "o0", "o1", "m0"]    # rarest first, but no more than 2 from one dataset
    assert "thin" not in names                             # low coverage still skipped
    assert "close" in names and "volume" in names          # basics kept
    assert names[-2:] == ["industry", "sector"]            # group fields: still the usual most-used ones
    plain = [e["name"] for e in ff.select_fields(matrix, [], top=5, min_coverage=0.8, least_used=True)]
    assert plain[:5] == ["a0", "a1", "a2", "a3", "a4"]     # per_dataset 0: no limit


def test_least_used_tops_up_from_held_back_fields_when_datasets_run_out():
    matrix = [field(f"a{i}", users=i + 1, dataset="analyst4") for i in range(6)] + [field("o0", users=50, dataset="option9")]
    names = [e["name"] for e in ff.select_fields(matrix, [], top=5, min_coverage=0.8, least_used=True, per_dataset=2)]
    assert names[:5] == ["a0", "a1", "o0", "a2", "a3"]     # 2 + 1, then the next-rarest held back, in order


def test_least_used_skips_fields_almost_nobody_uses_and_prefers_coverage():
    matrix = [field("one_user", users=1), field("nine", users=9), field("ten_a", users=10, alphas=5, coverage=0.85),
              field("ten_b", users=10, alphas=5, coverage=0.99), field("busy", users=5000)]
    names = [e["name"] for e in ff.select_fields(matrix, [], top=3, min_coverage=0.8, least_used=True, min_users=10)]
    assert names[:3] == ["ten_b", "ten_a", "busy"]         # under 10 users left out; equal rarity: better coverage first


def test_main_least_used_writes_the_rarest_says_so_and_compares_usage():
    fields = ([field(f"m{i}", users=1000 + i, dataset=f"ds{i % 3}") for i in range(30)]
              + [field(f"r{i}", users=10 + i, dataset=f"ds{i % 4}") for i in range(12)]
              + [field("tiny", users=2, dataset="ds9"), field("close", users=60000), field("industry", ftype="GROUP")])
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session(fields)), \
                 patch("time.sleep"), patch.object(ff.logger, "info") as info:
                ff.main(["--top", "5"])                   # the usual most-used list first...
                ff.main(["--top", "5", "--least-used"])   # ...then the least-used one
            ref = json.loads(Path("operators.json").read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd)
    names = [f["name"] for f in ref["fields"]]
    assert names[:5] == ["r0", "r1", "r2", "r3", "r4"]    # 10-14 users; "tiny" (2 users) left out
    assert "close" in names and "industry" in names
    note = ref["_fields_note"]
    assert "5 least-used fields" in note and "at most 8 from any one dataset" in note and "at least 10 people" in note
    said = [c.args[0] % c.args[1:] for c in info.call_args_list]
    # the most-used five were close (a basic, left out of the comparison) and m26-m29
    assert any("10-14 people have used each" in s and "previous list: 1,026-1,029" in s for s in said), said


def test_the_usual_most_used_list_has_no_per_dataset_limit():
    fields = [field(f"m{i}", users=100 + i) for i in range(20)] + [field("close")]   # all one dataset
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session(fields)), \
                 patch("time.sleep"):
                ff.main(["--top", "12"])
            ref = json.loads(Path("operators.json").read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd)
    assert [f["name"] for f in ref["fields"]][:12] == [f"m{i}" for i in range(19, 7, -1)]   # 12 from one dataset
    assert "at most" not in ref["_fields_note"] and "least-used" not in ref["_fields_note"]


def test_main_with_skip_takes_the_next_fields_and_says_so():
    fields = [field(f"m{i}", users=i) for i in range(60)] + [field("close"), field("industry", ftype="GROUP")]
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session(fields)), \
                 patch("time.sleep"):
                ff.main(["--top", "5", "--skip", "5"])
            ref = json.loads(Path("operators.json").read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd)
    names = [f["name"] for f in ref["fields"]]
    assert names[:5] == ["m54", "m53", "m52", "m51", "m50"]   # ranks 6-10, not 1-5
    assert "close" in names and "industry" in names
    assert "ranked 6-10" in ref["_fields_note"]


def test_skipping_past_every_field_leaves_operators_alone():
    fields = [field(f"m{i}", users=i) for i in range(30)] + [field("close")]
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        before = (Path(d) / "operators.json").read_text(encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session(fields)), \
                 patch("time.sleep"):
                try:
                    ff.main(["--skip", "80"])
                    assert False, "should have exited"
                except SystemExit as e:
                    assert e.code == 1
            assert Path("operators.json").read_text(encoding="utf-8") == before
        finally:
            os.chdir(cwd)


def test_write_operators_keeps_operators_and_backs_up():
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "operators.json"
        shutil.copyfile(HERE / "operators.json", target)
        before = json.loads(target.read_text(encoding="utf-8"))
        ff.write_operators([{"name": "close", "description": "c"}], top=80, min_coverage=0.8, path=target)
        after = json.loads(target.read_text(encoding="utf-8"))
        assert after["operators"] == before["operators"]                      # untouched
        assert after["fields"] == [{"name": "close", "description": "c"}]
        assert "fetch_fields.py" in after["_fields_note"]
        assert json.loads((Path(d) / "operators.json.bak").read_text(encoding="utf-8")) == before


def test_main_end_to_end_updates_operators_and_cache():
    fields = [field(f"m{i}", users=i) for i in range(60)] + [field("close"), field("industry", ftype="GROUP")]
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session(fields)), \
                 patch("time.sleep"):
                ff.main(["--top", "5"])
            ref = json.loads(Path("operators.json").read_text(encoding="utf-8"))
            cache = json.loads(Path(ff.CACHE_FILE).read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd)
    names = [f["name"] for f in ref["fields"]]
    assert names[:5] == ["m59", "m58", "m57", "m56", "m55"]
    assert "close" in names and "industry" in names
    assert len(ref["operators"]) == 66
    assert len(cache["matrix_fields"]) == 61 and len(cache["group_fields"]) == 1


def test_main_leaves_operators_alone_when_brain_returns_nothing():
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(HERE / "operators.json", Path(d) / "operators.json")
        before = (Path(d) / "operators.json").read_text(encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with patch.object(al, "get_authenticated_session", return_value=paged_session([])), \
                 patch("time.sleep"):
                try:
                    ff.main([])
                    assert False, "should have exited"
                except SystemExit as e:
                    assert e.code == 1
            assert Path("operators.json").read_text(encoding="utf-8") == before
            assert not Path("operators.json.bak").exists()
        finally:
            os.chdir(cwd)


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
