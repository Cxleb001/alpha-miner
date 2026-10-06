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
os.environ.setdefault("WQ_EMAIL", "e@x.com")
os.environ.setdefault("WQ_PASSWORD", "pw")

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