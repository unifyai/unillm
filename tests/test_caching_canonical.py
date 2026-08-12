"""Tests for canonical cache keying."""

import json
import os
import tempfile

import pytest

from unillm.caching import (
    CacheMissError,
    LocalSeparateCache,
    _get_cache,
    _write_to_cache,
    canonical_digest,
    canonical_digest_of_raw_key,
    parse_raw_key,
)
from unillm.caching.ndjson_index import NdjsonIndexedStore

from .test_caching import _CacheHandler


def _kw(system: str, user: str = "Hi", **extra) -> dict:
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **extra,
    }


class TestCanonicalDigest:
    """The digest is invariant to mundane churn and sensitive to contracts."""

    def test_identical_requests_share_a_digest(self):
        assert canonical_digest("fn", _kw("Be helpful.")) == canonical_digest(
            "fn",
            _kw("Be helpful."),
        )

    def test_block_reordering_is_invariant(self):
        ordered = _kw("# Rules\nBe kind.\n\n# Context\nIt is Tuesday.")
        reordered = _kw("# Context\nIt is Tuesday.\n\n# Rules\nBe kind.")
        assert canonical_digest("fn", ordered) == canonical_digest("fn", reordered)

    def test_block_rewording_changes_the_digest(self):
        assert canonical_digest(
            "fn",
            _kw("# Rules\nBe kind."),
        ) != canonical_digest("fn", _kw("# Rules\nBe extremely kind."))

    def test_assistant_content_order_is_significant(self):
        def with_assistant(content):
            kw = _kw("sys")
            kw["messages"].append({"role": "assistant", "content": content})
            return kw

        assert canonical_digest(
            "fn",
            with_assistant("First.\n\nSecond."),
        ) != canonical_digest("fn", with_assistant("Second.\n\nFirst."))

    def test_whitespace_is_invariant(self):
        assert canonical_digest(
            "fn",
            _kw("Be helpful.  \r\nAlways.\n\n\n\nReally."),
        ) == canonical_digest("fn", _kw("Be helpful.\nAlways.\n\nReally."))

    def test_message_order_is_significant(self):
        forward = _kw("sys", user="one")
        swapped = {
            "model": "gpt-4",
            "messages": list(reversed(forward["messages"])),
        }
        assert canonical_digest("fn", forward) != canonical_digest("fn", swapped)

    def test_volatile_timestamps_are_scrubbed(self):
        assert canonical_digest(
            "fn",
            _kw("Current time: 2026-08-10T12:31:15Z."),
        ) == canonical_digest("fn", _kw("Current time: 2026-08-04T18:08:08Z."))

    def test_uuids_and_tmp_paths_are_scrubbed(self):
        assert canonical_digest(
            "fn",
            _kw(
                "Session 123e4567-e89b-12d3-a456-426614174000 "
                "logs to /tmp/run-a/out.txt.",
            ),
        ) == canonical_digest(
            "fn",
            _kw(
                "Session 00000000-0000-0000-0000-000000000000 "
                "logs to /tmp/run-b/out.txt.",
            ),
        )

    def test_fixture_dates_are_not_scrubbed(self):
        assert canonical_digest(
            "fn",
            _kw("Alice joined on 2020-01-01."),
        ) != canonical_digest("fn", _kw("Alice joined on 2021-06-30."))

    def test_tool_description_rewording_is_invariant(self):
        def with_tool(description):
            return _kw(
                "sys",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "q": {
                                        "type": "string",
                                        "description": description,
                                    },
                                },
                                "required": ["q"],
                            },
                        },
                    },
                ],
            )

        assert canonical_digest("fn", with_tool("Look things up.")) == (
            canonical_digest("fn", with_tool("Searches for things."))
        )

    def test_tool_parameter_change_alters_the_digest(self):
        def with_param_type(param_type):
            return _kw(
                "sys",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": param_type}},
                            },
                        },
                    },
                ],
            )

        assert canonical_digest("fn", with_param_type("string")) != (
            canonical_digest("fn", with_param_type("integer"))
        )

    def test_tool_order_is_invariant(self):
        def tool(name):
            return {"type": "function", "function": {"name": name}}

        assert canonical_digest(
            "fn",
            _kw("sys", tools=[tool("a"), tool("b")]),
        ) == canonical_digest("fn", _kw("sys", tools=[tool("b"), tool("a")]))

    def test_property_named_description_is_kept(self):
        def with_required(required):
            return _kw(
                "sys",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "save",
                            "parameters": {
                                "type": "object",
                                "properties": {"description": {"type": "string"}},
                                "required": required,
                            },
                        },
                    },
                ],
            )

        assert canonical_digest("fn", with_required(["description"])) != (
            canonical_digest("fn", with_required([]))
        )

    def test_stream_options_are_dropped(self):
        assert canonical_digest(
            "fn",
            _kw("sys", stream=False, stream_options={"include_usage": True}),
        ) == canonical_digest("fn", _kw("sys", stream=False))

    def test_model_change_alters_the_digest(self):
        one = _kw("sys")
        other = dict(one, model="gpt-5")
        assert canonical_digest("fn", one) != canonical_digest("fn", other)

    def test_version_prefix_present(self):
        assert canonical_digest("fn", _kw("sys")).startswith("canon-v")


class TestParseRawKey:
    def test_roundtrip(self):
        kw = _kw("sys")
        raw = f"chat.completions.create_{json.dumps(kw)}"
        parsed = parse_raw_key(raw)
        assert parsed == ("chat.completions.create", kw)

    def test_unparseable_key_has_no_digest(self):
        assert canonical_digest_of_raw_key("not-a-request-key") is None
        assert canonical_digest_of_raw_key("fn_{broken json") is None

    def test_raw_key_digest_matches_live_digest(self):
        kw = _kw("sys")
        raw = f"fn_{json.dumps(kw)}"
        assert canonical_digest_of_raw_key(raw) == canonical_digest("fn", kw)


class TestIndexCanonicalLookup:
    def _store(self, tmp) -> NdjsonIndexedStore:
        store = NdjsonIndexedStore(os.path.join(tmp, ".cache.ndjson"))
        store.open_or_create()
        return store

    def test_canonical_hit_after_block_reorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            recorded = _kw("# A\nfirst.\n\n# B\nsecond.")
            store.append(f"fn_{json.dumps(recorded)}", '"answer"')

            drifted = _kw("# B\nsecond.\n\n# A\nfirst.")
            digest = canonical_digest("fn", drifted)
            result = store.get_canonical(digest)
            assert result is not None
            value, res_types, matched_key = result
            assert value == "answer"
            assert matched_key == f"fn_{json.dumps(recorded)}"

    def test_canonical_miss_for_different_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.append(f"fn_{json.dumps(_kw('sys'))}", '"answer"')
            digest = canonical_digest("fn", _kw("entirely different prompt"))
            assert store.get_canonical(digest) is None

    def test_newest_recording_wins_on_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            old = _kw("Time: 2026-08-01T00:00:00Z.")
            new = _kw("Time: 2026-08-02T00:00:00Z.")
            store.append(f"fn_{json.dumps(old)}", '"old"')
            store.append(f"fn_{json.dumps(new)}", '"new"')
            digest = canonical_digest("fn", _kw("Time: 2026-08-03T00:00:00Z."))
            value, _, _ = store.get_canonical(digest)
            assert value == "new"

    def test_index_rebuild_recovers_canonical_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            recorded = _kw("# A\nfirst.\n\n# B\nsecond.")
            store.append(f"fn_{json.dumps(recorded)}", '"answer"')
            os.remove(store.idx_path)

            fresh = self._store(tmp)
            digest = canonical_digest("fn", _kw("# B\nsecond.\n\n# A\nfirst."))
            result = fresh.get_canonical(digest)
            assert result is not None
            assert result[0] == "answer"

    def test_stale_sidecar_without_canon_map_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.append(f"fn_{json.dumps(_kw('sys'))}", '"answer"')
            with open(store.idx_path) as f:
                sidecar = json.load(f)
            del sidecar["canon"]
            with open(store.idx_path, "w") as f:
                json.dump(sidecar, f)

            fresh = self._store(tmp)
            assert fresh.get_canonical(canonical_digest("fn", _kw("sys"))) is not None


class TestGetCacheKeying:
    def test_exact_keying_misses_on_drift(self):
        with _CacheHandler():
            _write_to_cache("fn", _kw("# A\nx.\n\n# B\ny."), "answer")
            result, hit_kind = _get_cache(
                fn_name="fn",
                kw=_kw("# B\ny.\n\n# A\nx."),
                keying="exact",
            )
            assert result is None
            assert hit_kind is None

    def test_canonical_keying_hits_on_drift(self):
        with _CacheHandler():
            _write_to_cache("fn", _kw("# A\nx.\n\n# B\ny."), "answer")
            result, hit_kind = _get_cache(
                fn_name="fn",
                kw=_kw("# B\ny.\n\n# A\nx."),
                keying="canonical",
            )
            assert result == "answer"
            assert hit_kind == "canonical"

    def test_exact_match_wins_over_canonical(self):
        with _CacheHandler():
            drifted = _kw("# B\ny.\n\n# A\nx.")
            _write_to_cache("fn", _kw("# A\nx.\n\n# B\ny."), "recorded-first")
            _write_to_cache("fn", drifted, "recorded-exact")
            result, hit_kind = _get_cache(fn_name="fn", kw=drifted, keying="canonical")
            assert result == "recorded-exact"
            assert hit_kind == "exact"

    def test_canonical_read_only_raises_on_genuine_miss(self):
        with _CacheHandler():
            _write_to_cache("fn", _kw("recorded prompt"), "answer")
            with pytest.raises(CacheMissError) as exc_info:
                _get_cache(
                    fn_name="fn",
                    kw=_kw("a genuinely different prompt"),
                    raise_on_empty=True,
                    keying="canonical",
                )
            assert "Key was not found in the cache" in str(exc_info.value)

    def test_keying_defaults_to_settings(self, monkeypatch):
        from unillm.settings import SETTINGS

        with _CacheHandler():
            _write_to_cache("fn", _kw("# A\nx.\n\n# B\ny."), "answer")
            monkeypatch.setattr(SETTINGS, "UNILLM_CACHE_KEYING", "canonical")
            result, hit_kind = _get_cache(fn_name="fn", kw=_kw("# B\ny.\n\n# A\nx."))
            assert result == "answer"
            assert hit_kind == "canonical"
            monkeypatch.setattr(SETTINGS, "UNILLM_CACHE_KEYING", "exact")
            result, hit_kind = _get_cache(fn_name="fn", kw=_kw("# B\ny.\n\n# A\nx."))
            assert result is None

    def test_multi_turn_trajectory_stays_coherent(self):
        """Turn N's key embeds turn N-1's stored assistant text verbatim."""
        with _CacheHandler():
            turn_one = _kw("# A\nrules.\n\n# B\ncontext.")
            _write_to_cache("fn", turn_one, "the stored reply")

            drifted_turn_one = _kw("# B\ncontext.\n\n# A\nrules.")
            reply, hit_kind = _get_cache(
                fn_name="fn",
                kw=drifted_turn_one,
                keying="canonical",
            )
            assert hit_kind == "canonical"

            turn_two = _kw("# A\nrules.\n\n# B\ncontext.")
            turn_two["messages"].append({"role": "assistant", "content": reply})
            turn_two["messages"].append({"role": "user", "content": "and then?"})
            _write_to_cache("fn", turn_two, "the second reply")

            drifted_turn_two = _kw("# B\ncontext.\n\n# A\nrules.")
            drifted_turn_two["messages"].append(
                {"role": "assistant", "content": reply},
            )
            drifted_turn_two["messages"].append(
                {"role": "user", "content": "and then?"},
            )
            second, hit_kind = _get_cache(
                fn_name="fn",
                kw=drifted_turn_two,
                keying="canonical",
            )
            assert second == "the second reply"
            assert hit_kind == "canonical"


class TestLocalSeparateCanonical:
    def _isolate(self, tmp):
        LocalSeparateCache._read_store = None
        LocalSeparateCache._cache_write = None
        LocalSeparateCache._canon_write = None
        LocalSeparateCache._cache_dir = tmp
        LocalSeparateCache.initialize_cache()

    def test_own_writes_are_canonically_addressable(self):
        original_dir = LocalSeparateCache._cache_dir
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self._isolate(tmp)
                recorded = _kw("# A\nx.\n\n# B\ny.")
                LocalSeparateCache.store_entry(
                    key=f"fn_{json.dumps(recorded)}",
                    value='"answer"',
                    res_types=None,
                )
                digest = canonical_digest("fn", _kw("# B\ny.\n\n# A\nx."))
                value, _ = LocalSeparateCache.retrieve_canonical(digest)
                assert value == "answer"
        finally:
            LocalSeparateCache._read_store = None
            LocalSeparateCache._cache_write = None
            LocalSeparateCache._canon_write = None
            LocalSeparateCache._cache_dir = original_dir

    def test_read_store_canonical_hit_promotes_to_write_cache(self):
        original_dir = LocalSeparateCache._cache_dir
        try:
            with tempfile.TemporaryDirectory() as tmp:
                recorded_key = f"fn_{json.dumps(_kw('# A\nx.\n\n# B\ny.'))}"
                with open(os.path.join(tmp, ".cache.ndjson"), "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "key": recorded_key,
                                "value": '"answer"',
                                "res_types": None,
                            },
                        )
                        + "\n",
                    )
                self._isolate(tmp)
                digest = canonical_digest("fn", _kw("# B\ny.\n\n# A\nx."))
                value, _ = LocalSeparateCache.retrieve_canonical(digest)
                assert value == "answer"
                assert recorded_key in LocalSeparateCache._cache_write
        finally:
            LocalSeparateCache._read_store = None
            LocalSeparateCache._cache_write = None
            LocalSeparateCache._canon_write = None
            LocalSeparateCache._cache_dir = original_dir
