"""Assert golden digests (bit-compat net)."""
from __future__ import annotations

import json
import os

import pytest

from tests.golden.data import make_digits_10k, make_swiss_cone_2k
from tests.golden.generate import EXPECTED_PATH, SEED, build_fixture_graph, load_expected


def _maybe_write(name: str, digests: dict) -> dict:
    expected = load_expected()
    if os.environ.get("LEANMAP_GOLDEN_WRITE") == "1":
        expected[name] = digests
        EXPECTED_PATH.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return expected
    return expected


@pytest.mark.skipif(
    not EXPECTED_PATH.exists() and os.environ.get("LEANMAP_GOLDEN_WRITE") != "1",
    reason="golden expected.json missing; LEANMAP_GOLDEN_WRITE=1 to create",
)
def test_golden_swiss_cone_2k():
    X, _ = make_swiss_cone_2k(SEED)
    digests = build_fixture_graph("swiss_cone_2k", X)
    _maybe_write("swiss_cone_2k", digests)
    exp = load_expected()["swiss_cone_2k"]
    for k, v in exp.items():
        assert digests[k] == v, f"mismatch {k}: {digests.get(k)!r} != {v!r}"


@pytest.mark.skipif(
    not EXPECTED_PATH.exists() and os.environ.get("LEANMAP_GOLDEN_WRITE") != "1",
    reason="golden expected.json missing; LEANMAP_GOLDEN_WRITE=1 to create",
)
def test_golden_digits_10k():
    X, _ = make_digits_10k(SEED)
    digests = build_fixture_graph("digits_10k", X)
    _maybe_write("digits_10k", digests)
    exp = load_expected()["digits_10k"]
    for k, v in exp.items():
        assert digests[k] == v, f"mismatch {k}: {digests.get(k)!r} != {v!r}"
