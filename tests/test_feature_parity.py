"""The two implementations of CONTRACT.md section 4 must stay identical.

Section 4 says the formulas are "shared in the group package so there is one code
path for training and serving". In this repository there are in fact two:

    mykrml_25740287.features.crypto   installed from TestPyPI, used by
                                      at3_crypto/features.py -> Student B
    at3_crypto/engineered_features    used by Student D, and vendored verbatim
                                      into the Student D FastAPI service

Student D's service cannot import the TestPyPI package at runtime and still be
independently deployable, which is why the second one exists. That is a
reasonable trade, but it turns the contract's "one code path" into a property
that has to be checked rather than one the code guarantees.

Nothing else in the pipeline would notice a drift. Both sides would keep
training, both services would keep answering, and the only symptom would be
Student B's model consuming a `Volatility20` computed slightly differently from
the one it was trained on - a silent accuracy loss, in production, attributable
to nothing in particular.

So this compares them on the committed snapshot, every contract feature, every
row, including where each one puts its NaNs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from at3_crypto.dataset import SNAPSHOT_PATH
from at3_crypto.engineered_features import FEATURE_OWNERS, engineer_features

package_features = pytest.importorskip(
    "mykrml_25740287.features.crypto",
    reason="the shared package is only installed via the TestPyPI index",
)

KEYS = ["token", "date"]


@pytest.fixture(scope="module")
def snapshot() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"no committed snapshot at {SNAPSHOT_PATH}")
    return pd.read_csv(SNAPSHOT_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def both(snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = engineer_features(snapshot).set_index(KEYS).sort_index()
    shared = package_features.compute_features_panel(snapshot).set_index(KEYS).sort_index()
    return local, shared


def test_both_implementations_publish_the_same_feature_names(both):
    """A feature missing from one side is a peer that cannot read it at all.

    The names are the JSON keys in `POST /features`, so this is the check that
    `VolumeSMA20` has not become `SMA20Volume` on one side only - the exact bug
    that made Student D's published feature unreadable once already.
    """
    local, shared = both
    contract = set(FEATURE_OWNERS)

    assert contract <= set(local.columns), sorted(contract - set(local.columns))
    assert contract <= set(shared.columns), sorted(contract - set(shared.columns))


@pytest.mark.parametrize("feature", sorted(FEATURE_OWNERS))
def test_every_contract_feature_agrees_exactly(both, feature):
    """Not "to within a tolerance": the same formula on the same rows.

    Both are pandas rolling windows over identical inputs, so anything other
    than an exact match means the two definitions have genuinely diverged, not
    that floating point drifted.
    """
    local, shared = both
    mine, theirs = local[feature], shared[feature]

    # Where a value exists on one side and not the other, the warm-up length or
    # the grouping differs - which changes how many rows each model can train on.
    only_one_side = mine.isna() ^ theirs.isna()
    assert not only_one_side.any(), (
        f"{feature}: {int(only_one_side.sum())} rows are NaN in one implementation "
        f"but not the other, first at {mine.index[only_one_side][0]}"
    )

    computed = ~mine.isna()
    assert (mine[computed] == theirs[computed]).all(), (
        f"{feature}: max |difference| = {(mine[computed] - theirs[computed]).abs().max():.3e}"
    )


def test_the_comparison_covers_the_whole_snapshot(both, snapshot):
    """Guard the test itself: an inner join on an empty overlap would pass silently."""
    local, shared = both

    assert len(local) == len(snapshot)
    assert local.index.equals(shared.index)
    assert snapshot["token"].nunique() == 8
