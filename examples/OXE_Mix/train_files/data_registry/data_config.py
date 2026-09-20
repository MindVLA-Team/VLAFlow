"""Full OXEMix (original OXE, OXE_AugE_v2, RoboCOIN_SUB) and original OXE recipes.

All sources live under DATA_ROOT. Original OXE includes DROID; WM variants use
future-frame targets with the same dataset composition.
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # examples/OXE_Mix/train_files/data_registry/


def _load_sibling(bench: str):
    """Load examples/<bench>/train_files/data_registry/data_config.py by file path."""
    path = _HERE.parents[2] / bench / "train_files" / "data_registry" / "data_config.py"
    mod_name = f"_oxemix_dep_{bench.replace('-', '_')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _prefix(entries: list, subdir: str) -> list:
    """Prepend *subdir* to every dataset name in a mixture entry list.

    Each entry is a 3-tuple: (dataset_name, weight, robot_type).
    """
    return [(f"{subdir}/{name}", w, rtype) for name, w, rtype in entries]


_oxe = _load_sibling("OXE")
_auge = _load_sibling("OXE_AugE")
_robocoin = _load_sibling("RoboCOIN")

# ── Merged registries ───────────────────────────────────────────────────────

ROBOT_TYPE_CONFIG_MAP = {
    **_oxe.ROBOT_TYPE_CONFIG_MAP,
    **_auge.ROBOT_TYPE_CONFIG_MAP,
    **_robocoin.ROBOT_TYPE_CONFIG_MAP,
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    **_oxe.ROBOT_TYPE_TO_EMBODIMENT_TAG,
    **_auge.ROBOT_TYPE_TO_EMBODIMENT_TAG,
    **_robocoin.ROBOT_TYPE_TO_EMBODIMENT_TAG,
}

# ── Source sub-lists (with subdirectory prefix) ──────────────────────────────
#
# Each source has its own subdirectory under the common data_root_dir.
# Prepend the appropriate subdir so paths resolve correctly.

# OXE original: Bridge + RT-1


_OXE_ALL = _prefix(
    _oxe.DATASET_NAMED_MIXTURES.get("oxe_no_lang", []),
    "openx-embodiment-lerobot",
)

# OXE_AugE: full mix (all datasets × 8 robots)
# NOTE: points to OXE_AugE_v2 — v3.0 datasets converted to v2.1 (H.264, per-episode parquets)
_OXE_AUGE_ALL = _prefix(
    _auge.DATASET_NAMED_MIXTURES.get("oxeauge_all", []),
    "OXE_AugE_v2",
)

# RoboCOIN: all 121 datasets
_ROBOCOIN_ALL = _prefix(
    _robocoin.DATASET_NAMED_MIXTURES.get("robocoin_all", []),
    "RoboCOIN_SUB",
)

# ── Named mixtures ──────────────────────────────────────────────────────────

DATASET_NAMED_MIXTURES: dict = {"oxe_mix_full": [*_OXE_ALL, *_OXE_AUGE_ALL, *_ROBOCOIN_ALL], "oxe_original": _OXE_ALL}


# ===========================================================================
# World Model (MindWPI) Mixtures
# ===========================================================================
# WM variants use the "_wm" suffixed robot types defined in each sub-config.
# These robot types inherit all transforms/modality from the base but add
# compute_action_loss and latent_view_key attributes for the WM pipeline.

# OXE original WM: Bridge + RT-1 (WM variants)


_OXE_ALL_WM = _prefix(
    _oxe.DATASET_NAMED_MIXTURES.get("oxe_all_wm", []),
    "openx-embodiment-lerobot",
)

# OXE_AugE WM: full mix (all datasets × 8 robots, WM variants)
_OXE_AUGE_ALL_WM = _prefix(
    _auge.DATASET_NAMED_MIXTURES.get("oxeauge_all_wm", []),
    "OXE_AugE_v2",
)

# RoboCOIN WM: all datasets (WM variants)
_ROBOCOIN_ALL_WM = _prefix(
    _robocoin.DATASET_NAMED_MIXTURES.get("robocoin_all_wm", []),
    "RoboCOIN_SUB",
)

# WM named mixtures
DATASET_NAMED_MIXTURES["oxe_mix_full_wm"] = [
    *_OXE_ALL_WM,
    *_OXE_AUGE_ALL_WM,
    *_ROBOCOIN_ALL_WM,
]


DATASET_NAMED_MIXTURES["oxe_original_wm"] = _OXE_ALL_WM
