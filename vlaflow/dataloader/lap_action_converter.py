"""
LAP (Language Action Policy) Action-to-Language Converter.

Converts robot action arrays to language descriptions for use with
Language Action Policy (LAP) training. Supports multiple formats:

  - "vla0"    : Discretized integers, e.g. "512 487 523 500 501 499 0"
                Works with any normalized action in [-1, 1].
                Best for CoTrain where physical units are unknown.

  - "verbose" : Natural language, e.g. "move forward 5 cm, close gripper"
                Assumes actions are delta EEF values in SI units
                (meters for translation, radians for rotation).

  - "compact" : Signed cm integers, e.g. "<+05 -03 +02 0>"

References:
  - LAP: https://arxiv.org/abs/2406.06699
  - VLA-0: discretized action token format
"""

import re

import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  VLA0 Format  (works with any normalized action in [-1, 1])
# ──────────────────────────────────────────────────────────────────────


def actions_to_vla0(
    actions: np.ndarray,
    num_bins: int = 1000,
    action_horizon: int = None,
) -> str:
    """Convert normalized actions [-1, 1] to VLA0 space-separated integer format.

    Args:
        actions: Shape [T, action_dim] or [action_dim]. Normalized to [-1, 1].
        num_bins: Number of discretization bins. Default 1000.
        action_horizon: If given, take the last `action_horizon` timesteps.

    Returns:
        Space-separated integer string, e.g. "512 487 523 500 501 499 0".
    """
    arr = np.asarray(actions, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]  # [1, D]
    if action_horizon is not None:
        arr = arr[-action_horizon:, :]

    arr = np.clip(arr, -1.0, 1.0)
    discretized = np.round((arr + 1.0) / 2.0 * num_bins).astype(int)
    discretized = np.clip(discretized, 0, num_bins)
    return " ".join(map(str, discretized.flatten()))


def parse_vla0_to_actions(
    text: str,
    num_bins: int = 1000,
    action_horizon: int = 1,
    action_dim: int = 7,
) -> np.ndarray:
    """Parse VLA0 text back to normalized actions.

    Args:
        text: Space-separated integers, e.g. "512 487 523 500 501 499 0".
        num_bins: Must match the value used during encoding.
        action_horizon: Number of timesteps to reconstruct.
        action_dim: Action dimension per timestep.

    Returns:
        np.ndarray of shape [action_horizon, action_dim] in [-1, 1].
    """
    try:
        ints = [int(x) for x in text.strip().split() if x.strip().isdigit() or (x.strip().lstrip("-").isdigit())]
    except ValueError:
        return np.zeros((action_horizon, action_dim), dtype=float)

    if not ints:
        return np.zeros((action_horizon, action_dim), dtype=float)

    continuous = np.array(ints, dtype=float) / num_bins * 2.0 - 1.0

    expected_len = action_horizon * action_dim
    if len(continuous) < expected_len:
        continuous = np.pad(continuous, (0, expected_len - len(continuous)))
    elif len(continuous) > expected_len:
        continuous = continuous[:expected_len]

    return continuous.reshape(action_horizon, action_dim)


# ──────────────────────────────────────────────────────────────────────
#  Verbose / Compact Format  (requires delta EEF actions in SI units)
# ──────────────────────────────────────────────────────────────────────


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _summarize_compact(arr: np.ndarray, include_rotation: bool = False) -> str:
    """Compact format: "<+05 -03 +02 0>" (signed cm integers)."""
    dx_cm = int(round(float(arr[..., 0].sum()) * 100.0))
    dy_cm = int(round(float(arr[..., 1].sum()) * 100.0))
    dz_cm = int(round(float(arr[..., 2].sum()) * 100.0))
    parts = [f"{dx_cm:+03d}", f"{dy_cm:+03d}", f"{dz_cm:+03d}"]

    if include_rotation:
        droll = _round_to_nearest_n(float(arr[..., 3].sum()) * 180.0 / np.pi, 5)
        dpitch = _round_to_nearest_n(float(arr[..., 4].sum()) * 180.0 / np.pi, 5)
        dyaw = _round_to_nearest_n(float(arr[..., 5].sum()) * 180.0 / np.pi, 5)
        parts.extend([f"{droll:+03d}", f"{dpitch:+03d}", f"{dyaw:+03d}"])

    g_last = float(arr[-1, 6])
    parts.append(str(1 if g_last >= 0.5 else 0))
    return "<" + " ".join(parts) + ">"


def summarize_numeric_actions(
    arr_like,
    format: str = "0f",
    include_rotation: bool = False,
    rotation_precision: int = 10,
) -> str | None:
    """Convert action array to natural language description (verbose format).

    Args:
        arr_like: Action array of shape [T, action_dim] or [action_dim].
                  Assumes delta EEF: [:,0:3] = (dx, dy, dz) in **meters**,
                  [:,3:6] = (droll, dpitch, dyaw) in **radians**,
                  [:,6] = gripper state (>=0.5 → open).
        format: "compact" | "Nf" (N decimal places) | "no_number".
        include_rotation: Whether to include rotation description.
        rotation_precision: Round rotation to nearest N degrees.

    Returns:
        Natural language description string, or None if shape invalid.
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    if format == "compact":
        return _summarize_compact(arr, include_rotation)

    # Determine decimal places
    if format in {"no_number", "nearest_10"}:
        decimals = 0
    else:
        m = re.fullmatch(r"(\d+)f", format)
        decimals = int(m.group(1)) if m else 0

    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    droll_rad = dpitch_rad = dyaw_rad = 0.0
    droll = dpitch = dyaw = 0
    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), rotation_precision)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), rotation_precision)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), rotation_precision)

    parts: list[str] = []

    if format == "no_number":
        # Direction-only descriptions
        if dx_m > 0 and dx != 0:
            parts.append("move forward")
        elif dx_m < 0 and dx != 0:
            parts.append("move back")
        if dy_m > 0 and dy != 0:
            parts.append("move left")
        elif dy_m < 0 and dy != 0:
            parts.append("move right")
        if dz_m > 0 and dz != 0:
            parts.append("move up")
        elif dz_m < 0 and dz != 0:
            parts.append("move down")
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = f"{dx:.{decimals}f}"
        fmt_dy = f"{dy:.{decimals}f}"
        fmt_dz = f"{dz:.{decimals}f}"
        # X-axis: forward/back
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        # Z-axis: up/down
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        # Y-axis: left/right
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    g_last = float(arr[-1, 6])
    parts.append("open gripper" if g_last >= 0.5 else "close gripper")

    if not parts:
        return "stay still, close gripper" if g_last < 0.5 else "stay still, open gripper"
    return ", ".join(parts)


# ──────────────────────────────────────────────────────────────────────
#  Unified entry point
# ──────────────────────────────────────────────────────────────────────


def convert_action_to_language(
    actions: np.ndarray,
    lang_action_format: str = "vla0",
    action_horizon: int = None,
    include_rotation: bool = False,
    num_bins: int = 1000,
    decimal_places: int = 0,
) -> str:
    """Convert action array to language string.

    Args:
        actions: Shape [T, action_dim] or [action_dim].
        lang_action_format: "vla0" | "verbose" | "compact" | "no_number".
        action_horizon: For VLA0: take last action_horizon steps.
                        For verbose/compact: the entire array is summed.
        include_rotation: Include rotation components (verbose/compact only).
        num_bins: Discretization bins for VLA0.
        decimal_places: Decimal places for verbose format.

    Returns:
        Language action string.
    """
    arr = np.asarray(actions, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    if lang_action_format == "vla0":
        return actions_to_vla0(arr, num_bins=num_bins, action_horizon=action_horizon)
    elif lang_action_format in {"verbose", "no_number", "nearest_10"} or re.match(r"\d+f", lang_action_format):
        fmt = f"{decimal_places}f" if lang_action_format == "verbose" else lang_action_format
        if action_horizon is not None:
            arr = arr[-action_horizon:, :]
        result = summarize_numeric_actions(arr, format=fmt, include_rotation=include_rotation)
        return result or "stay still, close gripper"
    elif lang_action_format == "compact":
        if action_horizon is not None:
            arr = arr[-action_horizon:, :]
        result = summarize_numeric_actions(arr, format="compact", include_rotation=include_rotation)
        return result or "<+00 +00 +00 0>"
    else:
        raise ValueError(
            f"Unknown lang_action_format: {lang_action_format!r}. "
            f"Expected 'vla0', 'verbose', 'compact', or 'no_number'."
        )


def parse_language_to_actions(
    text: str,
    lang_action_format: str = "vla0",
    action_horizon: int = 1,
    action_dim: int = 7,
    num_bins: int = 1000,
) -> np.ndarray:
    """Parse language action string back to action array.

    Args:
        text: Generated language action text.
        lang_action_format: "vla0" or "verbose" / "compact".
        action_horizon: Number of timesteps for VLA0.
        action_dim: Action dimension.
        num_bins: For VLA0 decoding.

    Returns:
        np.ndarray of shape [action_horizon, action_dim].
    """
    if lang_action_format == "vla0":
        return parse_vla0_to_actions(text, num_bins=num_bins, action_horizon=action_horizon, action_dim=action_dim)

    # For verbose / compact: parse a single-step delta action
    movement = np.zeros(6, dtype=float)
    gripper = 0.0

    if lang_action_format == "compact":
        # Pattern: <+05 -03 +02 ...>
        pat6 = re.search(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>", text)
        pat3 = re.search(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>", text)
        if pat6:
            g = pat6.groups()
            movement[:3] = np.array([g[0], g[1], g[2]], dtype=float) / 100.0
            movement[3:6] = np.array([g[3], g[4], g[5]], dtype=float) * np.pi / 180.0
            gripper = float(g[6])
        elif pat3:
            g = pat3.groups()
            movement[:3] = np.array([g[0], g[1], g[2]], dtype=float) / 100.0
            gripper = float(g[3])
    else:
        # Verbose parsing
        text = text.replace("slightly", "1.5 cm").replace("moderately", "5 cm").replace("a lot", "10 cm")
        move_pat = re.compile(r"move\s+(forward|back(?:ward)?|left|right|up|down)(?:\s+([\d.]+)\s*cm)?", re.IGNORECASE)
        dx = dy = dz = 0.0
        for m in move_pat.finditer(text):
            d = m.group(1).lower()
            v = float(m.group(2)) if m.group(2) else 0.0
            if d == "forward":
                dx += v
            elif d in ("back", "backward"):
                dx -= v
            elif d == "left":
                dy += v
            elif d == "right":
                dy -= v
            elif d == "up":
                dz += v
            elif d == "down":
                dz -= v
        movement[:3] = np.array([dx, dy, dz], dtype=float) / 100.0

        rot_pat = re.compile(
            r"(tilt left|tilt right|tilt back|tilt forward|rotate counterclockwise|rotate clockwise)\s+([\d.]+)\s*degrees",
            re.IGNORECASE,
        )
        droll = dpitch = dyaw = 0.0
        for m in rot_pat.finditer(text):
            kind = m.group(1).lower()
            v = float(m.group(2))
            if kind == "tilt left":
                droll += v
            elif kind == "tilt right":
                droll -= v
            elif kind == "tilt back":
                dpitch += v
            elif kind == "tilt forward":
                dpitch -= v
            elif kind == "rotate counterclockwise":
                dyaw += v
            elif kind == "rotate clockwise":
                dyaw -= v
        movement[3:6] = [droll * np.pi / 180.0, dpitch * np.pi / 180.0, dyaw * np.pi / 180.0]

        if "open gripper" in text.lower():
            gripper = 1.0
        elif "close gripper" in text.lower():
            gripper = 0.0
        else:
            gripper = 0.0

    # Build action array: repeat the single-step delta for action_horizon
    single = np.append(movement, gripper)[:action_dim]
    return np.tile(single, (action_horizon, 1))
