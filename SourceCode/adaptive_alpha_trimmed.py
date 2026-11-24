"""Optimized Adaptive Alpha-Trimmed Mean restoration for grayscale and RGB images.

Vectorized implementation based on Oten and de Figueiredo (IEEE TIP, 2004).
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ArrayImage = np.ndarray
NoiseModel = Literal["gaussian", "salt_pepper", "speckle", "poisson", "mixed"]
CriterionMode = Literal["spread", "paper"]


def _validate_image(image: ArrayImage) -> ArrayImage:
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError("image must have shape (height, width), RGB, or RGBA")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must contain numeric pixel values")
    return array


def add_noise(
    image: ArrayImage,
    model: NoiseModel,
    amount: float = 0.05,
    sigma: float = 0.08,
    seed: int | None = None,
) -> ArrayImage:
    """Return a clipped noisy copy of an image represented in [0, 1] or [0, 255]."""
    source = _validate_image(image).astype(np.float32)
    scale = 255.0 if source.max(initial=0) > 1.0 else 1.0
    normalized = np.clip(source / scale, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    model = model.lower()

    if model == "gaussian":
        noisy = normalized + rng.normal(0.0, sigma, normalized.shape).astype(np.float32)
    elif model == "salt_pepper":
        noisy = normalized.copy()
        mask = rng.random(normalized.shape[:2]) < amount
        salt = rng.random(normalized.shape[:2]) >= 0.5
        if noisy.ndim == 3:
            noisy[mask] = np.where(salt[mask, None], 1.0, 0.0)
        else:
            noisy[mask] = np.where(salt[mask], 1.0, 0.0)
    elif model == "speckle":
        noisy = normalized + normalized * rng.normal(0.0, sigma, normalized.shape).astype(np.float32)
    elif model == "poisson":
        levels = 30.0
        noisy = rng.poisson(np.clip(normalized, 0.0, 1.0) * levels).astype(np.float32) / levels
    elif model == "mixed":
        noisy = normalized + rng.normal(0.0, sigma, normalized.shape).astype(np.float32)
        mask = rng.random(normalized.shape[:2]) < amount
        salt = rng.random(normalized.shape[:2]) >= 0.5
        if noisy.ndim == 3:
            noisy[mask] = np.where(salt[mask, None], 1.0, 0.0)
        else:
            noisy[mask] = np.where(salt[mask], 1.0, 0.0)
    else:
        raise ValueError(f"unknown noise model: {model}")

    result = np.clip(noisy, 0.0, 1.0) * scale
    return result.astype(image.dtype, copy=False)


def _adaptive_value(values: np.ndarray, threshold: float, max_alpha: float) -> float:
    """Fast single-window fallback using Eq. (24)-(25) cumulative sums."""
    ordered = np.sort(values.astype(np.float32, copy=False))
    count = ordered.size
    mid = count // 2
    median = float(ordered[mid])

    left = ordered[mid - 1 :: -1]
    right = ordered[mid + 1 :]
    pair_sums = np.cumsum(left + right)
    idx = np.arange(1, mid + 1, dtype=np.float32)

    av_rest = (median + pair_sums) / (2.0 * idx + 1.0)
    asymmetry = np.abs(pair_sums / idx - 2.0 * median)
    spread = np.maximum(right - left, np.finfo(np.float32).eps)

    min_trim = max(mid - int(np.floor(max_alpha * count / 2.0)), 0)
    valid = asymmetry >= 0.0
    valid[:min_trim] = False
    valid &= asymmetry <= threshold * spread

    if np.any(valid):
        best_i = int(np.nonzero(valid)[0][-1])
        return float(av_rest[best_i])
    return median


def adaptive_alpha_trimmed(
    image: ArrayImage,
    window_size: int = 5,
    threshold: float = 0.5,
    max_alpha: float = 0.45,
    mode: CriterionMode = "spread",
    chunk_rows: int = 128,
) -> ArrayImage:
    """Vectorized locally adaptive alpha-trimmed mean filter.

    Parameters
    ----------
    image : np.ndarray
        Input 2D grayscale or 3D RGB/RGBA image.
    window_size : int
        Odd window size >= 3 (default 5).
    threshold : float
        Symmetry tolerance T (default 0.5).
    max_alpha : float
        Maximum trimming fraction in [0, 0.5).
    mode : {"spread", "paper"}
        - "spread": vectorized version of the normalized spread symmetry test.
        - "paper": exact recurrence from Section IV-B (Eq. 23-25) of Oten (2004).
    chunk_rows : int
        Number of image rows processed per batch to keep RAM usage low.
    """
    source = _validate_image(image)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if not 0.0 <= threshold or not 0.0 <= max_alpha < 0.5:
        raise ValueError("threshold must be non-negative and max_alpha in [0, 0.5)")

    original_dtype = source.dtype
    scale = 255.0 if source.max(initial=0) > 1.0 else 1.0
    data = np.clip(source.astype(np.float32) / scale, 0.0, 1.0)
    channels = data[..., None] if data.ndim == 2 else data[..., :3]

    height, width, num_channels = channels.shape
    radius = window_size // 2
    n = window_size * window_size
    mid = n // 2  # N in paper (n = 2N + 1)

    padded = np.pad(channels, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(channels)

    idx = np.arange(1, mid + 1, dtype=np.float32)
    counts = 2.0 * idx + 1.0
    eps = np.finfo(np.float32).eps

    # Process in row chunks so memory stays small even for large images
    for r_start in range(0, height, chunk_rows):
        r_end = min(r_start + chunk_rows, height)
        patch = padded[r_start : r_end + 2 * radius, :, :]

        # Shape: (chunk_H, W, C, window_size, window_size) -> (chunk_H, W, C, n)
        windows = sliding_window_view(patch, (window_size, window_size), axis=(0, 1))
        flat_windows = windows.reshape(r_end - r_start, width, num_channels, n)

        # Sort samples inside every window simultaneously in C
        ordered = np.sort(flat_windows, axis=-1)
        median = ordered[..., mid]  # S(0) = Av(0)

        # Samples expanding symmetrically outward from median: i = 1 .. N
        left = ordered[..., mid - 1 :: -1]   # x_{(N+1-i)}
        right = ordered[..., mid + 1 :]      # x_{(N+1+i)}

        # Eq. (25): S(i) = S(i-1) + x_{(N+1-i)} + x_{(N+1+i)}
        pair_cumsum = np.cumsum(left + right, axis=-1)
        av_rest = (median[..., None] + pair_cumsum) / counts  # Av(1) .. Av(N)

        if mode == "paper":
            # Exact Section IV-B (Eq. 23-25): D(i) = |Av(i) - Av(i-1)| <= T / n
            av_all = np.concatenate([median[..., None], av_rest], axis=-1)
            diff = np.abs(av_all[..., 1:] - av_all[..., :-1])
            min_trim = int(np.ceil((0.5 - max_alpha) * n)) // 2
            max_i = max(mid - min_trim, 1)

            cond = diff <= (threshold / n)
            if max_i < mid:
                cond[..., max_i:] = False
            # Step 1: while D(i) < T/n and i <= N (continuous from i=1)
            still_symmetric = np.cumprod(cond, axis=-1).astype(bool)
            # Pick largest i before symmetry fails; fallback to Av(0)=median
            valid_count = np.sum(still_symmetric, axis=-1)
            output[r_start:r_end] = np.take_along_axis(
                av_all, valid_count[..., None], axis=-1
            )[..., 0]
        else:
            # Vectorized spread criterion (matches original _adaptive_value logic)
            asymmetry = np.abs(pair_cumsum / idx - 2.0 * median[..., None])
            spread = np.maximum(right - left, eps)
            valid = asymmetry <= (threshold * spread)

            max_trim = min(int(np.floor(max_alpha * n / 2.0)), mid)
            min_i = mid - max_trim  # minimum central expansion index
            if min_i > 1:
                valid[..., : min_i - 1] = False

            # Find largest expansion index i (smallest trim) that satisfies symmetry
            any_valid = np.any(valid, axis=-1)
            rev_argmax = valid.shape[-1] - 1 - np.argmax(valid[..., ::-1], axis=-1)
            chosen_av = np.take_along_axis(av_rest, rev_argmax[..., None], axis=-1)[..., 0]

            # Fallback for trim == max_trim (when min_i == 0, i.e., central has 1 sample = median)
            output[r_start:r_end] = np.where(any_valid, chosen_av, median)

    if data.ndim == 2:
        result = output[..., 0]
    else:
        result = output
        if source.shape[2] == 4:
            result = np.dstack((result, data[..., 3]))
    return (np.clip(result, 0.0, 1.0) * scale).astype(original_dtype)


def psnr(reference: ArrayImage, restored: ArrayImage) -> float:
    """Compute peak signal-to-noise ratio in dB for two equally sized images."""
    first = _validate_image(reference).astype(np.float64)
    second = _validate_image(restored).astype(np.float64)
    if first.shape != second.shape:
        raise ValueError("reference and restored images must have the same shape")
    peak = 255.0 if first.max(initial=0) > 1.0 or second.max(initial=0) > 1.0 else 1.0
    mse = float(np.mean((first - second) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10((peak * peak) / mse)