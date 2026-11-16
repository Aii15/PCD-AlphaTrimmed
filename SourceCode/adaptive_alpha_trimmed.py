"""Adaptive alpha-trimmed mean restoration for grayscale and RGB images.

The adaptive rule follows Oten and de Figueiredo (IEEE TIP, 2004):
for every local window, begin with the most aggressively trimmed central
sample set and decrease the trim until the retained samples are sufficiently
locally symmetric.  The implementation is channel-independent for RGB data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


ArrayImage = np.ndarray
NoiseModel = Literal["gaussian", "salt_pepper", "speckle", "poisson", "mixed"]


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
    """Return a clipped noisy copy of an image represented in [0, 1] or [0, 255].

    ``amount`` is the impulse density for salt-and-pepper noise and the
    proportion of the two components for mixed noise. ``sigma`` is the
    standard deviation of Gaussian/speckle noise in normalized [0, 1] units.
    """
    source = _validate_image(image).astype(np.float32)
    scale = 255.0 if source.max(initial=0) > 1.0 else 1.0
    normalized = np.clip(source / scale, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    model = model.lower()

    if model == "gaussian":
        noisy = normalized + rng.normal(0.0, sigma, normalized.shape)
    elif model == "salt_pepper":
        noisy = normalized.copy()
        mask = rng.random(normalized.shape[:2]) < amount
        salt = rng.random(normalized.shape[:2]) >= 0.5
        noisy[mask] = salt[mask, None] if noisy.ndim == 3 else salt[mask]
        if noisy.ndim == 3:
            noisy[mask] = np.where(salt[mask, None], 1.0, 0.0)
        else:
            noisy[mask] = np.where(salt[mask], 1.0, 0.0)
    elif model == "speckle":
        noisy = normalized + normalized * rng.normal(0.0, sigma, normalized.shape)
    elif model == "poisson":
        levels = 30.0
        noisy = rng.poisson(np.clip(normalized, 0.0, 1.0) * levels) / levels
    elif model == "mixed":
        noisy = normalized + rng.normal(0.0, sigma, normalized.shape)
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
    ordered = np.sort(values.astype(np.float32, copy=False))
    count = ordered.size
    median = float(np.median(ordered))
    max_trim = min(int(np.floor(max_alpha * count / 2.0)), (count - 1) // 2)

    # Equation (23)'s symmetry condition is evaluated on each central
    # interval.  The scale keeps the test meaningful for flat windows.
    for trim in range(max_trim, -1, -1):
        central = ordered[trim:count - trim]
        left = central[: central.size // 2]
        right = central[-left.size:] if left.size else central
        if left.size:
            asymmetry = abs(float(np.mean(left) + np.mean(right) - 2.0 * median))
            spread = float(np.max(central) - np.min(central))
            scale = max(spread, np.finfo(np.float32).eps)
            if asymmetry <= threshold * scale:
                return float(np.mean(central))
    return median


def adaptive_alpha_trimmed(
    image: ArrayImage,
    window_size: int = 5,
    threshold: float = 0.5,
    max_alpha: float = 0.45,
) -> ArrayImage:
    """Restore an image using the paper's locally adaptive alpha-trimmed mean.

    ``window_size`` must be odd. ``threshold`` controls tolerance for local
    asymmetry; the paper uses 0.5 in its image experiments. ``max_alpha`` is
    capped below 0.5 so at least one central sample remains.
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
    radius = window_size // 2
    padded = np.pad(channels, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(channels)

    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            window = padded[row:row + window_size, column:column + window_size]
            for channel in range(channels.shape[2]):
                output[row, column, channel] = _adaptive_value(
                    window[..., channel].reshape(-1), threshold, max_alpha
                )

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
