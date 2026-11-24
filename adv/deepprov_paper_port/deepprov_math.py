"""Dependency-light, testable pieces of a DeepProv-inspired Keras *port*.

The OFFICIAL DeepProv GNN/IPG attribution extraction is NOT reproduced here.
The activation intervention and train-only trade-off are adapted from the
paper and official PyTorch source: https://github.com/um-dsp/DeepProv
"""
from __future__ import print_function
import numpy as np


def act_on_numpy(x, reference, beta, mask, alpha=1.0):
    """Official PyTorch act_on semantics: x - alpha * (x-reference) if masked.

    beta can be a scalar or an array broadcastable to x.
    """
    x = np.asarray(x)
    reference = np.asarray(reference, dtype=x.dtype)
    beta = np.asarray(beta, dtype=x.dtype)
    mask = np.asarray(mask, dtype=bool)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('Paper alpha must be in [0,1]')
    delta = x - reference
    eligible = mask & (np.abs(delta) < beta)
    return np.where(eligible, x - alpha * delta, x)


def tradeoff_score(original_clean, original_adv, candidate_clean, candidate_adv):
    """Paper TS for a single targeted attack (two accuracy metrics)."""
    return (candidate_clean - original_clean) + (candidate_adv - original_adv)


def relu_frequency(activations):
    a = np.asarray(activations)
    dims = tuple(range(a.ndim - 1))
    return np.mean(np.abs(a) > 1e-7, axis=dims)


def avg_channel(values):
    arr = np.asarray(values)
    if arr.ndim == 2:
        return np.mean(arr, axis=0)
    if arr.ndim == 4:
        return np.mean(arr, axis=(0, 1, 2))
    raise ValueError('Only Dense (N,C) or Conv (N,H,W,C) activations supported')


def density_peak_reference(values, bandwidth=0.1, grid=100, seed=0):
    """KDE-reference approximation, deterministic. Uses numpy Gaussian KDE.

    From official actions_selections.py: Gaussian KDE on benign activation
    values, then sample from the region with density > max/2.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(arr):
        raise ValueError('Empty activation samples')
    if np.max(arr) - np.min(arr) < 1e-12:
        return float(arr[0])
    if len(arr) > 2048:
        arr = np.random.RandomState(seed).choice(arr, 2048, replace=False)
    xx = np.linspace(np.min(arr), np.max(arr), grid)
    scale = float(max(1e-6, bandwidth))
    density = np.zeros(grid, dtype=np.float64)
    for piece in np.array_split(arr, max(1, int(np.ceil(len(arr) / 128)))):
        density += np.exp(-0.5 * ((xx[:, None] - piece[None, :]) / scale) ** 2).sum(axis=1)
    eligible = density > density.max() * 0.5
    weights = density[eligible]
    weights /= weights.sum()
    return float(np.random.RandomState(seed).choice(xx[eligible], p=weights))


def summarize_per_channel(clean, adv, score_clean_grad, score_adv_grad,
                          seed=0, reference_mode='kde', bandwidth=0.1):
    """IPG-inspired per-channel features, not official GNN attribution.

    The returned beta is a per-channel empirical average absolute difference
    between balanced benign and adversarial samples. When exact source-image
    pairing is unavailable, this is a labelled approximation.
    """
    clean = np.asarray(clean, dtype=np.float32)
    adv = np.asarray(adv, dtype=np.float32)
    assert clean.ndim == adv.ndim and clean.shape[1:] == adv.shape[1:]
    channels = clean.shape[-1]
    freq_b = relu_frequency(clean)
    freq_a = relu_frequency(adv)
    mean_b = avg_channel(clean)
    mean_a = avg_channel(adv)
    shift = np.abs(mean_a - mean_b)
    attr_b = np.asarray(score_clean_grad, dtype=np.float64)
    attr_a = np.asarray(score_adv_grad, dtype=np.float64)
    if len(attr_b) != channels or len(attr_a) != channels:
        raise ValueError('Attribution channel count mismatch')
    # Deterministically pair same-position sampled examples. NOT ground-truth
    # matching of original/FGSM inputs; this is intentionally disclosed.
    m = min(len(clean), len(adv))
    if clean.ndim == 2:
        beta = np.mean(np.abs(clean[:m] - adv[:m]), axis=0)
    else:
        beta = np.mean(np.abs(clean[:m] - adv[:m]), axis=(0, 1, 2))
    beta = np.maximum(beta, 1e-6)
    reference = np.mean(clean, axis=0)
    if reference_mode == 'kde':
        # Per-channel spatial pooling for convolutions is explicitly a port
        # choice; original DeepProv's graph nodes may be spatially resolved.
        for j in range(channels):
            val = density_peak_reference(clean[..., j], bandwidth=bandwidth, seed=seed + j)
            reference[..., j] = val
    elif reference_mode != 'mean':
        raise ValueError('reference_mode: mean/kde')
    # Structural attribution in the official artifact comes from a pretrained
    # GNN explainer. Here the proxy is input-dependent |activation * gradient|.
    thr_b = np.quantile(attr_b, 0.5)
    thr_a = np.quantile(attr_a, 0.5)
    candidates = []
    max_s = max(float(np.max(shift)), 1e-9)
    for idx in range(channels):
        null = bool(freq_b[idx] < 1e-6 and freq_a[idx] > 0)
        high_b = bool(attr_b[idx] >= thr_b)
        high_a = bool(attr_a[idx] >= thr_a)
        if null:
            category = 'nullify'
        elif high_b and high_a and shift[idx] > 0:
            category = 'prioritized'
        elif high_b and shift[idx] > 0:
            category = 'regular'
        else:
            continue
        score = float(shift[idx] / max_s + 0.1 * np.abs(freq_a[idx] - freq_b[idx]) +
                      0.1 * attr_a[idx] / max(float(np.max(attr_a)), 1e-9))
        candidates.append({'neuron_idx': int(idx), 'score': score,
                           'category': category, 'beta': float(beta[idx]),
                           'freq_benign': float(freq_b[idx]),
                           'freq_adversarial': float(freq_a[idx]),
                           'mean_benign': float(mean_b[idx]),
                           'mean_adversarial': float(mean_a[idx]),
                           'attr_benign_proxy': float(attr_b[idx]),
                           'attr_adversarial_proxy': float(attr_a[idx])})
    pri = {'nullify': 0, 'prioritized': 1, 'regular': 2}
    candidates.sort(key=lambda c: (pri[c['category']], -c['score'], c['neuron_idx']))
    return candidates, reference.astype(np.float32), beta.astype(np.float32)
