import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ClientProfile:
    device_tier: int
    model_tier: int
    heterogeneity_score: float
    entropy_norm: float
    gini: float
    compute_power: float
    network_speed: int
    storage_capacity: int


HETEROGENEITY_THRESHOLD_A = 0.35
HETEROGENEITY_THRESHOLD_B = 0.90


def device_tier_label(device_tier: int) -> str:
    labels = {
        1: "低等",
        2: "中低",
        3: "中",
        4: "中高",
        5: "高等",
    }
    return labels.get(int(device_tier), "未知等级")


def _gini_from_nonnegative(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if np.any(values < 0):
        values = np.clip(values, 0, None)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * (index * sorted_vals).sum()) / (n * total) - (n + 1.0) / n
    return float(max(0.0, min(1.0, gini)))


def _entropy_norm_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    eps = 1e-12
    ent = -float(np.sum(probs * np.log(probs + eps)))
    base = math.log(max(2, probs.size))
    return float(max(0.0, min(1.0, ent / base)))


def compute_heterogeneity_score_from_counts(counts: Sequence[int]) -> Tuple[float, float, float]:
    counts_arr = np.asarray(counts, dtype=np.float64)
    entropy_norm = _entropy_norm_from_counts(counts_arr)
    gini = _gini_from_nonnegative(counts_arr)
    score = 0.5 * (1.0 - entropy_norm) + 0.5 * gini
    score = float(max(0.0, min(1.0, score)))
    return score, entropy_norm, gini


def count_labels_from_dataloader(dataloader, n_classes: int, max_batches: Optional[int] = None) -> np.ndarray:
    counts = np.zeros(int(n_classes), dtype=np.int64)
    for batch_idx, (_, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        try:
            t = targets.detach().cpu().numpy()
        except Exception:
            t = np.asarray(targets)
        t = t.reshape(-1)
        for label in t.tolist():
            if 0 <= int(label) < n_classes:
                counts[int(label)] += 1
    return counts


def allocate_device_tiers(client_number: int, rng: Optional[random.Random] = None) -> Dict[int, int]:
    rng = rng or random
    tier_weights = [0.20, 0.25, 0.20, 0.20, 0.15]
    tiers = rng.choices(range(1, 6), weights=tier_weights, k=int(client_number))
    return {cid: int(tiers[cid]) for cid in range(int(client_number))}


def sample_resources_for_device_tier(device_tier: int, rng: Optional[random.Random] = None) -> Tuple[float, int, int]:
    rng = rng or random
    device_tier = int(device_tier)
    if device_tier == 5:
        compute_power = rng.uniform(0.90, 1.00)
        network_speed = rng.choice([200, 300])
        storage_capacity = rng.choice([1024, 2048])
    elif device_tier == 4:
        compute_power = rng.uniform(0.75, 0.85)
        network_speed = rng.choice([100, 150])
        storage_capacity = rng.choice([512, 768])
    elif device_tier == 3:
        compute_power = rng.uniform(0.55, 0.70)
        network_speed = rng.choice([50, 80])
        storage_capacity = rng.choice([256, 384])
    elif device_tier == 2:
        compute_power = rng.uniform(0.35, 0.50)
        network_speed = rng.choice([20, 30])
        storage_capacity = rng.choice([128, 192])
    else:
        compute_power = rng.uniform(0.15, 0.30)
        network_speed = rng.choice([5, 10])
        storage_capacity = rng.choice([32, 64])
    return float(compute_power), int(network_speed), int(storage_capacity)


def candidate_model_tiers_for_device_tier(device_tier: int) -> List[int]:
    device_tier = int(device_tier)
    mapping = {
        5: [1, 2, 3, 4],
        4: [3, 4, 5],
        3: [4, 5, 6],
        2: [5, 6, 7],
        1: [6, 7],
    }
    candidates = mapping.get(device_tier, [4, 5, 6])
    return sorted({int(t) for t in candidates})


def filter_candidates_by_heterogeneity(candidates: Sequence[int], heterogeneity_score: float) -> List[int]:
    uniq = sorted({int(t) for t in candidates})
    if not uniq:
        return [4]
    score = float(heterogeneity_score)
    a = float(HETEROGENEITY_THRESHOLD_A)
    b = float(HETEROGENEITY_THRESHOLD_B)

    if score < a:
        keep = min(2, len(uniq))
        return uniq[-keep:]

    if score > b:
        keep = min(2, len(uniq))
        return uniq[:keep]

    return uniq


def choose_model_tier(candidates: Sequence[int], heterogeneity_score: float) -> int:
    filtered = filter_candidates_by_heterogeneity(candidates, heterogeneity_score)
    filtered = sorted({int(t) for t in filtered})
    if not filtered:
        return 4
    return int(max(filtered))


def build_initial_client_profiles(
    client_number: int,
    train_data_local_dict: Dict[int, object],
    n_classes: int,
    seed: int = 42,
) -> Dict[int, ClientProfile]:
    rng = random.Random(int(seed))
    device_tiers = allocate_device_tiers(client_number, rng=rng)
    profiles: Dict[int, ClientProfile] = {}
    for client_id in range(int(client_number)):
        counts = count_labels_from_dataloader(train_data_local_dict[client_id], n_classes=n_classes)
        score, entropy_norm, gini = compute_heterogeneity_score_from_counts(counts)
        candidates = candidate_model_tiers_for_device_tier(device_tiers[client_id])
        model_tier = choose_model_tier(candidates, score)
        compute_power, network_speed, storage_capacity = sample_resources_for_device_tier(device_tiers[client_id], rng=rng)
        profiles[client_id] = ClientProfile(
            device_tier=int(device_tiers[client_id]),
            model_tier=int(model_tier),
            heterogeneity_score=float(score),
            entropy_norm=float(entropy_norm),
            gini=float(gini),
            compute_power=float(compute_power),
            network_speed=int(network_speed),
            storage_capacity=int(storage_capacity),
        )
    return profiles


def mutate_device_tiers(
    current_device_tiers: Dict[int, int],
    fraction: float = 0.30,
    seed: int = 0,
) -> Dict[int, int]:
    rng = random.Random(int(seed))
    client_ids = list(current_device_tiers.keys())
    k = int(round(len(client_ids) * float(fraction)))
    k = max(0, min(len(client_ids), k))
    chosen = rng.sample(client_ids, k=k) if k > 0 else []
    new_tiers = dict(current_device_tiers)
    for cid in chosen:
        old = int(new_tiers[cid])
        candidates = [t for t in range(1, 6) if t != old]
        new_tiers[cid] = int(rng.choice(candidates))
    return new_tiers


def recompute_model_tiers_from_device_and_scores(
    device_tiers: Dict[int, int],
    heterogeneity_scores: Dict[int, float],
) -> Dict[int, int]:
    model_tiers: Dict[int, int] = {}
    for cid, d_tier in device_tiers.items():
        score = float(heterogeneity_scores.get(cid, 0.5))
        candidates = candidate_model_tiers_for_device_tier(d_tier)
        model_tiers[cid] = int(choose_model_tier(candidates, score))
    return model_tiers
