import numpy as np
import pandas as pd
from data import MODEL_FEATURES, PRIMARY, transform_raw

RATIO_COLUMNS = ['fare_ratio', 'distance_ratio', 'duration_ratio']


RULE_FEATURES = RATIO_COLUMNS + ['speed', 'request_wait_min']


RULES = {0: {'name': 'excess_distance_fare_duration_low_speed', 'ranges': [[1.5, 3.0], [1.5, 3.0], [1.75, 4.0], [5.0, 18.0], [0.0, 8.0]]}, 1: {'name': 'near_baseline_normal_candidate', 'ranges': [[0.85, 1.15], [0.85, 1.15], [0.8, 1.2], [25.0, 65.0], [0.0, 8.0]]}, 2: {'name': 'longer_faster_time_saving_candidate', 'ranges': [[1.4, 2.5], [1.4, 2.5], [0.5, 0.95], [35.0, 90.0], [0.0, 8.0]]}, 3: {'name': 'duration_excess_low_speed_candidate', 'ranges': [[0.85, 1.15], [0.85, 1.15], [1.75, 4.0], [5.0, 15.0], [0.0, 8.0]]}, 4: {'name': 'long_request_to_pickup_wait_candidate', 'ranges': [[0.85, 1.15], [0.85, 1.15], [0.8, 1.2], [25.0, 65.0], [15.0, 45.0]]}}


COMMON_RANGES = {'baseline_price_x': [5.0, 150.0], 'baseline_drive_mile_x': [2.0, 40.0], 'baseline_drive_time_x': [5.0, 90.0], 'price_x': [5.0, 300.0], 'drive_mile_x': [1.0, 80.0], 'drive_time_x': [3.0, 180.0], 'speed': [5.0, 100.0], 'request_wait_min': [0.0, 60.0]}


TOTAL_ROWS = 50000


MODE_CAP = 10000


MIN_MODE_ROWS = 100


def add_ratios(frame):
    result = frame.copy()
    result['fare_ratio'] = result.price_x / result.baseline_price_x
    result['distance_ratio'] = result.drive_mile_x / result.baseline_drive_mile_x
    result['duration_ratio'] = result.drive_time_x / result.baseline_drive_time_x
    return result


def assign_candidates(frame):
    """Rules are disjoint by construction; margins rank prototypes, not DBSCAN fits."""
    common = (frame.baseline_source == 'provider_OD') & (frame.baseline_count >= 100) & (frame.request_wait_missing == 0) & (frame.duration_clock_difference_sec.abs() <= 60)
    for feature, (lower, upper) in COMMON_RANGES.items():
        common &= frame[feature].between(lower, upper)
    common &= np.isfinite(frame[MODEL_FEATURES + RATIO_COLUMNS].to_numpy(float)).all(axis=1)
    values = frame[RULE_FEATURES].to_numpy(float)
    modes = np.full(len(frame), -1, dtype=np.int8)
    margins = np.full(len(frame), np.nan)
    hits = np.zeros(len(frame), dtype=np.int8)
    for mode, rule in RULES.items():
        bounds = np.asarray(rule['ranges'])
        inside = common.to_numpy() & ((values >= bounds[:, 0]) & (values <= bounds[:, 1])).all(axis=1)
        hits += inside
        modes[inside] = mode
        normalized = (values[inside] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        margins[inside] = (2 * np.minimum(normalized, 1 - normalized)).min(axis=1)
    if np.any(hits > 1):
        raise AssertionError('Diagnostic selection rules overlap')
    return (modes, margins, common.to_numpy())


def retain_best(previous, candidate, capacity):
    if len(candidate) > capacity:
        candidate = candidate.sort_values(['diagnostic_margin', 'selection_priority', 'record_id'], ascending=[False, True, True], kind='stable').head(capacity)
    merged = candidate if previous is None else pd.concat([previous, candidate], ignore_index=True)
    return merged.sort_values(['diagnostic_margin', 'selection_priority', 'record_id'], ascending=[False, True, True], kind='stable').head(capacity).reset_index(drop=True)


def allocate_quotas(counts):
    if any((counts.get(mode, 0) < MIN_MODE_ROWS for mode in RULES)):
        raise ValueError(f'Insufficient candidates for five patterns (minimum {MIN_MODE_ROWS} each): {dict(counts)}')
    quotas = {mode: min(MODE_CAP, counts[mode]) for mode in RULES if mode != 1}
    quotas[1] = TOTAL_ROWS - sum(quotas.values())
    if counts[1] < quotas[1]:
        raise ValueError('Insufficient normal candidates to fill the fixed 50,000-order dataset')
    return dict(sorted(quotas.items()))
