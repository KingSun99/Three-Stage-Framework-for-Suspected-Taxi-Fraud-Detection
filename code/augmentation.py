import random
import numpy as np
import pandas as pd
from data import DISCRETE_PRIMITIVES, MODEL_FEATURES, NUMERIC_PRIMITIVES, reconstruct

class ConstraintTransform:
    """Represent zero atoms explicitly; log positive magnitudes, then reverse."""
    positive = {'drive_time_x', 'drive_mile_x', 'baseline_drive_time_x', 'baseline_drive_mile_x'}

    def fit_transform(self, frame):
        self.fill = {}
        output = frame[DISCRETE_PRIMITIVES].copy()
        self.discrete = list(DISCRETE_PRIMITIVES)
        for name in NUMERIC_PRIMITIVES:
            values = frame[name].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (name in self.positive and (values <= 0).any()):
                raise ValueError(f'Invalid real training values for constraint: {name}')
            positive_values = values[values > 0]
            fill = float(np.median(positive_values)) if len(positive_values) else 1.0
            self.fill[name] = fill
            output[name] = np.log(np.where(values > 0, values, fill))
            if name not in self.positive:
                flag = name + '__zero'
                output[flag] = (values == 0).astype(int)
                self.discrete.append(flag)
        return output

    def inverse(self, frame, rush_hours):
        output = frame[DISCRETE_PRIMITIVES].copy()
        for name in NUMERIC_PRIMITIVES:
            values = np.exp(frame[name].to_numpy(dtype=float).clip(-40, 40))
            if name not in self.positive:
                values = np.where(frame[name + '__zero'].to_numpy() == 1, 0.0, values)
            output[name] = values
        return reconstruct(output, rush_hours)


def constraint_audit(frame):
    if len(frame) == 0:
        return {'rows': 0, 'finite_fraction': None, 'nonnegative_fraction': None, 'speed_consistency_fraction': None, 'weekend_consistency_fraction': None, 'deviation_consistency_fraction': None}
    finite = np.isfinite(frame[MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)
    positive = (frame.drive_time_x > 0) & (frame.drive_mile_x > 0)
    nonnegative = (frame[['price_x', 'request_wait_min', 'tolls', 'airport_fee', 'congestion_surcharge', 'cbd_congestion_fee']] >= 0).all(axis=1) & positive
    expected_speed = frame.drive_mile_x / (frame.drive_time_x / 60.0)
    consistent = np.isclose(frame.speed, expected_speed, rtol=1e-07, atol=1e-07)
    result = {'rows': len(frame), 'finite_fraction': float(finite.mean()), 'nonnegative_fraction': float(nonnegative.mean()), 'speed_consistency_fraction': float(consistent.mean()), 'weekend_consistency_fraction': float((frame.isWeekend == (frame.weekday >= 5)).mean())}
    if all(('baseline_' + name in frame for name in ['price_x', 'drive_time_x', 'drive_mile_x'])):
        consistent_deviation = np.ones(len(frame), dtype=bool)
        for name, dv in [('price_x', 'DV_price'), ('drive_time_x', 'DV_drive_time'), ('drive_mile_x', 'DV_drive_mile')]:
            consistent_deviation &= np.isclose(frame[dv], frame[name] - frame['baseline_' + name], rtol=1e-07, atol=1e-07)
        result['deviation_consistency_fraction'] = float(consistent_deviation.mean())
    return result


def seed_everything(seed, threads=4):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)


def generate_smote(train, targets, config):
    from imblearn.over_sampling import SMOTENC
    from sklearn.preprocessing import MinMaxScaler
    if not targets:
        return (pd.DataFrame(columns=MODEL_FEATURES + ['pattern_label']), [])
    features = train[MODEL_FEATURES]
    labels = train.pattern_label.to_numpy()
    scaler = MinMaxScaler().fit(features)
    categorical = [MODEL_FEATURES.index(name) for name in ['weekday', 'isWeekend', 'isRushHour', 'request_wait_missing']]
    min_count = min((int((labels == label).sum()) for label in targets))
    sampling = {label: int((labels == label).sum()) + n for label, n in targets.items()}
    sampler = SMOTENC(categorical_features=categorical, sampling_strategy=sampling, k_neighbors=min(5, min_count - 1), random_state=config['seed'])
    x_augmented, y_augmented = sampler.fit_resample(scaler.transform(features), labels)
    generated = pd.DataFrame(scaler.inverse_transform(x_augmented[len(train):]), columns=MODEL_FEATURES)
    generated['pattern_label'] = y_augmented[len(train):]
    return (generated, [{'method': 'smote', 'variant': 'SMOTENC (categorical temporal indicators)', **constraint_audit(generated)}])
