import numpy as np
import pandas as pd

CORE = ['hvfhs_license_num', 'PULocationID', 'DOLocationID', 'pickup_datetime', 'dropoff_datetime', 'request_datetime', 'trip_miles', 'trip_time', 'base_passenger_fare']


FEES = ['tolls', 'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee']


FLAGS = ['shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 'wav_request_flag', 'wav_match_flag']


PRIMARY = ['price_x', 'drive_time_x', 'drive_mile_x']


MODEL_FEATURES = ['price_x', 'drive_time_x', 'drive_mile_x', 'speed', 'request_wait_min', 'request_wait_missing', 'weekday', 'isWeekend', 'isRushHour', 'tolls', 'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee', 'DV_price', 'DV_drive_time', 'DV_drive_mile']


NUMERIC_PRIMITIVES = ['price_x', 'drive_time_x', 'drive_mile_x', 'request_wait_min', 'baseline_price_x', 'baseline_drive_time_x', 'baseline_drive_mile_x', 'tolls', 'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee']


DISCRETE_PRIMITIVES = ['weekday', 'pickup_hour', 'request_wait_missing']


def transform_raw(raw, config):
    df = raw.copy()
    pickup = pd.to_datetime(df.pickup_datetime)
    dropoff = pd.to_datetime(df.dropoff_datetime)
    request = pd.to_datetime(df.request_datetime)
    df['drive_time_x'] = pd.to_numeric(df.trip_time, errors='coerce') / 60.0
    df['drive_mile_x'] = pd.to_numeric(df.trip_miles, errors='coerce') * 1.609344
    df['price_x'] = pd.to_numeric(df.base_passenger_fare, errors='coerce')
    df['speed'] = df.drive_mile_x / (df.drive_time_x / 60.0)
    wait = (pickup - request).dt.total_seconds() / 60.0
    df['request_wait_missing'] = (wait.isna() | (wait < 0)).astype(int)
    df['request_wait_min'] = wait.where(df.request_wait_missing == 0)
    df['weekday'] = pickup.dt.dayofweek
    df['pickup_hour'] = pickup.dt.hour
    df['isWeekend'] = (df.weekday >= 5).astype(int)
    df['isRushHour'] = df.pickup_hour.isin(config['rush_hours']).astype(int)
    df['duration_clock_difference_sec'] = (dropoff - pickup).dt.total_seconds() - df.trip_time
    conditions = {'outside_month_or_missing_pickup': pickup.isna() | (pickup.dt.strftime('%Y-%m') != config['month']), 'nonpositive_or_missing_duration': ~np.isfinite(df.drive_time_x) | (df.drive_time_x <= 0), 'nonpositive_or_missing_distance': ~np.isfinite(df.drive_mile_x) | (df.drive_mile_x <= 0), 'negative_or_missing_base_fare': ~np.isfinite(df.price_x) | (df.price_x < 0), 'invalid_dropoff_sequence': dropoff.isna() | (dropoff <= pickup), 'unknown_or_nonspatial_zone': ~df.PULocationID.between(1, 263) | ~df.DOLocationID.between(1, 263), 'missing_provider': df.hvfhs_license_num.isna()}
    if config['exclude_shared_accessible']:
        for column in FLAGS:
            conditions['service_' + column] = df[column].fillna('unknown').ne('N')
    for column in FEES:
        conditions['negative_or_missing_' + column] = ~np.isfinite(df[column]) | (df[column] < 0)
    invalid = pd.Series(False, index=df.index)
    for mask in conditions.values():
        invalid |= mask
    counts = {name: int(mask.sum()) for name, mask in conditions.items()}
    return (df.loc[~invalid].copy(), counts)


class ODBaseline:

    def __init__(self, min_count=20, rush_hours=(7, 8, 9, 16, 17, 18)):
        self.min_count = min_count
        self.rush_hours = rush_hours

    def fit(self, reference):
        self.tables = []
        keys_list = [['hvfhs_license_num', 'PULocationID', 'DOLocationID'], ['PULocationID', 'DOLocationID'], ['hvfhs_license_num', 'PULocationID']]
        for keys in keys_list:
            grouped = reference.groupby(keys, observed=True)
            table = grouped[PRIMARY].mean().add_prefix('baseline_')
            table['baseline_count'] = grouped.size()
            self.tables.append((keys, table.loc[table.baseline_count >= self.min_count].reset_index()))
        self.global_mean = reference[PRIMARY].mean()
        self.wait_fill = float(reference.request_wait_min.median())
        if not np.isfinite(self.wait_fill):
            self.wait_fill = 0.0
        return self

    def fit_population_aggregates(self, aggregate, sampled_reference):
        self.tables = []
        for keys in [['hvfhs_license_num', 'PULocationID', 'DOLocationID'], ['PULocationID', 'DOLocationID'], ['hvfhs_license_num', 'PULocationID']]:
            sums = aggregate.groupby(keys, observed=True)[PRIMARY + ['baseline_count']].sum()
            table = sums[PRIMARY].div(sums.baseline_count, axis=0).add_prefix('baseline_')
            table['baseline_count'] = sums.baseline_count
            self.tables.append((keys, table.loc[table.baseline_count >= self.min_count].reset_index()))
        self.global_mean = aggregate[PRIMARY].sum() / aggregate.baseline_count.sum()
        self.reference_population_rows = int(aggregate.baseline_count.sum())
        self.wait_fill = float(sampled_reference.request_wait_min.median())
        if not np.isfinite(self.wait_fill):
            self.wait_fill = 0.0
        return self

    def transform(self, data):
        result = data.copy().reset_index(drop=True)
        result['baseline_source'] = 'global_reference'
        result['baseline_count'] = 0
        for name in PRIMARY:
            result['baseline_' + name] = np.nan
        for level, (keys, table) in enumerate(self.tables):
            merged = result[keys].merge(table, on=keys, how='left', sort=False, validate='many_to_one')
            mask = result.baseline_price_x.isna() & merged.baseline_price_x.notna()
            cols = ['baseline_' + name for name in PRIMARY] + ['baseline_count']
            result.loc[mask, cols] = merged.loc[mask, cols].to_numpy()
            result.loc[mask, 'baseline_source'] = ['provider_OD', 'OD', 'provider_origin'][level]
        for name in PRIMARY:
            result['baseline_' + name] = result['baseline_' + name].fillna(self.global_mean[name])
        result['request_wait_min'] = result.request_wait_min.fillna(self.wait_fill)
        return reconstruct(result, self.rush_hours)


def reconstruct(frame, rush_hours=(7, 8, 9, 16, 17, 18)):
    result = frame.copy()
    result['speed'] = result.drive_mile_x / (result.drive_time_x / 60.0)
    result['isWeekend'] = (result.weekday >= 5).astype(int)
    result['isRushHour'] = result.pickup_hour.isin(rush_hours).astype(int)
    for actual, deviation in [('price_x', 'DV_price'), ('drive_time_x', 'DV_drive_time'), ('drive_mile_x', 'DV_drive_mile')]:
        result[deviation] = result[actual] - result['baseline_' + actual]
    return result
