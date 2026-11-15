import numpy as np
from data import MODEL_FEATURES

BEHAVIOR_COLUMNS = ['log_distance_ratio', 'log_duration_ratio', 'log_fare_ratio', 'log_wait_time', 'log_speed']


def behavior_values(frame):
    """Use the same five coordinates as the frozen clustering experiment."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.column_stack([np.log(frame.drive_mile_x.to_numpy(float)) - np.log(frame.baseline_drive_mile_x.to_numpy(float)), np.log(frame.drive_time_x.to_numpy(float)) - np.log(frame.baseline_drive_time_x.to_numpy(float)), np.log(frame.price_x.to_numpy(float)) - np.log(frame.baseline_price_x.to_numpy(float)), np.log1p(frame.request_wait_min.to_numpy(float)), np.log(60.0 * frame.drive_mile_x.to_numpy(float) / frame.drive_time_x.to_numpy(float))])
