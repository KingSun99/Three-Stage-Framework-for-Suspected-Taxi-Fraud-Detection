from skopt.space import Integer, Real

def search_space():
    return [Integer(30, 200, name='n_estimators'), Real(0.01, 2.0, name='learning_rate'), Integer(1, 30, name='max_depth'), Integer(1, 20, name='min_child_weight'), Real(0.0, 10.0, name='gamma'), Real(0.01, 1.0, name='subsample'), Real(0.01, 1.0, name='colsample_bytree'), Real(0.01, 10.0, name='reg_lambda'), Real(0.01, 10.0, name='reg_alpha')]


def params_from_values(dimensions, values):
    result = {}
    for dimension, value in zip(dimensions, values):
        result[dimension.name] = int(value) if isinstance(dimension, Integer) else float(value)
    return result
