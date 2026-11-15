from types import MethodType
import numpy as np
import pandas as pd
from ctgan.data_transformer import DataTransformer

def _numeric_inverse_discrete(transformer, info, data):
    categories = transformer._exact_numeric_categories.get(info.column_name)
    if categories is None:
        return transformer._original_inverse_discrete(info, data)
    return pd.Series(categories[np.argmax(data, axis=1)])


def enable_exact_numeric_decode(model):
    transformer = model._transformer
    if not isinstance(transformer, DataTransformer):
        return 0
    if hasattr(transformer, '_original_inverse_discrete'):
        return len(transformer._exact_numeric_categories)
    categories = {}
    for info in transformer._column_transform_info_list:
        if info.column_type != 'discrete':
            continue
        values = np.asarray(info.transform.dummies)
        if values.ndim == 1 and values.dtype.kind in 'biuf' and np.isfinite(values).all() and (len(values) == info.output_dimensions):
            categories[info.column_name] = values
    transformer._exact_numeric_categories = categories
    transformer._original_inverse_discrete = transformer._inverse_transform_discrete
    transformer._inverse_transform_discrete = MethodType(_numeric_inverse_discrete, transformer)
    return len(categories)
