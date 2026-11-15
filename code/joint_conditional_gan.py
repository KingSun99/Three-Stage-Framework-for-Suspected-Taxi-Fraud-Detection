import importlib.metadata
import time
import numpy as np
from ctgan import CTGAN
from ctgan.data_sampler import DataSampler

class TrainingProgress:

    def fit(self, train_data, discrete_columns=(), epochs=None):
        self._progress_steps = max(len(train_data) // self._batch_size, 1)
        self._progress_calls = 0
        self._progress_started = time.perf_counter()
        self._progress_active = True
        try:
            return super().fit(train_data, discrete_columns, epochs)
        finally:
            self._progress_active = False

    def _cond_loss(self, data, c, m):
        value = super()._cond_loss(data, c, m)
        if getattr(self, '_progress_active', False):
            self._progress_calls += 1
            if self._progress_calls % (50 * self._progress_steps) == 0:
                print(f'GAN EPOCH {self._progress_calls // self._progress_steps}/{self._epochs}; seconds={time.perf_counter() - self._progress_started:.1f}', flush=True)
        return value


class LabelConditionedSampler(DataSampler):
    """Select the label column for every training condition, preserving log frequencies."""

    def __init__(self, original, discrete_id, values):
        self.__dict__.update(original.__dict__)
        self.label_discrete_id = int(discrete_id)
        self.label_values = np.asarray(values, dtype=int)
        self.label_start = int(self._discrete_column_cond_st[discrete_id])
        self.label_width = int(self._discrete_column_n_category[discrete_id])
        self.training_condition_counts = np.zeros(self.label_width, dtype=np.int64)
        if len(values) != self.label_width or len(set(values)) != self.label_width:
            raise ValueError('Ambiguous label encoding')

    def sample_condvec(self, batch):
        columns = np.full(batch, self.label_discrete_id, dtype=np.int32)
        categories = self._random_choice_prob_index(columns)
        cond = np.zeros((batch, self._n_categories), dtype=np.float32)
        mask = np.zeros((batch, self._n_discrete_columns), dtype=np.float32)
        cond[np.arange(batch), self.label_start + categories] = 1
        mask[:, self.label_discrete_id] = 1
        self.training_condition_counts += np.bincount(categories, minlength=self.label_width)
        return (cond, mask, columns, categories)

    def sample_original_condvec(self, batch):
        counts = np.array([len(rows) for rows in self._rid_by_cat_cols[self.label_discrete_id]], dtype=float)
        categories = np.random.choice(self.label_width, batch, p=counts / counts.sum())
        result = np.zeros((batch, self._n_categories), dtype=np.float32)
        result[np.arange(batch), self.label_start + categories] = 1
        return result

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        start = int(self._discrete_column_cond_st[condition_info['discrete_column_id']])
        result = np.zeros((batch, self._n_categories), dtype=np.float32)
        result[:, start + condition_info['value_id']] = 1
        return result


class JointConditionalCTGAN(TrainingProgress, CTGAN):

    def __init__(self, label_column='label', **kwargs):
        if importlib.metadata.version('ctgan') != '0.10.0':
            raise RuntimeError('Verified joint implementation requires ctgan==0.10.0')
        self.label_column = label_column
        self._sampler = None
        super().__init__(**kwargs)

    @property
    def _data_sampler(self):
        return self._sampler

    @_data_sampler.setter
    def _data_sampler(self, value):
        if value is not None and (not isinstance(value, LabelConditionedSampler)):
            info = self._transformer.convert_column_name_value_to_id(self.label_column, 0)
            transform = next((item for item in self._transformer._column_transform_info_list if item.column_name == self.label_column))
            values = np.asarray(transform.transform.dummies, dtype=float)
            if not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
                raise ValueError('Class labels must be finite integers')
            value = LabelConditionedSampler(value, info['discrete_column_id'], values.astype(int))
        self._sampler = value

    def fit(self, train_data, discrete_columns=(), epochs=None):
        if self.label_column not in discrete_columns:
            raise ValueError('Joint model requires the label as a discrete column')
        if set(train_data[self.label_column].unique()) != set(range(5)):
            raise ValueError('Joint training must include all five frozen classes')
        return super().fit(train_data, discrete_columns, epochs)
