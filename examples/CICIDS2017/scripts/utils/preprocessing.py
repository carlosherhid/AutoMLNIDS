"""Input transformations compared against the near-zero rescaling (see apply_preprocessing)."""
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler

LOG_MINMAX_EPS = 1e-9


def apply_preprocessing(train_data, test_data, val_data, method):
    """Alternative input transformations to compare against --scale_close_to_zero.

    The CSVs are already Min-Max normalized to [0, 1] (dataset_creator.py), which squeezes
    heavy-tailed features into a tiny range near zero. Each method below spreads those
    values out in a different way; statistics are fitted on the training split only.
      standard : z-score (StandardScaler)
      logminmax: log10(x + eps) mapped back to [0, 1] (eps = 1e-9), a stateless log scaling
      quantile : QuantileTransformer to a uniform [0, 1] distribution (rank-based)
    'none' leaves the data untouched."""
    if method == 'none':
        return train_data, test_data, val_data

    feature_columns = [c for c in train_data.columns if c not in ('ID', 'Label')]

    if method == 'logminmax':
        low = np.log10(LOG_MINMAX_EPS)

        def transform(values):
            return (np.log10(np.clip(values, 0, None) + LOG_MINMAX_EPS) - low) / (-low)
    elif method in ('standard', 'quantile'):
        transformer = StandardScaler() if method == 'standard' else QuantileTransformer(
            n_quantiles=1000, output_distribution='uniform', subsample=100_000, random_state=0)
        transformer.fit(train_data[feature_columns].values)
        transform = transformer.transform
    else:
        raise ValueError(f'Unknown preprocessing method: {method}')

    processed = []
    for data in (train_data, test_data, val_data):
        data = data.copy()
        data[feature_columns] = transform(data[feature_columns].values.astype(np.float64))
        processed.append(data)
    return tuple(processed)
