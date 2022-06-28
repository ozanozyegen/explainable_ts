
import numpy as np
from math import ceil
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, stride=1, step=1, single_step=False):
    """ Train test split of multivariate data for Neural Network training """
    data, labels = [], []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index, stride):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


def gen_covariates(times, num_covariates=3, standardize=False):
    """Return standardized weekday, month and week_of_month
    -1 added for month and weekofmonth to start from 0
    """
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday()
        covariates[i, 1] = input_time.month - 1
        covariates[i, 2] = week_of_month(input_time) - 1
    if standardize:
        for i in range(num_covariates):
            covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates[:, :num_covariates]
