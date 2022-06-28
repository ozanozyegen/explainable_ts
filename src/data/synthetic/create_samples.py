import numpy as np

def create_dataset_feat_ranking(total_length=10000, seed=0, num_series=10,
    remove_first_n=10):
    """

    """
    single_series_length = total_length // num_series
    np.random.seed(seed)
    time = np.arange(50)
    values = time * 5
    # Uniform random noise [0, 1]
    x2 = np.random.rand(total_length - (remove_first_n * num_series)).tolist()
    # y_noise = np.random.rand(total_length) * 6 - 3 # Random noise [-5, +5]
    # Gaussian Random noise [-30, +30]
    x1_noise = np.random.randn(total_length)

    c1_all = np.random.randint(0, 5, size=total_length)
    c1_all = [str(c1) for c1 in c1_all]

    data = {
        'c1': [],
        'x1': [],
        'x2': x2,
        'y': [],
        'time_idx': [],
        'series_id': []
    }
    for ts_id in range(num_series):
        series_id = str(ts_id)
        y_init = 80 #+ np.random.randint(low=-10, high=10)
        for i in range(remove_first_n, single_series_length):
            i = i + ts_id * single_series_length
            x1 = values[i % len(values)] #+ x1_noise[i]
            c1 = c1_all[i]

            y = y_init  # + y_noise[i] # Heart rate start

            # x1 some continous measure
            # only has impact if larger than 200 or less than 50 in the last timestep
            if x1 > 200:
                y += 120
            elif x1 < 50:
                y -= 120
            # C1 lagged impact
            if c1_all[i - 9] == '0':
                y += 100
            if c1_all[i - 4] == '1':
                y += 60
            # C1 instant impact
            if c1 == '3':
                y -= 50
            elif c1 == '4':
                y += 80

            data['c1'].append(c1)
            data['x1'].append(x1)
            data['y'].append(y)
            data['time_idx'].append(i)
            data['series_id'].append(series_id)

    return data
