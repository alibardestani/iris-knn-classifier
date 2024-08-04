import numpy as np
from scipy import stats

irises = np.load('irises.npy')

types = np.load('types.npy')

new_irises = np.load('new_irises.npy')

n, m = len(irises), len(new_irises)

def calc_no_loop(new_points, points):
    return np.sum(np.square(new_points[:, np.newaxis, :] - points[np.newaxis, :, :]), axis=2)

d = calc_no_loop(new_irises, irises)

k = 10
k_nearest = np.argpartition(d, k, axis=1)[:, :k]

k_nearest_types = types[k_nearest]

predicted_types = stats.mode(k_nearest_types, axis=1).mode.reshape(m)

new_types = np.load('new_types.npy')
accuracy = np.mean(predicted_types == new_types)
print('Accuracy:', accuracy)
