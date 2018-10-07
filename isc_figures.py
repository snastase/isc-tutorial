import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Create a toy ISFC matrix
seeds = np.random.randn(3, 60)
networks = np.vstack((seeds + np.random.randn(3, 60),
                      seeds[0] + np.random.randn(1, 60),
                      seeds[0] + np.random.randn(1, 60),
                      -seeds[2] * .85 + np.random.randn(1, 60),
                      seeds[2] + np.random.randn(1, 60)))
networks = networks[[0, 3, 4, 5, 1, 2, 6], :]
six = np.random.randint(200, 1900, 6)
seven = np.append(six, (8128 - np.sum(six)))
voxels = np.vstack([np.tile(network, (extent, 1)) for network, extent in zip(networks, seven)])

areas = [0] + sorted(np.random.randint(0, 8128, 16))
areas = np.diff(areas).tolist() + [(8128 - areas[-1])]
assert len(areas) == 17
assert sum(areas) == 8128

noise_sources = np.random.randn(7, 60)
structured_noise = np.vstack([np.tile(
    (noise_sources[np.random.choice(range(7)), :] *
     np.random.uniform(.6, 1.0) * np.random.choice([-1, 1, 1, 1])), (extent, 1))
                              for extent in areas])
assert structured_noise.shape == (8128, 60)
voxels = gaussian_filter1d(voxels, 8.0, axis=0)

subject1 = voxels + structured_noise + np.random.randn(8128, 60)*2.3
subject2 = voxels + structured_noise + np.random.randn(8128, 60)*2.3

corrs = np.corrcoef(subject1, subject2)
isfcs = np.tanh(np.mean(np.dstack((np.arctanh(corrs[:8128, 8128:]), 
                                   np.arctanh(corrs[8128:, :8128]))), axis=2))
print(np.amax(isfcs))

plt.matshow(isfcs/2, cmap="RdYlBu_r", vmin=-.3, vmax=.3)
plt.axis('off')

plt.savefig('isfc_matrix8.svg', dpi=300, bbox_inches='tight',
            pad_inches=0, transparent=True)


# Create a toy time series
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import zscore

ts = np.random.randn(60)
ts_a = ts + np.random.randn(60) * .25
ts_b = ts + np.random.randn(60) * .25
smooth_a = zscore(gaussian_filter1d(ts_a, .7))
smooth_b = zscore(gaussian_filter1d(ts_b, .7))

plt.figure(figsize=(3.5, 1))
plt.plot(smooth_a, color='maroon', linewidth=2)
plt.xticks([], [])
plt.yticks([], [])
plt.xlim(-1.5, 60.5)
plt.ylim(-3, 3)
plt.savefig('time_series3b.svg', dpi=300, transparent=True)


# Create a toy response pattern
pattern1 = np.random.randn(3, 3)
pattern2 = pattern1 + np.random.randn(3, 3) * .5

#smooth_pattern1 = gaussian_filter1d(pattern1, .55, axis=1)
#smooth_pattern2 = gaussian_filter1d(pattern2, .55, axis=1)

pattern1 = np.random.randn(9*3, 1)
plt.matshow(pattern1, cmap="RdYlBu_r", vmin=-1.6, vmax=1.6)
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('pattern4b.svg', dpi=300, transparent=True)


# Create toy time-point RDM
from scipy.spatial.distance import pdist, squareform
patterns = np.random.randn(12, 9)
eleven = np.random.randint(1, 10, 11)
twelve = np.append(eleven, (60 - np.sum(eleven)))
data = np.vstack([np.tile(pattern, (extent, 1)) for pattern, extent in zip(patterns, twelve)])
smooth_data = gaussian_filter1d(data, 2.0, axis=0)
noisy_data1 = smooth_data + np.random.randn(60, 9) * .75
noisy_data2 = noisy_data1 + np.random.randn(60, 9) * .5
rdm1 = squareform(pdist(noisy_data1, 'correlation'))
rdm2 = squareform(pdist(noisy_data2, 'correlation'))
print(np.amin(rdm1), np.amax(rdm1))

#from matplotlib import ticker
plt.matshow(rdm1, cmap="RdYlBu_r", vmin=0, vmax=2)
#cb = plt.colorbar()
#tick_locator = ticker.MaxNLocator(nbins=4)
#cb.locator = tick_locator
#cb.update_ticks()
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('timepoint_rdm4b.svg', dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0)


# Create toy intersubject time-point RDM
from scipy.spatial.distance import pdist, squareform
patterns = np.random.randn(12, 9)
eleven = np.random.randint(1, 10, 11)
twelve = np.append(eleven, (60 - np.sum(eleven)))
data = np.vstack([np.tile(pattern, (extent, 1)) for pattern, extent in zip(patterns, twelve)])
smooth_data = gaussian_filter1d(data, 2.0, axis=0)

s1_data = smooth_data + np.random.randn(60, 9) * .75
s2_data = smooth_data + np.random.randn(60, 9) * .75

corrs = np.corrcoef(s1_data, s2_data)
rdm = np.tanh(np.mean(np.dstack((np.arctanh(corrs[:60, 60:]),
                                 np.arctanh(corrs[60:, :60]))), axis=2))
print(rdm.shape, np.amin(rdm), np.amax(rdm))

#from matplotlib import ticker
plt.matshow(rdm, cmap="RdYlBu_r", vmin=-1, vmax=1)
#cb = plt.colorbar()
#tick_locator = ticker.MaxNLocator(nbins=4)
#cb.locator = tick_locator
#cb.update_ticks()
plt.xticks([], [])
plt.yticks([], [])

plt.savefig('isc_timepoint_rdm6.svg', dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0)