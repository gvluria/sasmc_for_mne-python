"""
============================
Compute SASMC on evoked data
============================

Compute and visualize SASMC solution on the auditory sample dataset.
"""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#
# License: BSD (3-clause)

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import read_forward_solution, pick_types_forward, read_evokeds
from mne import Dipole as mneDipole
from mne.label import _n_colors

from sasmc import Sesame  # , estimate_noise_std
from mayavi import mlab

data_path = sample.data_path()
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
subject = 'sample'
subjects_dir = op.join(data_path, 'subjects')

###############################################################################
# Load fwd model and evoked data
meg_sensor_type = True  # All meg sensors will be included

# Fwd model
fwd = read_forward_solution(fname_fwd, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type, eeg=False, ref_meg=False)

# Evoked Data
condition = 'Left Auditory'
evoked = read_evokeds(fname_evoked, condition=condition, baseline=(None, 0))
evoked = evoked.pick_types(meg=meg_sensor_type, eeg=False, exclude='bads')
# TODO: there should be some check of consistency with fwd

###############################################################################
# Define SASMC parameters

# Time window
time_in = 0.055
time_fin = 0.135
subsample = None
sample_min, sample_max = evoked.time_as_index([time_in, time_fin],
                                              use_rounding=True)
fig = evoked.plot(show=False)
for ax in fig.get_axes():
    ax.axvline(time_in, color='r', linewidth=2.0)
    ax.axvline(time_fin, color='r', linewidth=2.0)
plt.show()

# Standard deviation of dipole-moment prior and noise distribution
# (Alberto's Trick)
data = evoked.data[:, sample_min:sample_max+1]
sigma_q = 15 * np.max(abs(data)) / np.max(abs(fwd['sol']['data']))
# sigma_q = None
# sigma_q = 4.08e-08
sigma_noise = 0.2 * np.max(abs(data))
# sigma_noise = estimate_noise_std(evoked.data, 0, 100)
# sigma_noise = 3.2e-12
print('    Sigma noise: {0}'.format(sigma_noise))
print('    Sigma q: {0}'.format(sigma_q))

###############################################################################
# Run SASMC
# TODO: print inside our functions should be more 'understandable'
_sesame = Sesame(fwd, evoked, n_parts=10, s_noise=sigma_noise,
               sample_min=sample_min, sample_max=sample_max,
               s_q=sigma_q, subsample=subsample, verbose=False)
_sesame.apply_sesame()

print('    Estimated number of sources: {0}'.format(_sesame.est_n_dips[-1]))
print('    Estimated source locations: {0}'.format(_sesame.est_locs[-1]))

###############################################################################
# Visualize amplitude of the estimated sources as function of time
est_n_dips = _sesame.est_n_dips[-1]
est_locs = _sesame.est_locs[-1]

times = evoked.times[_sesame.s_min:_sesame.s_max+1]
amplitude = np.array([np.linalg.norm(_sesame.est_q[:, i_d:3 * (i_d + 1)],
                                     axis=1) for i_d in range(est_n_dips)])
colors = _n_colors(est_n_dips)
plt.figure()
for idx, amp in enumerate(amplitude):
    plt.plot(times, amp, color=colors[idx], linewidth=2)
plt.title('Source time-courses')
plt.show()
# TODO: add (i) x/y labels (ii) title

###############################################################################
# Visualize intensity measure and estimated source locations on inflated brain
stc = _sesame.to_stc(subject)
clim = dict(kind='value', lims=[1e-4, 1e-1, 1])
brain = stc.plot(subject, surface='inflated', hemi='both', clim=clim,
                 time_label=' ', subjects_dir=subjects_dir)
nv_lh = stc.vertices[0].shape[0]
for idx, loc in enumerate(est_locs):
    if loc < nv_lh:
        brain.add_foci(stc.vertices[0][loc], coords_as_verts=True,
                       hemi='lh', color=colors[idx], scale_factor=0.3)
    else:
       brain.add_foci(stc.vertices[1][loc-nv_lh], coords_as_verts=True,
                      hemi='rh', color=colors[idx], scale_factor=0.3)
mlab.show()

# clim = dict(kind='value', lims=[1e-4, 1e-1, 1])
#
# brain_lh = _sasmc.plot_itensity_measure(subject,  subjects_dir=subjects_dir, hemi='lh', clim=clim)
# brain_rh = _sasmc.plot_itensity_measure(subject,  subjects_dir=subjects_dir, hemi='rh', clim=clim)

#plt.show()

###############################################################################
# Da qui in poi materiale da riguardare (eventualmente da eliminare)
# _sasmc.plot_itensity_measure(subject, subjects_dir, hemi='rh', clim=clim)
# _sasmc.plot_itensity_measure(subject, subjects_dir, hemi='lh', clim=clim)
# mlab.show()
times = np.tile(evoked.times[_sesame.s_min:_sesame.s_max+1], (2, 1))
pos = _sesame.source_space[_sesame.est_locs[-1]]
num_dip = pos.shape[0]
amplitude = np.array([
      np.linalg.norm(_sesame.est_q[:, i_dip:3*(i_dip+1)], axis=1) for i_dip in range(num_dip)])
# TODO: Understand/compute orientation and gof (xfit documentation)
#       One possibility: pick and plot one selected time-point.
orientation = np.array([np.array([0, 0, 1]) for i_dip in range(num_dip)])
gof = np.array([0 for i_dip in range(num_dip)])

dips = mneDipole(np.tile(times[0], num_dip), pos, amplitude[:, 0], orientation, gof)
fname_trans = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')

for i_dip in range(num_dip):
    dips.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview', idx=i_dip)


plt.show()

# # In[]: Step 4. Visualize
# est_cs = _sasmc.est_locs[-1]
# est_q = np.zeros((est_cs.shape[0], _sasmc.est_q.shape[0], 3))
# for i in range(_sasmc.est_q.shape[0]):
#     _temp = _sasmc.est_q[i, :].reshape(-1, 3)
#     for j in range(est_cs.shape[0]):
#         est_q[j, i, :] += _temp[j]
#
# fname_trans = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
# # TODO: How to properly deal with time-point selection?
# # TODO: How is related the time interval in dipole-fitting tutorial with N100????
# mlab.show()
# Motivation for control on integer
print('    Type ist_in = {}'.format(type(sample_min)))
print('    Compare with np.int_ = {}'.format(isinstance(sample_min, np.int_)))
print('    Compare with np.integer = {}'.format(isinstance(sample_min, np.integer)))
if isinstance(sample_min, np.int64):
    aux_type = np.int32
else:
    aux_type = np.int64
print('    Compare with np.int_ part ii = {}'.format(isinstance(sample_min.astype(aux_type), np.int_)))
print('    Compare with np.integer part ii = {}'.format(isinstance(sample_min.astype(aux_type), np.integer)))
