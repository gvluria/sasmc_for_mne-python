# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>

# This show the performance of SASMC on the sample, auditory data.

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import (read_forward_solution, pick_types_forward,
                 read_evokeds)
from mne import Dipole as mneDipole
from sasmc import SASMC, estimate_noise_std

#import sys
#sys.path.insert(0, './old_files/')
#import pysmc

# In[]: Step 1. Load and define input parameter
data_path = sample.data_path()
fname_fwd_meeg = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

meg_sensor_type = True

# 1.1. Fwd model
fwd = read_forward_solution(fname_fwd_meeg, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type, eeg=False, ref_meg=False, seeg=False)

# 1.2. (Evoked) Data
evoked = read_evokeds(fname_evoked, condition='Right Auditory', baseline=(None, 0))
evoked = evoked.pick_types(meg=meg_sensor_type, eeg=False, exclude='bads')
# TODO: there should be some check of consistency with fwd

# 1.3. SASMC parameters

# 1.3.1. Time points
ist_max = np.argmax(np.max(evoked.data, axis=0))
ist_in = ist_max - 25
ist_fin = ist_max + 25

print('    Analyzing sample in [{}, {}] - time in [{}, {}] ms'.format(
    ist_in, ist_fin, round(evoked.times[ist_in], 4), round(evoked.times[ist_fin], 4)))

# 1.3.2 Conversion time-sample
time_in = 0.045
time_fin = 0.128
conv_ist_in, conv_ist_fin = evoked.time_as_index([time_in, time_fin], use_rounding=True)

print('    Time [{}, {}] converted in [{}, {}]'.format(
    time_in, time_fin, conv_ist_in, conv_ist_fin))

# Motivation for use_rounding=True
_aux_index = np.atleast_1d(2.9)
print('    Time = {}'.format(_aux_index))
print('    use_rounding=False -> {}'.format(_aux_index.astype(int)[0]))
print('    use_rounding=True -> {}'.format(np.round(_aux_index).astype(int)[0]))

# Motivation for control on integer
print('    Type ist_in = {}'.format(type(ist_in)))
print('    Compare with np.int_ = {}'.format(isinstance(ist_in, np.int_)))
print('    Compare with np.integer = {}'.format(isinstance(ist_in, np.integer)))
if isinstance(ist_in, np.int64):
    aux_type = np.int32
else:
    aux_type = np.int64
print('    Compare with np.int_ part ii = {}'.format(isinstance(ist_in.astype(aux_type), np.int_)))
print('    Compare with np.integer part ii = {}'.format(isinstance(ist_in.astype(aux_type), np.integer)))



# 1.3.2. Sigma q and Sigma noise (Alberto's trick)
data = evoked.data[:, ist_in:ist_fin]

sigma_q = 15 * np.max(abs(data)) / np.max(abs(fwd['sol']['data']))
#sigma_q = None

sigma_noise = 0.2 * np.max(abs(data))
#sigma_noise = estimate_noise_std(evoked.data, 0, 100)

subsample = None

print('    Sigma noise: {0}'.format(sigma_noise))
print('    Sigma q: {0}'.format(sigma_q))
#print(estimate_noise_std(evoked.data, 0, 100))


# In[]: Step 2. Run SASMC
# TODO: print inside our functions should be more 'understandable'
_sasmc = SASMC(fwd, evoked, n_parts=100,  s_noise=sigma_noise, sample_min=ist_in, sample_max=ist_fin,
               s_q=sigma_q, subsample=subsample, verbose=True)
_sasmc.apply_sasmc()

print('    Estimated number of sources: {0}'.format(_sasmc.est_n_dips[-1]))
print('    Estimated sources locations: {0}'.format(_sasmc.est_locs[-1]))

# In[]: Step 3. Save point estimates as mne.Dipole.
times = evoked.times[_sasmc.tmin:_sasmc.tmax+1]
pos = _sasmc.source_space[_sasmc.est_locs[-1]]
num_dip = pos.shape[0]
amplitude = np.array([
     np.linalg.norm(_sasmc.est_q[:, i_dip:3*(i_dip+1)], axis=1) for i_dip in range(num_dip)])
# TODO: Understand/compute orientation and gof (xfit documentation)
#       One possibility: pick and plot one selected time-point.
orientation = np.array([np.array([0, 0, 1]) for i_dip in range(num_dip)])
gof = np.array([0 for i_dip in range(num_dip)])

dips = mneDipole(np.tile(times[0], num_dip), pos, amplitude[:, 0], orientation, gof)

# In[]: Step 4. Visualize
est_cs = _sasmc.est_locs[-1]
est_q = np.zeros((est_cs.shape[0], _sasmc.est_q.shape[0],3))
for i in range(_sasmc.est_q.shape[0]):
    _temp = _sasmc.est_q[i, :].reshape(-1, 3)
    for j in range(est_cs.shape[0]):
        est_q[j, i, :] += _temp[j]

fname_trans = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
subjects_dir = op.join(data_path, 'subjects')


plt.figure()
for _q in est_q:
    plt.plot(np.linalg.norm(_q, axis=1))

for i_dip in range(num_dip):
    dips.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview', idx=i_dip)

#plt.figure()
#plt.plot(evoked.times, evoked.data.T)
plt.show()

# TODO: How to properly deal with time-point selection?
# TODO: How is related the time interval in dipole-fitting tutorial with N100????


