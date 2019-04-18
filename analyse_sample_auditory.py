# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>

# This show the performance of SASMC on the sample, auditory data.

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import (read_forward_solution, pick_types_forward,
                 read_evokeds, Dipole)

import sys
sys.path.insert(0, './old_files/')
import pysmc

# In[]: Step 1. Load and define input parameter
data_path = sample.data_path()
fname_fwd_meeg = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

meg_sensor_type = True

# 1.1. Fwd model
fwd = read_forward_solution(fname_fwd_meeg, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type, eeg=False, ref_meg=False, seeg=False)

# 1.2. (Evoked) Data
evoked = read_evokeds(fname_evoked, condition='Right Auditory',
                           baseline=(None, 0))
evoked = evoked.pick_types(meg=meg_sensor_type, eeg=False, exclude='bads')
# TODO: there should be some check of consistency with fwd

# 1.3. SASMC parameters

# 1.3.1. Time points
ist_max = np.argmax(np.max(evoked.data, axis=0))
ist_in = ist_max - 25
ist_fin = ist_max + 25

print('Analyzing sample in [{}, {}] - time in [{}, {}] ms'.format(
    ist_in, ist_fin, evoked.times[ist_in], evoked.times[ist_fin]))

# 1.3.2 Conversion time-sample
time_in = 0.045
time_fin = 0.128
conv_ist_in, conv_ist_fin = evoked.time_as_index([time_in, time_fin], use_rounding=True)

print('Time [{}, {}] converted in [{}, {}]'.format(
    time_in, time_fin, conv_ist_in, conv_ist_fin))

# Motivation for use_rounding=True
_aux_index = np.atleast_1d(2.9)
print('Time = {}'.format(_aux_index))
print('use_rounding=False -> {}'.format(_aux_index.astype(int)[0]))
print('use_true=False -> {}'.format(np.round(_aux_index).astype(int)[0]))

# 1.3.2. Sigma q and Sigma noise (Alberto's trick)
data = evoked.data[:, ist_in:ist_fin]
sigma_q = 15 * np.max(abs(data)) / np.max(abs(fwd['sol']['data']))
sigma_noise = 0.2 * np.max(abs(data))

# In[]: Step 2. Run SASMC
# TODO: print inside our functions should be more 'understandable'
filt = pysmc.SA_SMC(fwd, evoked, s_noise=sigma_noise, time_in=ist_in,
                     time_fin=ist_fin, s_q=sigma_q)
filt.run_filter()

# In[]: Step 3. Save point estimates as mne.Dipole.
times = evoked.times[filt.ist_in:filt.ist_fin+1]
pos = filt.source_space[filt.est_locs[-1]]
num_dip = pos.shape[0]
amplitude = np.array([
     np.linalg.norm(filt.est_q[:, i_dip:3*(i_dip+1)], axis=1) for i_dip in range(num_dip)])
# TODO: Understand/compute orientation and gof (xfit documentation)
#       One possibility: pick and plot one selected time-point.
orientation = np.array([np.array([0, 0, 1]) for i_dip in range(num_dip)])
gof = np.array([0 for i_dip in range(num_dip)])

dips = Dipole(np.tile(times[0], num_dip), pos, amplitude[:, 0], orientation, gof)

# In[]: Step 4. Visualize
fname_trans = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_raw-trans.fif')
subjects_dir = op.join(data_path, 'subjects')
for i_dip in range(num_dip):
     dips.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview', idx=i_dip)


plt.figure()
plt.plot(evoked.times, evoked.data.T)

# TODO: How to properly deal with time-point selection?
# TODO: How is related the time interval in dipole-fitting tutorial with N100????


