# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#
# License: BSD (3-clause)


import mne
from os import path as op
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import sample
from sasmc import Dipole, Particle, EmpPdf, SASMC
try:
    from termcolor import colored
except:
    pass


seed = 26
np.random.seed(seed)

# Load and define input parameter
data_path = sample.data_path()
fname_fwd_meeg = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
meg_sensor_type = True

# Fwd model
fwd = mne.read_forward_solution(fname_fwd_meeg, exclude='bads')
fwd = mne.pick_types_forward(fwd, meg=meg_sensor_type, eeg=False, ref_meg=False, seeg=False)

# (Evoked) Data
_aux_evoked = mne.read_evokeds(fname_evoked, condition='Right Auditory', baseline=(None, 0))
_aux_evoked = _aux_evoked.pick_types(meg=meg_sensor_type, eeg=False, exclude='bads')
evoked = _aux_evoked.copy()



s_noise = 0.001
NI = 1000
s_q = 4.5
tmin = 1
tmax = 30

V = fwd['source_rr']
NC = V.shape[0]
G = fwd['sol']['data']

Rdata = evoked.data.real
Idata = evoked.data.imag


dip = Dipole(5)

assert(dip.loc == 5)

print(dip)


print("************************************************************")
print("*       Phase 1: Test the methods of Particle class        *")
print("************************************************************")

print("Fixed seed: " + str(seed) + '\n')
# Create a particle
try:
    print(colored('1a Create a particle:', 'red'))
except:
    print('1a Create a particle:')
p1 = Particle(NC, 5)
print(p1)

# Add a dipole
try:
    print(colored("1b Add a dipole:", 'red'))
except:
    print("1b Add a dipole:")
p1.add_dipole(NC)
print(p1)

# Compute prior
try:
    print(colored("1c Compute the prior:", 'red'))
except:
    print("1c Compute the prior:")
try:
    print(colored("Prior =  ", 'green') + str(p1.compute_prior(5)))
except:
    print("Prior =  " + str(p1.compute_prior(5)))

# Compute likelihood
try:
    print(colored("1d Compute the likelihood:", 'red'))
except:
    print("1d Compute the likelihood:")
lkl = p1.compute_loglikelihood_unit(Rdata, G, s_noise, s_q)
try:
    print(colored("Likelihood =  ", 'green') + str(lkl))
except:
    print("Likelihood =  " + str(lkl))

try:
    print(colored("1e Remove dipoles:", 'red'))
except:
    print("1e Remove dipoles:")
p1.remove_dipole(0)
p1.remove_dipole(2)
print(p1)

try:
    print(colored("1f Evolve dipole number:", 'red'))
except:
    print("1f Evolve dipole number:")
p1 = p1.birth_or_death(NC, Rdata, Idata, G, 10, 1e-05, s_noise, 5, sigma_q=s_q)
print(p1)
p1 = p1.birth_or_death(NC, Rdata, Idata, G, 10, 1e-05, s_noise, 5, sigma_q=s_q)
print(p1)
p1 = p1.birth_or_death(NC, Rdata, Idata, G, 10, 1e-05, s_noise, 5, sigma_q=s_q)
print(p1)
p1 = p1.birth_or_death(NC, Rdata, Idata, G, 10, 1e-05, s_noise, 5, sigma_q=s_q)
print(p1)

try:
    print(colored("1g Evolve dipoles location:", 'red'))
except:
    print("1g Evolve dipoles location:")
p1 = p1.evol_c_single(0, N, NP, Rdata, Idata, G, 1e-05, s_noise, 5,
                      sigma_q=s_q)
print(p1)
p1 = p1.evol_c_single(0, N, NP, Rdata, Idata, G, 1e-05, s_noise, 5,
                      sigma_q=s_q)
print(p1)
p1 = p1.evol_c_single(0, N, NP, Rdata, Idata, G, 1e-05, s_noise, 5,
                      sigma_q=s_q)
print(p1)