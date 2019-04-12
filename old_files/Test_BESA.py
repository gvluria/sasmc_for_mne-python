# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:11:27 2015

@author: gv
"""

from __future__ import print_function, division
import numpy as np
from pysasmc_BESA import *
import BoostRandom as BR
try:
    from termcolor import colored
except:
    pass


seed = 26
BR.setseed(seed)

s_noise = 0.001
NI = 1000
s_q = 4.5
ist_in = 1
ist_fin = 30

folder = '../Input/'
dato = 'Data_2dip_[1848 2881].txt'
path = {'G': folder + 'G_BESA.txt',
        'V': folder + 'V_BESA.txt',
        'N': folder + 'N_BESA.txt',
        'NP': folder + 'NP_BESA.txt',
        'DATA': folder + dato}

V = np.loadtxt(path['V'])
NC = V.shape[0]
G = np.loadtxt(path['G'])
N = np.loadtxt(path['N'], 'int32')
NP = np.loadtxt(path['NP'])

data = np.loadtxt(path['DATA'])
b_ist = data[:, ist_in-1:ist_fin]
Rdata = b_ist.real
Idata = b_ist.imag

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


print("************************************************************")
print("*       Phase 2: Test the methods of SASMC class           *")
print("************************************************************\n")


f1 = SA_SMC(2, 1, 1, s_noise, 4.5, 10)
f1.emp = EmpPdf(2, NC, 5)
try:
    print(colored('2a Create an instance of SASMC and print its EmpPdf:',
                  'red'))
except:
    print('2a Create an instance of SASMC and print its EmpPdf:')
print(f1.emp)

try:
    print(colored('2b Compute priors and likelihoods:', 'red'))
except:
    print('2b Compute priors and likelihoods:')
for part in range(f1.emp.samples.shape[0]):
    f1.emp.samples[part].compute_loglikelihood_unit(Rdata, G, s_noise, 4.5)
    print("Prior " + str(part+1) + ": " + str(f1.emp.samples[part].prior))
    print("Likelihood " + str(part+1) + ": "
          + str(f1.emp.samples[part].loglikelihood_unit) + "\n")

try:
    print(colored('2c Compute exponent:', 'red'))
except:
    print('2c Compute exponent:')
[f1.emp.exponents, f1.emp.logweights, f1.emp.ESS] = \
    f1.emp.compute_exponent(s_noise=s_noise)
print("Exponent: " + str(f1.emp.exponents[2]) + "\n")

try:
    print(colored('2d Simulate a sampling step:', 'red'))
except:
    print('2d Simulate a sampling step:')
f1.emp = f1.emp.sample(2, NC, Rdata, Idata, G, N, NP, s_noise, 5, 10,
                       1/3, 1/20, sigma_q=s_q)
print(f1.emp)

try:
    print(colored('2d Simulate a resampling step:', 'red'))
except:
    print('2d Simulate a resampling step:')
f1.emp = f1.emp.resample()
print(f1.emp)
