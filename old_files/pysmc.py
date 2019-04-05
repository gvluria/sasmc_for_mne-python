# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva     <sommariva@dima.unige.it>

"""
This file provides the implementation of the two algorithms SMC [1] and SASMC
[2].

References
----------
[1] A. Sorrentino, G. Luria, and R. Aramini. Bayesian multi-dipole modeling of
a single topography in MEG by adaptive Sequential Monte-Carlo samplers.
Inverse Problems, 30:045010, 2014.
[2] S. Sommariva and A. Sorrentino. Sequential Monte Carlo samplers for
semi--linear inverse problems and application to magnetoencephalography.
Inverse Problems, 30:114020, 2014.
"""

from __future__ import division, print_function
import math
import copy
import itertools
import scipy.io
import scipy.spatial.distance
import scipy.linalg as slin
import time
import mne
import numpy as np
from scipy import stats

"""
GLOBAL VARIABLES

q_birth  : float
    The probability to propose the addition (birth) of
    a dipole in a Particle instance.

q_death : float
    The probability to propose the removal (death) of
    a dipole from a Particle instance.

gamma_high : float
    Parameter for the adaptive choice of the the sequence
    of artificial distributions

gamma_low : float
    Parameter for the adaptive choice of the the sequence
    of artificial distributions

delta_min : float
    Parameter for the adaptive choice of the the sequence
    of artificial distributions, corresponding to the
    minimum achievable increase of the likelihood exponent

delta_max : float
    Parameter for the adaptive choice of the the sequence
    of artificial distributions, corresponding to the
    maximum achievable increase of the likelihood exponent

q_in_range : float
    Parameter of the prior distribution of the dipole strength

s_orient : float
    Parameter of the Gaussian transition kernel for the
    source orientation

_lam : float
    Parameter of the Poisson probability distribution used for
    determining the number of dipoles in a particle.

_n_dip_max : int
    Maximum number of dipoles allowed in a Particle instance
"""
q_birth = 1 / 3
q_death = 1 / 20
gamma_high = 0.99
gamma_low = 0.9
delta_min = 1e-05
delta_max = 0.1
q_in_range = 3
s_orient = np.pi/48
_lam = 0.25
_n_dip_max = 10

"""
CLASSES
"""


class Dipole(object):
    """Single current dipole class for semi-analytic SMC algorithm

    Parameters
    ----------
    loc : int
        The dipole location (as an index of a brain grid).
    """

    def __init__(self, loc):
        self.loc = loc

    def __repr__(self):
        s = "location : %s" % str(self.loc)
        return "<Dipole  |  %s>" % s


class DipoleSMC(Dipole):
    """Single current dipole class for SMC algorithm

    Parameters
    ----------
    loc : int
        The dipole location (as an index of a brain grid).
    zeta : float
        The height of the dipole moment.
    phi : float
        The azimuth of the dipole moment.
    re_q : float
        The real part of the dipole strength.
    im_q : float
        The imaginary part of the dipole strength.
    """

    def __init__(self, loc, zeta, phi, re_q, im_q):
        super(DipoleSMC, self).__init__(loc)
        self.zeta = zeta
        self.phi = phi
        self.re_q = re_q
        self.im_q = im_q

    def __repr__(self):
        s = "location : %s" % str(self.loc)
        s += ", z : %s" % str(np.float32(self.zeta))
        s += ", phi : %s" % str(np.float32(self.phi))
        s += ", Re(q) : %s" % str(np.float32(self.re_q))
        s += ", Im(q) : %s" % str(np.float32(self.im_q))
        return "<Dipole  |  %s>" % s


class Particle(object):
    """Particle class for the semi-analytic SMC algorithm.
    Used to store a single particle of an empirical pdf.

    Parameters
    ----------
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.

    Attributes
    ----------
    n_dips : int
        The number of dipoles in the particle.
    dipoles : array of instances of Dipole, shape(n_dips,)
        The particle's dipoles.
    loglikelihood_unit : float
        The logarithm of the marginal likelihood, evaluated in the particle.
    prior : float
        The prior pdf, evaluated in the particle.
    """

    def __init__(self, n_verts, lam):
        """Initialization: the initial number of dipoles is Poisson
           distribuited; the initial locations are uniformly distribuited
           within the brain grid points, with no dipoles in the same position.
        """
        self.n_dips = 0
        self.dipoles = np.array([])
        self.prior = None
        self.loglikelihood_unit = None

        self.add_dipole(n_verts, np.random.poisson(lam))
        self.compute_prior(lam)

    def __repr__(self):
        s = "n_dips : %s" % self.n_dips
        for i in range(self.n_dips):
            s += ", dipole " + str(i+1) + " : " + str(self.dipoles[i])
        s += ", prior : %s" % str(self.prior)
        return "<Particle  |  %s>" % s

    def add_dipole(self, n_verts, num_dip=1):
        """ Add new dipoles to the particle.

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        num_dip : int
            The number of dipoles to add.
        """

        new_locs = np.random.randint(0, n_verts, num_dip)

        for loc in new_locs:
            while loc in [self.dipoles[dip].loc
                          for dip in range(self.n_dips)]:
                loc = np.random.randint(0, n_verts)

            self.dipoles = np.append(self.dipoles, Dipole(loc))
            self.n_dips += 1

    def remove_dipole(self, diprip):
        """ Remove a dipole from the particle.

        Parameters
        ----------
        diprip : int
            The index representing the dipoles array entry to be removed.
        """

        self.dipoles = np.delete(self.dipoles, diprip)
        self.n_dips -= 1

    def compute_loglikelihood_unit(self, r_data, lead_field, s_noise, s_q):
        """ Evaluates the logarithm of the likelihood function in the present particle.

        Parameters
        ----------
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        s_noise : float
            The standard deviation of the noise distribution.
        s_q : float
            The standard deviation of the prior pdf of the dipole moment.

        Returns
        -------
        loglikelihood_unit : float
            The logarithm of the likelihood function in the present particle.
        """

        [n_sens, n_ist] = r_data.shape

        # Step 1: compute variance of the likelihood.
        if self.n_dips == 0:
            sigma = np.eye(n_sens)
        else:
            # 1a: compute the leadfield of the particle
            idx = np.ravel([[3*self.dipoles[idip].loc,
                             3*self.dipoles[idip].loc+1,
                             3*self.dipoles[idip].loc+2]
                            for idip in range(self.n_dips)])
            Gc = lead_field[:, idx]
            # 1b: compute the variance
            sigma = (s_q / s_noise)**2 * np.dot(Gc, np.transpose(Gc)) + \
                np.eye(n_sens)

        # Step 2: compute inverse and determinant of the variance
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)

        # Step 3: compute the log_likelihood
        self.loglikelihood_unit = - (n_ist * s_noise**2) * math.log(det_sigma)
        for ist in range(n_ist):
            self.loglikelihood_unit -= \
                np.transpose(r_data[:, ist]).dot(inv_sigma).dot(r_data[:, ist])
        return self.loglikelihood_unit

    def compute_prior(self, lam):
        """Evaluate the prior pdf in the present particle.

        Parameters
        ----------
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Returns
        -------
        prior : float
            The prior pdf evaluated in the present particle.
        """
        self.prior = 1/math.factorial(self.n_dips) * np.exp(-lam) *\
            (lam**self.n_dips)
        return self.prior

    def evol_n_dips(self, n_verts, r_data, i_data, lead_field, N_dip_max,
                    lklh_exp, s_noise, lam, **kw_parameters):
        """Perform a Reversible Jump Markov Chain Monte Carlo step in order
           to explore the "number of sources" component of the state space.
           Recall that we are working in a variable dimension model.

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        N_dip_max : int
            The maximum number of dipoles allowed in the particle.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SASMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        SASMC additional parameters
        ---------------------------
        s_q : float
            standard deviation of the prior of the dipole moment.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """

        prop_part = copy.deepcopy(self)
        birth_death = np.random.uniform(1e-16, 1)

        if not hasattr(self, 'loglikelihood_unit'):
            if 'sigma_q' in kw_parameters:
                self.compute_loglikelihood_unit(r_data, lead_field, s_noise,
                                                kw_parameters['sigma_q'])
            else:
                self.compute_loglikelihood_unit(r_data, i_data, lead_field)

        if birth_death < q_birth and prop_part.n_dips < N_dip_max:
            if 'sigma_q' in kw_parameters:
                prop_part.add_dipole(n_verts)
            elif 'Q_in' in kw_parameters:
                prop_part.add_dipole(n_verts, kw_parameters['Q_in'])
        elif prop_part.n_dips > 0 and birth_death > 1-q_death:
            sent_to_death = np.random.random_integers(0, self.n_dips-1)
            prop_part.remove_dipole(sent_to_death)

        # Compute alpha rjmcmc
        if prop_part.n_dips != self.n_dips:
            prop_part.compute_prior(lam)

            if 'sigma_q' in kw_parameters:
                prop_part.compute_loglikelihood_unit(r_data, lead_field,
                                                     s_noise,
                                                     kw_parameters['sigma_q'])
            else:
                prop_part.compute_loglikelihood_unit(r_data, i_data,
                                                     lead_field)

            log_prod_like = prop_part.loglikelihood_unit - \
                self.loglikelihood_unit

            if prop_part.n_dips > self.n_dips:
                if 'sigma_q' in kw_parameters:
                    alpha = np.amin([1, ((q_death * prop_part.prior) /
                                    (q_birth * self.prior)) *
                        np.exp((lklh_exp/(2*s_noise**2)) *
                               log_prod_like)])
                else:
                    alpha = \
                        np.amin([1, (q_death * prop_part.prior *
                                math.fabs(prop_part.dipoles[prop_part.n_dips-1].re_q) *
                                math.fabs(prop_part.dipoles[prop_part.n_dips-1].im_q) /
                                (q_birth * self.prior)) *
                                np.exp((lklh_exp/(2*s_noise**2)) *
                                       log_prod_like)])
            elif prop_part.n_dips < self.n_dips:
                if 'sigma_q' in kw_parameters:
                    alpha = np.amin([1, ((q_birth * prop_part.prior) /
                                    (q_death * self.prior)) *
                        np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)])
                else:
                    alpha = \
                        np.amin([1, ((q_birth * prop_part.prior) /
                                (q_death * self.prior *
                                 math.fabs(self.dipoles[sent_to_death].re_q) *
                                math.fabs(self.dipoles[sent_to_death].im_q))) *
                            np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)])

            if np.random.rand() < alpha:
                self = copy.deepcopy(prop_part)
        return self

    def evol_loc(self, dip, neigh, neigh_p, r_data, i_data, lead_field,
                 lklh_exp, s_noise, lam, **kw_parameters):
        """Perform a Markov Chain Monte Carlo step in order to explore the
           dipole location component of the state space. The dipole is
           allowed to move only to a restricted set of brain points,
           called "neighbours", with a probability that depends on the point.

        Parameters
        ----------
        dip : int
            index of the Particle.dipoles array.
        neigh : array of ints
            The neighbours of each point in the brain discretization.
        neigh_p : array of floats
            The neighbours' probabilities.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SASMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        SASMC additional parameters
        ---------------------------
        s_q : float
            standard deviation of the prior of the dipole moment.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """
        # Step 1: Drawn of the new location.
        prop_part = copy.deepcopy(self)
        p_part = np.cumsum(neigh_p[prop_part.dipoles[dip].loc,
                           np.where(neigh[prop_part.dipoles[dip].loc] != -1)])
        new_pos = False

        while new_pos is False:
            n_rand = np.random.random_sample(1)
            ind_p = np.digitize(n_rand, p_part)[0]
            prop_loc = neigh[prop_part.dipoles[dip].loc, ind_p]
            new_pos = True

            for k in np.delete(range(prop_part.n_dips), dip):
                if prop_loc == prop_part.dipoles[k].loc:
                    new_pos = False

        prob_new_move = neigh_p[prop_part.dipoles[dip].loc, ind_p]

        prob_opp_move = neigh_p[prop_loc,
                                np.argwhere(neigh[prop_loc] ==
                                            prop_part.dipoles[dip].loc)[0][0]]
        prop_part.dipoles[dip].loc = prop_loc
        comp_fact_delta_r = prob_opp_move / prob_new_move

        # Compute alpha mcmc
        prop_part.compute_prior(lam)
        if 'sigma_q' in kw_parameters:
            prop_part.compute_loglikelihood_unit(
                r_data, lead_field, s_noise, kw_parameters['sigma_q'])
        else:
            prop_part.compute_loglikelihood_unit(r_data, i_data, lead_field)

        if not hasattr(self, 'loglikelihood_unit'):
            if 'sigma_q' in kw_parameters:
                self.compute_loglikelihood_unit(
                    r_data, lead_field, s_noise, kw_parameters['sigma_q'])
            else:
                self.compute_loglikelihood_unit(r_data, i_data, lead_field)

        log_prod_like = prop_part.loglikelihood_unit -\
            self.loglikelihood_unit
        alpha = np.amin([1, (comp_fact_delta_r *
                         (prop_part.prior/self.prior) *
                         np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like))])

        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)
        return self


class ParticleSMC(Particle):
    """Particle class for the SMC samplers algorithm.
       Used to store a single particle of an empirical pdf.

    Parameters
    ----------
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    Q_in???????? ___________________________________________________________________


    Attributes
    ----------
    n_dips : int
        The number of dipoles in the particle.
    dipoles : array of instances of Dipole, shape(n_dips,)
        The particle's dipoles.
    loglikelihood_unit : float
        The logarithm of the marginal likelihood, evaluated in the particle.
    prior : float
        The prior pdf, evaluated in the particle.
    """

    def __init__(self, n_verts, lam, Qin):
        """Initialization: initial number of dipoles are Poisson distribuited;
           initial locations are uniformly distribuited within the brain grid
           points, with no dipoles in the same position; initial zetas are
           uniform within [0,1); initial phis are uniform within [0,2\pi);
           initial dipoles strenght are log-uniform with uniform random sign.
        """
        self.n_dips = 0
        self.dipoles = np.array([])
        self.add_dipole(n_verts, Qin, np.random.poisson(lam))
        self.compute_prior(lam)

    def __repr__(self):
        s = "n_dips : %s" % self.n_dips
        for i in range(self.n_dips):
            s += ", dipole " + str(i+1) + " : " + str(self.dipoles[i])
        s += ", prior : %s" % str(self.prior)
        return "<Particle  |  %s>" % s

    def add_dipole(self, n_verts, Qin, num_dip=1):
        """ Add new dipoles to a particle

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        num_dip : int
            The number of dipoles to add.
        """

        new_locs = np.random.randint(0, n_verts, num_dip)
        new_zetas = np.random.random_integers(0, 10**8, num_dip)/10**8
        new_phis = np.random.uniform(0, 2*math.pi, num_dip)
        new_re_qs = [np.sign(np.random.randn()) *
                     10**(np.random.rand()*q_in_range) * (Qin / 35)
                     for _ in itertools.repeat(None, num_dip)]
        new_im_qs = [np.sign(np.random.randn()) *
                     10**(np.random.rand()*q_in_range) * (Qin / 35)
                     for _ in itertools.repeat(None, num_dip)]

        for loc, zeta, phi, re_q, im_q in zip(new_locs, new_zetas,
                                              new_phis, new_re_qs,
                                              new_im_qs):
            while loc in [self.dipoles[dip].loc
                          for dip in range(self.n_dips)]:
                loc = np.random.randint(0, n_verts)

            if zeta == 0:
                phi = np.random.uniform(0, math.pi)

            self.dipoles = np.append(self.dipoles, DipoleSMC(loc, zeta, phi,
                                                             re_q, im_q))
            self.n_dips += 1

    def compute_loglikelihood_unit(self, r_data, i_data, lead_field):
        """ Compute the logarithm of the likelihood of data
            given number and locations of dipoles.

        Parameters
        ----------
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.

        Returns
        -------
        loglikelihood_unit : float
            Particle class attribute.
        """

        if self.n_dips == 0:
            [r_field, i_field] = [np.zeros(lead_field.shape[0]),
                                  np.zeros(lead_field.shape[0])]
        else:
            idx = np.ravel([[3*self.dipoles[idip].loc,
                             3*self.dipoles[idip].loc+1,
                             3*self.dipoles[idip].loc+2]
                            for idip in range(self.n_dips)])
            Gc = lead_field[:, idx]
            [r_field, i_field] = [np.zeros(lead_field.shape[0]),
                                  np.zeros(lead_field.shape[0])]
            for dip in range(self.n_dips):
                # Step 1: compute the direction of the dipole moment
                q_dir = np.array([math.sin(math.acos(self.dipoles[dip].zeta)) *
                                  math.cos(self.dipoles[dip].phi),
                                  math.sin(math.acos(self.dipoles[dip].zeta)) *
                                  math.sin(self.dipoles[dip].phi),
                                  self.dipoles[dip].zeta])
                r_field += self.dipoles[dip].re_q *\
                    np.dot(Gc[:, 3*dip:3*(dip+1)], q_dir)
                i_field += self.dipoles[dip].im_q *\
                    np.dot(Gc[:, 3*dip:3*(dip+1)], q_dir)

        # Step 2: Compute the likelihood_unit distribution:
        #                e^(-|Rfield - Rdata|^2 - |Ifield - Idata|^2)
        #         where Rfield, Ifield are the magnetic field produced by the
        #         particle and Rdata, Idata are the measured ones.

        self.loglikelihood_unit = -(np.linalg.norm(r_field -
                                    r_data.transpose())**2 +
                                    np.linalg.norm(i_field -
                                                   i_data.transpose())**2)
        return self.loglikelihood_unit

    def compute_prior(self, lam):
        """Evaluate the prior pdf in the present particle.

        Parameters
        ----------
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Returns
        -------
        prior : float
            The prior pdf evaluated in the present particle.
        """
        self.prior = 1/math.factorial(self.n_dips) * np.exp(-lam) * \
                     (lam**self.n_dips) * \
            np.prod(np.array([1/math.fabs(self.dipoles[dip].re_q)
                    for dip in range(self.n_dips)])) * \
            np.prod(np.array([1/math.fabs(self.dipoles[dip].im_q)
                    for dip in range(self.n_dips)]))
        return self.prior

    def evol_zeta_phi(self, dip, r_data, i_data, lead_field, lklh_exp,
                      s_noise, lam):
        """Perform a Markov Chain Monte Carlo step in order to explore the
           dipole orientation component of the state space.

        Parameters
        ----------
        dip : int
            index of the Particle.dipoles array.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """

        prop_part = copy.deepcopy(self)

        u = np.array([math.sin(math.acos(prop_part.dipoles[dip].zeta)) *
                      math.cos(prop_part.dipoles[dip].phi),
                     math.sin(math.acos(prop_part.dipoles[dip].zeta)) *
                      math.sin(prop_part.dipoles[dip].phi),
                     prop_part.dipoles[dip].zeta]) +\
            s_orient * np.random.randn(3)

        u = u / np.linalg.norm(u)
        rho = np.linalg.norm(u[0:2])

        prop_part.dipoles[dip].zeta = u[2]

        if np.fabs(u[0]) < np.spacing(1) and np.fabs(u[1]) < np.spacing(1):
            prop_part.dipoles[dip].zeta = 1
            prop_part.dipoles[dip].phi = 0
        elif u[0] >= 0:
            prop_part.dipoles[dip].phi = math.asin(u[1] / rho)
        else:
            prop_part.dipoles[dip].phi = -math.asin(u[1] / rho) + math.pi

        if prop_part.dipoles[dip].zeta > 1:
            print('Pay attention!!')
            # TODO: Gestire meglio l'eccezione
            prop_part.dipoles[dip].zeta = 1
            prop_part.dipoles[dip].phi += math.pi
        elif prop_part.dipoles[dip].zeta < 0:
            prop_part.dipoles[dip].zeta *= -1
            prop_part.dipoles[dip].phi += math.pi
            prop_part.dipoles[dip].re_q *= -1
            prop_part.dipoles[dip].im_q *= -1
        elif prop_part.dipoles[dip].zeta == 0:
            prop_part.dipoles[dip].zeta = np.spacing(1)

        while prop_part.dipoles[dip].phi < 0:
            prop_part.dipoles[dip].phi += 2*math.pi

        while prop_part.dipoles[dip].phi > 2*math.pi:
            prop_part.dipoles[dip].phi -= 2*math.pi

        # Compute alpha in MCMC
        prop_part.compute_prior(lam)
        prop_part.compute_loglikelihood_unit(r_data, i_data, lead_field)

        if not hasattr(self, 'loglikelihood_unit'):
            self.compute_loglikelihood_unit(r_data, i_data, lead_field)
        # TODO: gestire la likelihood come nel cpp tipo calcolarla quando si inizializza il smc.

        log_prod_like = prop_part.loglikelihood_unit - \
            self.loglikelihood_unit
        alpha = np.amin([1, ((prop_part.prior/self.prior) *
                        np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like))])

        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)
        return self

    def evol_q(self, dip, r_data, i_data, lead_field, lklh_exp, s_noise, lam):
        """Perform a Markov Chain Monte Carlo step in order to explore the
           dipole strength component of the state space.

        Parameters
        ----------
        dip : int
            Index of the Particle.dipoles array.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """
        prop_part = copy.deepcopy(self)

        prop_part.dipoles[dip].re_q *= ((np.random.randn()/6)+1)
        prop_part.dipoles[dip].im_q *= ((np.random.randn()/6)+1)
        comp_fact_delta_q = np.exp(-(np.linalg.norm(
            prop_part.dipoles[dip].re_q - self.dipoles[dip].re_q)**2) / 18 *
            (1/(self.dipoles[dip].re_q**2) -
                1/(prop_part.dipoles[dip].re_q**2))) * \
            np.exp(-(np.linalg.norm(prop_part.dipoles[dip].im_q -
                   self.dipoles[dip].im_q)**2) / 18 *
                   (1/(self.dipoles[dip].im_q**2) -
                   1/(prop_part.dipoles[dip].im_q**2)))

        # Compute alpha in MCMC
        prop_part.compute_prior(lam)
        prop_part.compute_loglikelihood_unit(r_data, i_data, lead_field)

        if not hasattr(self, 'loglikelihood_unit'):
            self.compute_loglikelihood_unit(r_data, i_data, lead_field)
        # TODO: Come prima, gestire la likelihood

        log_prod_like = prop_part.loglikelihood_unit - \
            self.loglikelihood_unit


        alpha = np.amin([1, (comp_fact_delta_q *
                        (prop_part.prior/self.prior) *
                        np.exp((lklh_exp/(2*s_noise**2)) *
                           log_prod_like))])



        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)
        return self


class EmpPdf(object):
    """ Empirical probability density function class for the
        semi-analytic SMC samplers algorithm.

    Parameters
    ----------
    n_parts : int
        The number of particles forming the empirical pdf.
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.

    Attributes
    ----------
    samples : array of instances of Particle, shape(n_parts,)
        The EmpPdf's particles.
    logweights : array of floats, shape(n_parts,)
        The logarithm of the weights of the particles forming the
        Empirical pdf.
    ESS : float
        The Effective Sample Size
    exponents : array of floats
        Array whose entries represent points in the space of artificial
        distributions. It is used to keep track of the path followed
        by the SASMC samplers algorithm.
    model_sel : array of floats
        Marginal posterior probability of the number of sources.
    est_n_dips : float
        Estimated number of sources.
    blob : array of floats, shape(est_n_dips x n_verts)
        Intensity measure of the point process.
    est_locs : array of ints
        Estimated sources locations
    """
    def __init__(self, n_parts, n_verts, lam):
        self.samples = np.array([Particle(n_verts, lam)
                                 for _ in itertools.repeat(None, n_parts)])
        self.logweights = np.array([np.log(1/n_parts)
                                    for _ in itertools.repeat(None, n_parts)])
        self.ESS = np.float32(1. / np.square(np.exp(self.logweights)).sum())
        self.exponents = np.array([0, 0])
        self.model_sel = None
        self.est_n_dips = None
        self.blob = None
        self.est_locs = None
        # TODO: controlla di non aver fatto casino con le dichiarazioni

    # def __repr__(self):
    #     pass
    # TODO: fare il repr

    def __str__(self):
        s = ""
        for part in range(self.samples.shape[0]):
            s = s + '---- Particle ' + str(part+1) + \
                '(W = ' + str(np.exp(self.logweights[part])) + \
                ' number of dipoles = ' + str(self.samples[part].nu) + \
                '): \n' + str(self.samples[part]) + '\n'
        return s

    def sample(self, n_parts, n_verts, r_data, i_data, lead_field, neigh,
               neigh_p, s_noise, lam, N_dip_max, **kw_parameters):
        """Perform a full evolution step of the whole empirical pdf.
        This is done by calling the evol_n_dips method on each particle
        forming the empirical pdf and calling the evol_loc method on each dipole of
        each particle forming the empirical pdf.

        Parameters
        ----------
        n_parts : int
            The number of particles forming the empirical pdf.
        n_verts : int
            The number of the points in the given brain discretization.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        neigh : array of ints
            The neighbours of each point in the brain discretization.
        neigh_p : array of floats
            The neighbours' probabilities.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            The parameter of the prior Poisson pdf of the number of dipoles.
        N_dip_max : int
            The maximum number of dipoles allowed in each particle forming the
            empirical pdf.

        Additional parameters
        ---------------------------
        s_q : float
            The standard deviation of the prior of the dipole moment
        """

        for part in range(n_parts):
            self.samples[part] = \
                self.samples[part].evol_n_dips(n_verts, r_data, i_data,
                                               lead_field, N_dip_max,
                                               self.exponents[-1],
                                               s_noise, lam,
                                               **kw_parameters)
            for dip in reversed(range(self.samples[part].n_dips)):
                self.samples[part] = \
                    self.samples[part].evol_loc(dip, neigh, neigh_p, r_data,
                                                i_data, lead_field,
                                                self.exponents[-1],
                                                s_noise, lam,
                                                **kw_parameters)

    def resample(self):
        """Performs a systematic resampling step of the whole empirical pdf
         in which the particles having small normalized importance weights
         are most likely discarded whereas the best particles are replicated
         in proportion to their importance weights. This is done in order to
         prevent the degeneracy of the sample (namely the circumstance in which
         all but one particle have negligible weights).
        """
        weights = np.exp(self.logweights)
        w_part = np.cumsum(weights)

        # ------------------------------------
        w_part[-1] = 1
        w_part[np.where(w_part > 1)] = 1
        # ------------------------------------

        u_part = (np.arange(weights.shape[0], dtype=float) +
                  np.random.uniform()) / weights.shape[0]

        new_ind = np.digitize(u_part, w_part)
        new_ind_ord = np.array(sorted(list(new_ind),
                               key=list(new_ind).count, reverse=True))
        self.samples = self.samples[new_ind_ord]
        self.logweights[:] = np.log(1. / self.logweights.shape[0])
        self.ESS = self.logweights.shape[0]

    def compute_exponent(self, s_noise):
        """The choice for the sequence of artificial distributions  consists
        in starting from the prior distribution and moving towards the
        posterior by increasing the exponent of the likelihood function with
        the iterations.

        This method computes the exponent to be used in the next iteration in
        an "adaptive" manner in order to optimize the trade-off between the
        computational speed and the quality of the approximation.
        Moreover, the method updates the particle weights.

        Parameters
        ----------
        s_noise : float
            The standard deviation of the noise distribution.
        """
        if self.exponents[-1] == 1:
            print('Last iteration...')
            self.exponents = np.append(self.exponents, 1.01)
        else:
            delta_a = delta_min
            delta_b = delta_max
            delta = delta_max
            ESS_new = 0
            iterations = 1
            last_op_incr = False

            while ESS_new/self.ESS > gamma_high or \
                    ESS_new/self.ESS < gamma_low:

                # log of the unnormalized weights
                log_weights_aux = \
                    np.array([self.logweights[i_part] +
                             (delta/(2*s_noise**2)) *
                             self.samples[i_part].loglikelihood_unit
                             for i_part in range(self.samples.shape[0])])
                # normalization
                w = log_weights_aux.max()
                log_weights_aux = log_weights_aux - w - \
                    math.log(np.exp(log_weights_aux - w).sum())
                # Actual weights:
                weights_aux = np.exp(log_weights_aux)

                ESS_new = np.float32(1. / np.square(weights_aux).sum())

                if ESS_new / self.ESS > gamma_high:

                    delta_a = delta
                    delta = min([(delta_a + delta_b)/2, delta_max])
                    last_op_incr = True

                    if (delta_max - delta) < delta_max/100:

                        # log of the unnormalized weights
                        log_weights_aux = \
                            np.array([self.logweights[i_part] +
                                     (delta/(2*s_noise**2)) *
                                     self.samples[i_part].loglikelihood_unit
                                     for i_part in
                                     range(self.samples.shape[0])])
                        # normalization
                        w = log_weights_aux.max()
                        log_weights_aux = log_weights_aux - w - \
                            math.log(np.exp(log_weights_aux - w).sum())
                        # Actual weights:
                        weights_aux = np.exp(log_weights_aux)

                        break

                elif ESS_new / self.ESS < gamma_low:

                    delta_b = delta
                    delta = max([(delta_a + delta_b)/2, delta_min])

                    if (delta - delta_min) < delta_min/10 or \
                            (iterations > 1 and last_op_incr):

                        # log of the unnormalized weights
                        log_weights_aux = \
                            np.array([self.logweights[i_part] +
                                     (delta/(2*s_noise**2)) *
                                     self.samples[i_part].loglikelihood_unit
                                     for i_part in
                                     range(self.samples.shape[0])])
                        # normalization
                        w = log_weights_aux.max()
                        log_weights_aux = log_weights_aux - w - \
                            np.log(np.exp(log_weights_aux - w).sum())
                        # Actual weights:
                        weights_aux = np.exp(log_weights_aux)

                        break
                    last_op_incr = False

                iterations += 1

            if self.exponents[-1] + delta > 1:

                delta = 1 - self.exponents[-1]

                # log of the unnormalized weights
                log_weights_aux =  \
                    np.array([self.logweights[i_part] +
                             (delta/(2*s_noise**2)) *
                             self.samples[i_part].loglikelihood_unit
                             for i_part in range(self.samples.shape[0])])
                # normalization
                w = log_weights_aux.max()
                log_weights_aux = log_weights_aux - w - \
                    math.log(np.exp(log_weights_aux - w).sum())
                # Actual weights:
                weights_aux = np.exp(log_weights_aux)

            self.exponents = np.append(self.exponents,
                                       self.exponents[-1]+delta)
            self.logweights = log_weights_aux
            self.ESS = np.float32(1. / np.square(weights_aux).sum())

    def point_estimate(self, D, N_dip_max):
        """Computation of a point estimate for the number of dipoles and their
        parameters from the empirical distribution

        Parameters
        ----------
        V : float 2D array of shape n_vertices x 3
            Source space
        N_dip_max : int
            maximum allowed number of dipoles in the Particle instance

        Returns
        -------
        self : EmpPdf istance
            updated EmpPdf
        """

        print('Computing estimates...')

        weights = np.exp(self.logweights)

        # Step1: Number of Dipoles
        #    1a) Compute model_selection
        self.model_sel = np.zeros(N_dip_max+1)

        for par in range(self.samples.shape[0]):

            if self.samples[par].n_dips <= N_dip_max:
                self.model_sel[self.samples[par].n_dips] += weights[par]

        #     1b) Compute point estimation
        self.est_n_dips = np.argmax(self.model_sel)

        # Step2: Positions of the dipoles
        if self.est_n_dips == 0:
            self.est_locs = np.array([])
            self.blob = np.array([])
        else:
            nod = np.array([self.samples[part].n_dips
                           for part in range(self.samples.shape[0])])
            selectedsamples = np.delete(self.samples,
                                        np.where(nod != self.est_n_dips))
            selectedweights = np.delete(weights,
                                        np.where(nod != self.est_n_dips))
            ind_bestpart = np.argmax(selectedweights)
            bestpart_locs = \
                np.array([selectedsamples[ind_bestpart].dipoles[dip].loc
                         for dip in range(self.est_n_dips)])
            order_dip = np.empty([selectedsamples.shape[0], self.est_n_dips],
                                 dtype='int')

            all_perms_index = \
                np.asarray(list(itertools.permutations(
                    range(self.est_n_dips))))

            for part in range(selectedsamples.shape[0]):
                part_locs = np.array([selectedsamples[part].dipoles[dip].loc
                                     for dip in range(self.est_n_dips)])

                OSPA = np.mean(D[part_locs[all_perms_index], bestpart_locs],
                               axis=1)

                bestperm = np.argmin(OSPA)
                order_dip[part] = all_perms_index[bestperm]

            self.blob = np.zeros([self.est_n_dips, D.shape[0]])

            for dip in range(self.est_n_dips):
                for par in range(selectedsamples.shape[0]):
                    loc = selectedsamples[par].dipoles[order_dip[par, dip]].loc
                    self.blob[dip, loc] += selectedweights[par]

            self.est_locs = np.argmax(self.blob, axis=1)


class EmpPdfSMC(EmpPdf):
    """ Empirical probability density function class for the
        SMC samplers algorithm.

    Parameters
    ----------
    n_parts : int
        The number of particles forming the empirical pdf.
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.

    Attributes
    ----------
    samples : array of instances of Particle, shape(n_parts,)
        The EmpPdf's particles.
    logweights : array of floats, shape(n_parts,)
        The logarithm of the weights of the particles forming the
        Empirical pdf.
    ESS : float
        The Effective Sample Size
    exponents : array of floats
        Array whose entries represent points in the space of artificial
        distributions. It is used to keep track of the path followed
        by the SASMC samplers algorithm.
    model_sel : array of floats
        Marginal posterior probability of the number of sources.
    est_n_dips : float
        Estimated number of sources.
    blob : array of floats, shape(est_n_dips x n_verts)
        Intensity measure of the point process.
    est_locs : array of ints
        Estimated sources locations
    blob_q : array of floats, shape (est_n_dips, n_verts, 3)
        Marginal posterior distribution of the dipole moment
    est_q : array of floats, shape (1, 3*est_n_dips)
        Estimated dipole moments
    """
    def __init__(self, n_parts, n_verts, lam, Qin):
        self.samples = np.array([ParticleSMC(n_verts, lam, Qin)
                                 for _ in itertools.repeat(None, n_parts)])
        self.logweights = np.array([np.log(1/n_parts)
                                    for _ in itertools.repeat(None, n_parts)])
        self.ESS = n_parts
        self.exponents = np.array([0, 0])

    def sample(self, n_parts, n_verts, r_data, i_data, lead_field, neigh,
               neigh_p, s_noise, lam, N_dip_max, **kw_parameters):
        """Perform a full evolution step of the whole empirical pdf.
        This is done by calling the evol_n_dips method on each particle
        forming the empirical pdf and calling the evol_loc, the evol_zeta_phi
        and the evol_q methods on each dipole of each particle forming the
        empirical pdf.

        Parameters
        ----------
        n_parts : int
            The number of particles forming the empirical pdf.
        n_verts : int
            The number of the points in the given brain discretization.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        i_data : array of floats, shape (n_sens, n_ist)
            The imaginary part of the data; n_sens is the number of sensors
            and n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        neigh : array of ints
            The neighbours of each point in the brain discretization.
        neigh_p : array of floats
            The neighbours' probabilities.
        s_noise : float
            The standard deviation of the noise distribution.
        lam : float
            The parameter of the prior Poisson pdf of the number of dipoles.
        N_dip_max : int
            The maximum number of dipoles allowed in each particle forming the
            empirical pdf.
        """

        for part in range(n_parts):
            self.samples[part] = \
                self.samples[part].evol_n_dips(n_verts, r_data, i_data,
                                               lead_field,N_dip_max,
                                               self.exponents[-1],
                                               s_noise, lam,
                                               **kw_parameters)
            for dip in reversed(range(self.samples[part].n_dips)):
                self.samples[part] = \
                    self.samples[part].evol_loc(dip, neigh, neigh_p, r_data,
                                                i_data, lead_field,
                                                self.exponents[-1],
                                                s_noise, lam,
                                                **kw_parameters)
                self.samples[part] = \
                    self.samples[part].evol_zeta_phi(dip, r_data, i_data,
                                                     lead_field,
                                                     self.exponents[-1],
                                                     s_noise, lam)
                self.samples[part] = \
                    self.samples[part].evol_q(dip, r_data, i_data, lead_field,
                                              self.exponents[-1], s_noise, lam)

    def point_estimate(self, D, N_dip_max):
        """Computation of a point estimate for the number of dipoles and their
        parameters from the empirical distribution

        Parameters
        ----------
        D : float 2D array of shape n_vertices x n_vertices
            Distance matrix
        N_dip_max : int
            maximum allowed number of dipoles in the Particle istance
        """

        print('Computing estimates...')

        weights = np.exp(self.logweights)

        # Step1: Number of Dipoles
        #    1a) Compute model_selection
        self.model_sel = np.zeros(N_dip_max+1)

        for par in range(self.samples.shape[0]):

            if self.samples[par].n_dips <= N_dip_max:
                self.model_sel[self.samples[par].n_dips] += weights[par]

        #     1b) Compute point estimation
        self.est_n_dips = np.argmax(self.model_sel)

        # Step2: Positions of the dipoles
        if self.est_n_dips == 0:
            self.est_locs = np.array([])
            self.est_re_q = np.array([])
            self.est_im_q = np.array([])
            self.blob = np.array([])
            self.blob_re_q = np.array([])
            self.blob_im_q = np.array([])
        else:
            nod = np.array([self.samples[part].n_dips
                           for part in range(self.samples.shape[0])])
            selectedsamples = np.delete(self.samples,
                                        np.where(nod != self.est_n_dips))
            selectedweights = np.delete(weights,
                                        np.where(nod != self.est_n_dips))

            ind_bestpart = np.argmax(selectedweights)
            bestpart_locs = \
                np.array([selectedsamples[ind_bestpart].dipoles[dip].loc
                         for dip in range(self.est_n_dips)])

            order_dip = np.empty([selectedsamples.shape[0],
                                  self.est_n_dips], dtype='int')
            all_perms_index = np.asarray(list(itertools.permutations(
                range(self.est_n_dips))))
            for part in range(selectedsamples.shape[0]):
                part_locs = np.array([selectedsamples[part].dipoles[dip].loc
                                   for dip in range(self.est_n_dips)])

                ospa = np.mean(D[part_locs[all_perms_index], bestpart_locs],
                               axis=1)

                bestperm = np.argmin(ospa)
                order_dip[part] = all_perms_index[bestperm]

            self.blob = np.zeros([self.est_n_dips, D.shape[0]])
            self.blob_re_q = np.zeros([self.est_n_dips, D.shape[0], 3])
            self.blob_im_q = np.zeros([self.est_n_dips, D.shape[0], 3])

            for dip in range(self.est_n_dips):
                for par in range(selectedsamples.shape[0]):

                    loc = selectedsamples[par].dipoles[order_dip[par, dip]].loc
                    zeta = \
                        selectedsamples[par].dipoles[order_dip[par, dip]].zeta
                    phi = selectedsamples[par].dipoles[order_dip[par, dip]].phi
                    re_q = selectedsamples[par].dipoles[order_dip[par, dip]].re_q
                    im_q = selectedsamples[par].dipoles[order_dip[par, dip]].im_q

                    self.blob[dip, loc] += selectedweights[par]

                    self.blob_re_q[dip, loc, 0:3] += selectedweights[par] * re_q * \
                        np.array([math.sin(math.acos(zeta))*math.cos(phi),
                                 math.sin(math.acos(zeta))*math.sin(phi),
                                 zeta])

                    self.blob_im_q[dip, loc, 0:3] += selectedweights[par] * im_q * \
                        np.array([math.sin(math.acos(zeta))*math.cos(phi),
                                 math.sin(math.acos(zeta))*math.sin(phi),
                                 zeta])

            for dip in range(self.est_n_dips):

                nonvoid_loc = np.where(self.blob[dip, 0:D.shape[0]] > 0)[0]

                for j in range(3):
                    self.blob_re_q[dip, nonvoid_loc, j] = \
                        np.divide(self.blob_re_q[dip, nonvoid_loc, j],
                                  self.blob[dip, nonvoid_loc])
                    self.blob_im_q[dip, nonvoid_loc, j] = \
                        np.divide(self.blob_im_q[dip, nonvoid_loc, j],
                                  self.blob[dip, nonvoid_loc])

            self.est_locs = np.argmax(self.blob, axis=1)

            est_re_q_temp = np.array([self.blob_re_q[dip, self.est_locs[dip], 0:3]
                                      for dip in range(self.est_n_dips)])
            self.est_re_q = np.reshape(est_re_q_temp, [1, 3*self.est_n_dips], order='C')

            est_im_q_temp = np.array([self.blob_im_q[dip, self.est_locs[dip], 0:3]
                                      for dip in range(self.est_n_dips)])
            self.est_im_q = np.reshape(est_im_q_temp, [1, 3*self.est_n_dips], order='C')


class SA_SMC(object):
    """ Class representing the SASMC samplers algorithm

    Parameters
    ----------
    forward : dict
        Forward operator
    evoked : instance of Evoked
        The evoked data
    s_noise : float
        The standard deviation of the noise distribution.
    radius : float | None (default None)
        The maximum distance (in cm) that is allowed between two point of
        the source space to be considered neighbours.
        If None it is set equal to 1 cm.
    sigma_neigh: float | None (default None)
        Standard deviation of the probability distribution of neighbours.
        If None it is set equal to radius/2.
    n_parts : int (default 100)
        The number of particles forming the empirical pdf.
    time_in : float | None (default None)
        First instant (in ms) of the time window in which data are analyzed.
        If None time window starts from the first instant of data.
    time_fin : float | None (default None)
        Last istant (in ms) of the time window in which dara are analyzed.
        If None time window ends with the last instant of data.
    s_q : float | None (default None)
        The standard deviation of the prior of the dipole moment.
        If None its value is automatic estimated.
    lam : float (default 0.25)
        The parameter of the prior Poisson pdf of the number of dipoles.
    N_dip_max : int (default 10)
        The maximum number of dipoles allowed in each particle.

    Attributes
    ----------
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    lead_field : array of floats, shape (n_sens x 3*n_verts)
        The leadfield matrix.
    source_space : array of floats, shape  (n_verts, 3)
        The coordinates of the points in the brain discretization.
    forward : dict
        The forward structure.
    neigh : array of ints
        The neighbours of each point in the brain discretization.
    neigh_p : array of floats
        The neighbours' probabilities.
    ###ist_in : int
        The first time point of the time window in which data are analyzed.
    ####ist_fin : int
        The last time point of the time window in which data are analyzed.
    r_data : array of floats, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    i_data : array of floats, shape (n_sens, n_ist)
        The imaginary part of the data; n_sens is the number of sensors
        and n_ist is the number of time-points or of frequencies.
    emp : instance of EmpPdf
        The empirical pdf approximated by the particles at each iteration.
    _resample_it : list of ints
        The iterations during which a resampling step has been performed
    ESS : list of floats
        The Effective Sample Size over the iterations.
    model_sel : list of arrays of floats
        The model selection (i.e. the posterior distribution of the number
        of dipoles) over the iterations.
    est_n_dips : list of ints
        The estimated number of dipoles over the iterations.
    blob : list of 2D arrays of floats
        The intensity measure of the point process over the iterations.
    est_locs : list of array of ints
        The estimated source locations over the iterations.
    est_q : array of floats, shape (n_ist x (3*est_n_dips[-1]))
        The sources' moments estimated at the last iteration.
    gof : float
        The goodness of fit at a fixed iteration, i.e.
                gof = 1 - ||meas_field - rec_field|| / ||meas_field||
        where:
        meas_field is the recorded data,
        rec_field is the reconstructed data,
        and || || is the Frobenius norm.
    """

    def __init__(self, forward, evoked, s_noise, radius=None, sigma_neigh=None,
                 n_parts=100, time_in=None, time_fin=None, subsample=None, s_q=None, lam=_lam,
                 N_dip_max=_n_dip_max):

        # 1) Choosen by the user
        self.n_parts = n_parts
        self.lam = lam
        self.N_dip_max = N_dip_max

        self.forward = forward
        if isinstance(self.forward, list):
            print('Analyzing MEG and EEG data together....')
            self.source_space = forward[0]['source_rr']
            self.n_verts = self.source_space.shape[0]
            s_noise_ratio = s_noise[0] / s_noise[1]
            self.lead_field = np.vstack((forward[0]['sol']['data'], s_noise_ratio*forward[1]['sol']['data']))
            self.s_noise = s_noise[0]
            print('Leadfield shape: ' + str(self.lead_field.shape))
        else:
            self.source_space = forward['source_rr']
            self.n_verts = self.source_space.shape[0]
            self.lead_field = forward['sol']['data']
            self.s_noise = s_noise

        if radius is None:
            self.radius = self.inizialize_radius()
        else:
            self.radius = radius
        self.neigh = self.create_neigh(self.radius)

        if sigma_neigh is None:
            self.sigma_neigh = self.radius/2
        else:
            self.sigma_neigh = sigma_neigh
        self.neigh_p = self.create_neigh_p(self.sigma_neigh)

        if time_in is None:
            self.ist_in = 0
        else:
            self.ist_in = time_in
            # self.ist_in = np.argmin(np.abs(evoked.times-time_in * 0.001))
            # TODO: pensare meglio alla definizione di distanza (istante piu' vicino? o istante prima/dopo?)
        if time_fin is None:
            self.ist_fin = evoked.data.shape[1]-1
        else:
            # self.ist_fin = np.argmin(np.abs(evoked.times - time_fin * 0.001))
            self.ist_fin = time_fin

        if subsample is not None:
            self.subsample = subsample

        if isinstance(evoked, mne.evoked.Evoked):
            if subsample is not None:
                print('Subsampling data with step {0}'.format(subsample))
                data_ist = evoked.data[:, self.ist_in:self.ist_fin + 1:subsample]
                print('Data shape: {0}'.format(data_ist.shape))
            else:
                data_ist = evoked.data[:, self.ist_in:self.ist_fin+1]
                print('Data shape: {0}'.format(data_ist.shape))
        elif isinstance(evoked, list):
            if subsample is not None:
                print('Subsampling data with step {0}'.format(subsample))
                data_ist_eeg = evoked[0][:, self.ist_in:self.ist_fin+1:subsample]
                data_ist_meg = evoked[1][:, self.ist_in:self.ist_fin+1:subsample]
                data_ist = np.vstack((data_ist_eeg, s_noise_ratio*data_ist_meg))
                print('Data shape: {0}'.format(data_ist.shape))
            else:
                data_ist_eeg = evoked[0][:, self.ist_in:self.ist_fin+1]
                data_ist_meg = evoked[1][:, self.ist_in:self.ist_fin+1]
                data_ist = np.vstack((data_ist_eeg, s_noise_ratio*data_ist_meg))
                print('Data shape: {0}'.format(data_ist.shape))
        else:
            if subsample is not None:
                print('Subsampling data with step {0}'.format(subsample))
                data_ist = evoked[:, self.ist_in:self.ist_fin + 1:subsample]
                print('Data shape: {0}'.format(data_ist.shape))
            else:
                data_ist = evoked[:, self.ist_in:self.ist_fin + 1]
                print('Data shape: {0}'.format(data_ist.shape))
        self.r_data = data_ist.real
        self.i_data = data_ist.imag
        del data_ist

        if s_q is None:
            self.s_q = self.estimate_s_q()
            print('Estimated sigma q: ' + str(self.s_q))
        else:
            self.s_q = s_q

        self._resample_it = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.model_sel = list()
        self.blob = list()
        self.SW_pv = list()

        self.emp = EmpPdf(self.n_parts, self.n_verts, self.lam)

        for part in range(self.n_parts):
            self.emp.samples[part].compute_loglikelihood_unit(self.r_data,
                                                              self.lead_field,
                                                              self.s_noise,
                                                              self.s_q)

    def run_filter(self, estimate_q=True):
        """Run the SASMC sampler algorithm and performs point estimation at
         the end of the main loop.

        Parameters
        ----------
        estimate_q : bool
            If true a point-estimate of the dipole moment is computed at the
            last iteration
        """

        # --------- INIZIALIZATION ------------
        # Samples are drawn from the prior distribution and weigths are set as
        # uniform.
        nd = np.array([self.emp.samples[i].n_dips for i in range(self.n_parts)])

        # Creation of dinstances matrix
        D = scipy.spatial.distance.cdist(self.source_space, self.source_space)

        while not np.all(nd <= self.N_dip_max):
            nd_wrong = np.where(nd > self.N_dip_max)[0]
            self.emp.samples[nd_wrong] =\
                np.array([Particle(self.n_verts, self.lam)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([self.emp.samples[i].n_dips for i in range(self.n_parts)])

        # Point estimation for the first iteraction
        self.emp.point_estimate(D, self.N_dip_max)

        self.est_n_dips.append(self.emp.est_n_dips)
        self.model_sel.append(self.emp.model_sel)
        self.est_locs.append(self.emp.est_locs)
        self.blob.append(self.emp.blob)

        # ----------- MAIN CICLE --------------

        while np.all(self.emp.exponents <= 1):
            time_start = time.time()
            print('iteration = ' + str(self.emp.exponents.shape[0]))
            print('exponent = ' + str(self.emp.exponents[-1]))
            print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 1: (possible) resampling
            if self.emp.ESS < self.n_parts/2:

                self._resample_it.append(int(self.emp.exponents.shape[0]))

                print('----- RESAMPLING -----')
                self.emp.resample()
                print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 2: Sampling.
            self.emp.sample(self.n_parts, self.n_verts, self.r_data,
                            self.i_data, self.lead_field, self.neigh,
                            self.neigh_p, self.s_noise, self.lam,
                            self.N_dip_max, sigma_q=self.s_q)

            # STEP 3: Point Estimation
            # self.emp.point_estimate(D, self.N_dip_max)
            #
            # self.est_n_dips.append(self.emp.est_n_dips)
            # self.model_sel.append(self.emp.model_sel)
            # self.est_locs.append(self.emp.est_locs)
            # self.blob.append(self.emp.blob)

            # STEP 4: compute new exponent e new weights
            self.emp.compute_exponent(self.s_noise)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            print("Computation time: " +
                  "{:.2f}".format(time_elapsed) + " seconds")
            print('-------------------------------')

        # Estimation
        self.emp.point_estimate(D, self.N_dip_max)

        self.est_n_dips.append(self.emp.est_n_dips)
        self.model_sel.append(self.emp.model_sel)
        self.est_locs.append(self.emp.est_locs)
        self.blob.append(self.emp.blob)
        if estimate_q:
            if self.est_n_dips[-1] == 0:
                self.est_q = np.array([])
            else:
                self.compute_q(self.est_locs[-1])

    def inizialize_radius(self):
        x_length = np.amax(self.source_space[:, 0]) - np.amin(self.source_space[:, 0])
        y_length = np.amax(self.source_space[:, 1]) - np.amin(self.source_space[:, 1])
        z_length = np.amax(self.source_space[:, 2]) - np.amin(self.source_space[:, 2])

        max_length = max(x_length, y_length, z_length)

        if max_length > 50:
            radius = 10
        elif max_length > 1:
            radius = 1
        else:
            radius = 0.01

        return radius

    def create_neigh(self, radius):
        n_max = 100
        n_min = 3
        D = scipy.spatial.distance.cdist(self.source_space, self.source_space)
        reached_points = np.array([0])
        counter = 0
        n_neigh = []
        list_neigh = []

        while counter < reached_points.shape[0] and self.source_space.shape[0] > reached_points.shape[0]:
            P = reached_points[counter]
            aux = np.array(sorted(np.where(D[P] <= radius)[0],
                                  key=lambda k: D[P, k]))
            n_neigh.append(aux.shape[0])

            # Check the number of neighbours
            if n_neigh[-1] < n_min:
                raise ValueError('Computation of neighbours aborted since '
                                 'their minimum number is definitely too small.\n'
                                 'Please choose a higher radius.')
            elif n_neigh[-1] > n_max:
                raise ValueError('Computation of neighbours aborted since'
                                 'their maximum number is definitely too big.\n'
                                 'Please choose a lower radius.')
            list_neigh.append(aux)
            reached_points = np.append(reached_points, aux[~np.in1d(aux, reached_points)])
            counter += 1

        if counter >= reached_points.shape[0]:
            raise ValueError('Too small value of the radius: the neighbour-matrix'
                             'is not connected')
        elif self.source_space.shape[0] == reached_points.shape[0]:
            while counter < self.source_space.shape[0]:
                P = reached_points[counter]
                aux = np.array(sorted(np.where(D[P] <= radius)[0],
                                      key=lambda k: D[P, k]))
                n_neigh.append(aux.shape[0])

                if n_neigh[-1] < n_min:
                    raise ValueError('Computation of neighbours aborted since '
                                     'their minimum number is definitely too small.\n'
                                     'Please choose a higher radius.')
                elif n_neigh[-1] > n_max:
                    raise ValueError('Computation of neighbours aborted since'
                                     'their maximum number is definitely too big.\n'
                                     'Please choose a lower radius.')

                list_neigh.append(aux)
                counter += 1

            n_neigh_max = max(n_neigh)

            #n_neigh_min = min(n_neigh)
            #n_neigh_mean = sum(n_neigh) / len(n_neigh)
            #print('***** Tested radius = ' + str(radius) + ' *****')
            #print('Maximum number of neighbours: ' + str(n_neigh_max))
            #print('Minimum number of neighbours: ' + str(n_neigh_min))
            #print('Average number of neighbours: ' + str(n_neigh_mean))

            neigh = np.zeros([self.source_space.shape[0], n_neigh_max], dtype=int) - 1
            for i in range(self.source_space.shape[0]):
                neigh[i, 0:list_neigh[i].shape[0]] = list_neigh[i]
            index_ord = np.argsort(neigh[:, 0])
            neigh = neigh[index_ord]
            return neigh

        else:
            raise RuntimeError('Some problems during computation of neighbours.')

    def create_neigh_p(self, sigma_neigh):
        D = scipy.spatial.distance.cdist(self.source_space, self.source_space)
        neigh_p = np.zeros(self.neigh.shape, dtype=float)
        for i in range(self.source_space.shape[0]):
            n_neig = len(np.where(self.neigh[i] > -1)[0])
            neigh_p[i, 0:n_neig] = \
                np.exp(-D[i, self.neigh[i, 0:n_neig]] ** 2 / (2 * sigma_neigh ** 2))
            neigh_p[i] = neigh_p[i] / np.sum(neigh_p[i])
        return neigh_p

    def estimate_s_q(self):
        dipoles_single_max = \
            np.array([np.max(np.absolute(self.lead_field[:, 3 * c:3 * c + 3]))
                      for c in range(self.source_space.shape[0])])
        s_q = np.amax(np.absolute(self.r_data)) / np.mean(dipoles_single_max)
        return s_q

    def compute_q(self, est_locs):
        """Point-estimation of the dipole moment

        Parameters
        ----------
        est_c : int array
            Estimated dipole location (index of the brain grid points)
        """
        est_num = est_locs.shape[0]
        [n_sens, n_time] = np.shape(self.r_data)

        ind = np.ravel([[3*est_locs[idip], 3*est_locs[idip]+1,
                       3*est_locs[idip]+2] for idip in range(est_num)])
        Gc = self.lead_field[:, ind]
        sigma = (self.s_q / self.s_noise)**2 * np.dot(Gc, np.transpose(Gc)) +\
            np.eye(n_sens)
        kal_mat = (self.s_q / self.s_noise)**2 * np.dot(np.transpose(Gc),
                                                       np.linalg.inv(sigma))
        self.est_q = np.array([np.dot(kal_mat, self.r_data[:, t])
                              for t in range(n_time)])

    def goodness_of_fit(self, iteration):
        """Evaluation of the perfomance

        Parameters
        ----------
        iteration : int
            Iteration of interest
        """

        if not hasattr(self, 'est_n_dips'):
            raise AttributeError('None estimation found!!!')

        if type(self).__name__ == 'SA_SMC':
            est_n_dips = self.est_n_dips[-1]
            est_locs = self.est_locs[-1]
            est_q = self.est_q
        elif type(self).__name__ == 'SMC':
            est_n_dips = self.est_n_dips[iteration - 1]
            est_locs = self.est_locs[iteration - 1]
            est_q = self.est_q[iteration - 1]

        meas_field = self.r_data

        # Step 1: error on the estimated number of dipoles
        if hasattr(self, 'true_num_dip'):

            self.est_error = est_n_dips - self.true_num_dip

            if self.est_error > 0:
                print('Number of dipoles overestimated by  ' +
                      str(self.est_error))
            elif self.est_error < 0:
                print('Number of dipoles underestimated by  ' +
                      str(-1 * self.est_error))
            else:
                print('Number of dipoles correctly estimated')

            if hasattr(self, 'true_cs'):
                # Step 2: error on the estimated dipole locations
                if est_n_dips <= self.true_num_dip:
                    all_perms =\
                        np.asarray(list(itertools.permutations(self.true_cs,
                                                               est_n_dips)))
                    ospa = np.array([])

                    for perm in range(all_perms.shape[0]):
                        diff_vctrs = self.source_space[est_locs] - self.source_space[all_perms[perm]]
                        norm_diff = np.linalg.norm(diff_vctrs, axis=1)
                        ospa = np.append(ospa, np.mean(norm_diff))
                    self.OSPA = np.amin(ospa)
                else:
                    all_perms =\
                        np.asarray(
                            list(itertools.permutations(
                                est_locs, self.true_num_dip)
                            ))
                    ospa = np.array([])

                    for perm in range(all_perms.shape[0]):
                        diff_vctrs = self.source_space[all_perms[perm]] - \
                            self.source_space[self.true_cs]
                        norm_diff = np.linalg.norm(diff_vctrs, axis=1)
                        ospa = np.append(ospa, np.mean(norm_diff))
                    self.OSPA = np.amin(ospa)

                print('OSPA metric:  ' + str(self.OSPA))

        # Step 3: GOODNESS OF FIT (aka chi-squared)
        rec_field = np.zeros(meas_field.shape)
        for i_d in range(est_n_dips):

            rec_field += np.dot(self.lead_field[:, 3*est_locs[i_d]:3*(est_locs[i_d]+1)],
                                est_q[:, 3*i_d:3*(i_d+1)].T)

        self.GOF = 1 - np.linalg.norm(meas_field - rec_field) /\
            np.linalg.norm(meas_field)

        print('GOF = ' + str(self.GOF))

    def goodness_of_fit_cart(self, iteration):
        """Evaluation of the perfomance when the true location are available in
           cartesian coordinate

        Parameters
        ----------
        iteration : int
            Iteration of interest
        """
        if not hasattr(self, 'est_num_dip'):
            raise AttributeError('None estimation found!!!')
        est_num_dip = self.est_num_dip[iteration - 1]
        est_cs = self.est_cs[iteration - 1]

        if type(self).__name__ == 'SA_SMC':
            est_q = self.est_q
        elif type(self).__name__ == 'SMC':
            est_q = self.est_q[iteration - 1]

        meas_field = self.Rdata

        # Step 1: error on the estimated number of dipoles
        if hasattr(self, 'true_num_dip'):

            self.est_error = est_num_dip - self.true_num_dip

            if self.est_error > 0:
                print('Number of dipoles overestimated by  ' +
                      str(self.est_error))
            elif self.est_error < 0:
                print('Number of dipoles underestimated by  ' +
                      str(-1 * self.est_error))
            else:
                print('Number of dipoles correctly estimated')

            if hasattr(self, 'true_cs'):
                # Step 2: error on the estimated dipole locations
                if est_num_dip <= self.true_num_dip:
                    all_perms =\
                        np.asarray(
                            list(itertools.permutations(
                                np.arange(self.true_num_dip), est_num_dip)
                                ))
                    ospa = np.array([])

                    for perm in range(all_perms.shape[0]):
                        diff_vctrs = self.source_space[est_cs] - \
                            self.true_cs[all_perms[perm]]
                        norm_diff = np.linalg.norm(diff_vctrs, axis=1)
                        ospa = np.append(ospa, np.mean(norm_diff))
                    self.OSPA = np.amin(ospa)
                else:
                    all_perms =\
                        np.asarray(
                            list(itertools.permutations(
                                est_cs, self.true_num_dip)
                                ))
                    ospa = np.array([])

                    for perm in range(all_perms.shape[0]):
                        diff_vctrs = self.source_space[all_perms[perm]] - self.true_cs
                        norm_diff = np.linalg.norm(diff_vctrs, axis=1)
                        ospa = np.append(ospa, np.mean(norm_diff))
                    self.OSPA = np.amin(ospa)

                print('OSPA metric:  ' + str(self.OSPA))

        # Step 3: GOODNESS OF FIT (aka chi-squared)
        rec_field = np.zeros(meas_field.shape)
        for i_d in range(est_num_dip):

            rec_field += np.dot(self.G[:, 3*est_cs[i_d]:3*(est_cs[i_d]+1)],
                                est_q[:, 3*i_d:3*(i_d+1)].T)

        self.GOF = 1 - np.linalg.norm(meas_field - rec_field) /\
            np.linalg.norm(meas_field)

        print('GOF = ' + str(self.GOF))

    def shapiro_wilk_test(self, iteration):
        """
        Perform Shapiro-Wilk normality test on the difference between the measured and
        the reconstructed field

        :param iteration:
        :return:
        """

        if not hasattr(self, 'est_n_dips'):
            raise AttributeError('None estimation found!!!')

        if type(self).__name__ == 'SA_SMC':
            # est_n_dips = self.est_n_dips[-1]
            # est_locs = self.est_locs[-1]
            # est_q = self.est_q
            raise NotImplementedError
        elif type(self).__name__ == 'SMC':
            est_n_dips = self.est_n_dips[iteration - 1]
            est_locs = self.est_locs[iteration - 1]
            est_q = self.est_q[iteration - 1]

        meas_field = self.r_data
        rec_field = np.zeros(meas_field.shape)
        for i_d in range(est_n_dips):
            rec_field += np.dot(self.lead_field[:, 3*est_locs[i_d]:3*(est_locs[i_d]+1)],
                                est_q[:, 3*i_d:3*(i_d+1)].T)

        SW_w, SW_pv = stats.shapiro(meas_field - rec_field)
        self.SW_pv.append(SW_pv)


    def filter_write_stc(self, file_name, fwd, subject=None):
        """Export results in .stc file

        Parameters
        ----------
        file_name : str
            Path and name of the file to be saved
        fwd : dict
            Forward structure from which the lead-field matrix and the source
            space were been extracted
        subject : str
            Name of the subject
        """
        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'est_cs') or not hasattr(self, 'est_q'):
            raise AttributeError('Do point estimate first!!')

        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        num_iter = self.Rdata.shape[1]

        dip_str = np.zeros([num_iter, nv_tot])
        ndir = 3

        for i_d in range(self.est_cs[-1].shape[0]):
            dip_str[:, self.est_cs[-1][i_d]] = \
                np.linalg.norm(self.est_q[:, ndir*i_d:ndir*(i_d+1)], axis=1)
        stc = SourceEstimate(data=dip_str.T, vertices=vertno, tmin=1,
                             tstep=1, subject=subject)
        # Sara
        stc.save(file_name)
        #return stc

    def to_stc(self, fwd, it_in=1, it_fin=None, subject=None):
        """Export results in .stc file

        Parameters
        ----------
        file_name : str
            Path and name of the file to be saved
        fwd : dict
            Forward structure from which the lead-field matrix and the source
            space were been extracted
        it_in and it_fin : int
            First and last iteration to be saved
        subject : str
            Name of the subject
        """
        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'blob'):
            raise AttributeError('Run filter first!!')

        if it_fin is None:
            it_fin = len(self.blob)

        blobs = self.blob
        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        num_iter = it_fin - it_in + 1
        blob_tot = np.zeros([nv_tot, num_iter])

        for it in np.arange(num_iter) + it_in-1:
            blob_tot[:, it] = np.sum(blobs[it], axis=0)

        stc = SourceEstimate(data=blob_tot, vertices=vertno, tmin=it_in,
                             tstep=1, subject=subject)
        return stc

    def write_stc(self, file_name, fwd, it_in=1, it_fin=None, subject=None):
            """Export results in .stc file

            Parameters
            ----------
            file_name : str
                Path and name of the file to be saved
            fwd : dict
                Forward structure from which the lead-field matrix and the source
                space were been extracted
            it_in and it_fin : int
                First and last iteration to be saved
            subject : str
                Name of the subject
            """

            stc = self.to_stc(fwd, it_in, it_fin, subject)

            stc.save(file_name)


class SMC(SA_SMC):
    """SMC filter

    Parameters
    ----------
    forward : dict
        Forward operator
    evoked : instance of Evoked
        The evoked data
    s_noise : float
        The standard deviation of the noise distribution.
    radius : float | None (default None)
        The maximum distance (in cm) that is allowed between two point of
        the source space to be considered neighbours.
        If None it is set equal to 1 cm.
    sigma_neigh: float | None (default None)
        Standard deviation of the probability distribution of neighbours.
        If None it is set equal to radius/2.
    n_parts : int (default 1000)
        The number of particles forming the empirical pdf.
    top : float | None (default None)
        The instant (in ms) of data to analyze. If None the instant corresponding
        to the peak of the data is analyzed.
    lam : float (default 0.25)
        The parameter of the prior Poisson pdf of the number of dipoles.
    N_dip_max : int (default 10)
        The maximum number of dipoles allowed in each particle.

    Attributes
    ----------
    lead_field : array of floats, shape (n_sens x 3*n_verts)
        The leadfield matrix.
    source_space : array of floats, shape  (n_verts, 3)
        The coordinates of the points in the brain discretization.
    neigh : array of ints
        The neighbours of each point in the brain discretization.
    neigh_p : array of floats
        The neighbours' probabilities.
    top_idx : int
        Index of the data matrix corresponding to the topography to analyze.
    r_data : array of floats, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    i_data : array of floats, shape (n_sens, n_ist)
        The imaginary part of the data; n_sens is the number of sensors
        and n_ist is the number of time-points or of frequencies.
    emp : instance of EmpPdf
        The empirical pdf approximated by the particles at each iteration.
    _resample_it : list of ints
        The iterations during which a resampling step has been performed
    ESS : list of floats
        The Effective Sample Size over the iterations.
    model_sel : list of arrays of floats
        The model selection (i.e. the posterior distribution of the number
        of dipoles) over the iterations.
    est_n_dips : list of ints
        The estimated number of dipoles over the iterations.
    blob : list of 2D arrays of floats
        The intensity measure of the point process over the iterations.
    est_locs : list of array of ints
        The estimated source locations over the iterations.
    blob_q : list of arrays of floats, each of shape (est_n_dips, n_verts, 3)
        Marginal posterior distribution of the dipole moment
        over the iterations.
    est_q : array of floats, shape (n_ist x (3*est_n_dips[-1]))
        The sources' moments estimated at the last iteration.
    gof : float
        The goodness of fit at a fixed iteration, i.e.
                gof = 1 - ||meas_field - rec_field|| / ||meas_field||
        where:
        meas_field is the recorded data,
        rec_field is the reconstructed data,
        and || || is the Frobenius norm.
    """

    def __init__(self, forward, evoked, s_noise, radius=None, sigma_neigh=None,
                 n_parts=1000, top=None, lam=_lam, N_dip_max=_n_dip_max):

        # TODO: dispare a video un messaggio se s_noise non e' dato in input?

        # 1) Choosen by the user
        self.n_parts = n_parts
        self.lam = lam
        self.N_dip_max = N_dip_max

        self.forward = forward
        if isinstance(self.forward, list):
            print('Analyzing MEG and EEG data together....')
            self.source_space = forward[0]['source_rr']
            self.n_verts = self.source_space.shape[0]
            s_noise_ratio = s_noise[0] / s_noise[1]
            self.lead_field = np.vstack((forward[0]['sol']['data'], s_noise_ratio*forward[1]['sol']['data']))
            self.s_noise = s_noise[0]
            print('Leadfield shape: ' + str(self.lead_field.shape))
        else:
            self.source_space = forward['source_rr']
            self.n_verts = self.source_space.shape[0]
            self.lead_field = forward['sol']['data']
            self.s_noise = s_noise

        if radius is None:
            self.radius = self.inizialize_radius()
        else:
            self.radius = radius
        self.neigh = self.create_neigh(self.radius)

        if sigma_neigh is None:
            self.sigma_neigh = self.radius/2
        else:
            self.sigma_neigh = sigma_neigh
        self.neigh_p = self.create_neigh_p(self.sigma_neigh)

        if top is None:
            self.top_idx = np.argmax(np.max(np.abs(evoked.data), axis=0))
        else:
            #self.top_idx = np.argmin(np.abs(evoked.times - top * 0.001))
            self.top_idx = top

        if isinstance(evoked, mne.evoked.Evoked):
            data_ist = evoked.data[:, self.top_idx:self.top_idx+1]
        elif isinstance(evoked, list):
            data_ist_eeg = evoked[0][:, self.top_idx:self.top_idx + 1]
            data_ist_meg = evoked[1][:, self.top_idx:self.top_idx + 1]
            data_ist = np.vstack((data_ist_eeg, s_noise_ratio*data_ist_meg))
            print(data_ist.shape)
        else:
            data_ist = evoked[:, self.top_idx:self.top_idx + 1]
        self.r_data = data_ist.real
        self.i_data = data_ist.imag

        self._resample_it = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.est_re_q = list()
        self.est_im_q = list()
        self.model_sel = list()
        self.blob = list()
        self.blob_re_q = list()
        self.blob_im_q = list()
        self.SW_pv = list()

        self.estimate_Qin()
        del data_ist

        self.emp = EmpPdfSMC(self.n_parts, self.n_verts, self.lam, self.Qin)

        for part in range(self.n_parts):
            self.emp.samples[part].compute_loglikelihood_unit(self.r_data,
                                                              self.i_data,
                                                              self.lead_field)

    def estimate_Qin(self):
        """
        Estimate Qin
        """
        dipoles_single_max = \
            np.array([np.max(np.absolute(
                self.lead_field[:, 3*c:3*c+3]))
                      for c in range(self.source_space.shape[0])])
        r_data_max = np.amax(np.absolute(self.r_data))
        i_data_max = np.amax(np.absolute(self.i_data))
        datamax = max(r_data_max, i_data_max)
        self.Qin = datamax / np.mean(dipoles_single_max)
        #self.Qin = 0.01
        del dipoles_single_max

    def run_filter(self):
        """ Run the SMC samplers algorithm
        """

        # --------- INIZIALIZATION ------------
        # Samples are drawn from the prior distribution and weights are set as
        # uniform.

        nd = np.array([self.emp.samples[i].n_dips for i in range(self.n_parts)])

        D = scipy.spatial.distance.cdist(self.source_space, self.source_space)

        while not np.all(nd <= self.N_dip_max):
            nd_wrong = np.where(nd > self.N_dip_max)[0]
            self.emp.samples[nd_wrong] =\
                np.array([ParticleSMC(self.n_verts, self.lam, self.Qin)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([self.emp.samples[i].n_dips for i in range(self.n_parts)])

        # Point estimation for the first iteraction
        self.emp.point_estimate(self.source_space, self.N_dip_max)
        self.est_n_dips.append(self.emp.est_n_dips)
        self.model_sel.append(self.emp.model_sel)
        self.est_locs.append(self.emp.est_locs)
        self.est_re_q.append(self.emp.est_re_q)
        self.est_im_q.append(self.emp.est_im_q)
        self.blob.append(self.emp.blob)
        self.blob_re_q.append(self.emp.blob_re_q)
        self.blob_im_q.append(self.emp.blob_im_q)

        # ----------- MAIN CICLE --------------

        while np.all(self.emp.exponents <= 1):  # and n <= 2:
            time_start = time.time()
            print('iteration = ' + str(self.emp.exponents.shape[0]))
            print('exponent = ' + str(self.emp.exponents[-1]))
            print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 1: (possible) resampling
            if self.emp.ESS < self.n_parts/2:

                self._resample_it.append(int(self.emp.exponents.shape[0]))
                print('----- RESAMPLING -----')
                self.emp.resample()
                print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 2: Sampling.
            self.emp.sample(self.n_parts, self.n_verts, self.r_data,
                            self.i_data, self.lead_field, self.neigh,
                            self.neigh_p, self.s_noise, self.lam,
                            self.N_dip_max, Q_in=self.Qin)

            # STEP 3: Point Estimation
            self.emp.point_estimate(D, self.N_dip_max)

            self.est_n_dips.append(self.emp.est_n_dips)
            self.model_sel.append(self.emp.model_sel)
            self.est_locs.append(self.emp.est_locs)
            self.est_re_q.append(self.emp.est_re_q)
            self.est_im_q.append(self.emp.est_im_q)
            self.blob.append(self.emp.blob)
            self.blob_re_q.append(self.emp.blob_re_q)
            self.blob_im_q.append(self.emp.blob_im_q)

            # STEP 4: compute new exponent e new weights
            self.emp.compute_exponent(self.s_noise)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            print("Computation time: " +
                  "{:.2f}".format(time_elapsed) + " seconds")
            print('---------------------------')

    def filter_write_stc(self, file_name, fwd, it_in=1, it_fin=None,
                         subject=None):
        """Export results in .stc file

        Parameters
        ----------
        file_name : str
            Path and name of the file to be saved
        fwd : dict
            Forward structure from which the lead-field matrix and the source
            space were been extracted
        it_in and it_fin : int
            First and last iteration to be saved
        subject : str
            Name of the subject
        """
        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'blob'):
            raise AttributeError('Run filter first!!')

        if it_fin is None:
            it_fin = len(self.blob)

        blobs = self.blob
        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        num_iter = it_fin - it_in + 1
        blob_tot = np.zeros([nv_tot, num_iter])

        for it in np.arange(num_iter) + it_in-1:
            blob_tot[:, it] = np.sum(blobs[it], axis=0)

        stc = SourceEstimate(data=blob_tot, vertices=vertno, tmin=it_in,
                             tstep=1, subject=subject)
        stc.save(file_name)

    def to_stc(self, fwd, it_in=1, it_fin=None, subject=None):
        """Export results in .stc file

        Parameters
        ----------
        file_name : str
            Path and name of the file to be saved
        fwd : dict
            Forward structure from which the lead-field matrix and the source
            space were been extracted
        it_in and it_fin : int
            First and last iteration to be saved
        subject : str
            Name of the subject
        """
        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'blob'):
            raise AttributeError('Run filter first!!')

        if it_fin is None:
            it_fin = len(self.blob)

        blobs = self.blob
        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        num_iter = it_fin - it_in + 1
        blob_tot = np.zeros([nv_tot, num_iter])

        for it in np.arange(num_iter) + it_in-1:
            blob_tot[:, it] = np.sum(blobs[it], axis=0)

        stc = SourceEstimate(data=blob_tot, vertices=vertno, tmin=it_in,
                             tstep=1, subject=subject)
        return stc

    def write_stc(self, file_name, fwd, it_in=1, it_fin=None, subject=None):
            """Export results in .stc file

            Parameters
            ----------
            file_name : str
                Path and name of the file to be saved
            fwd : dict
                Forward structure from which the lead-field matrix and the source
                space were been extracted
            it_in and it_fin : int
                First and last iteration to be saved
            subject : str
                Name of the subject
            """

            stc = self.to_stc(fwd, it_in, it_fin, subject)

            stc.save(file_name)


def estimate_noise_std(evoked, time_in=None, time_fin=None, picks=None):
    '''Estimate the standard deviation of the noise distribution from a
    portion of the data which is assumed to be generated from noise only.

    Parameters
    ----------
    evoked: instance of Evoked
        The evoked data
    time_in : float | None (default None)
        First instant (in ms) of the portion of the data used.
        If None it is set equal to the first instant of data.
    time_fin : float | None (default None)
        Last istant (in ms) of the portion of the data used.
        If None it is set equal to the last instant of data.
    picks: array-like of int | None (default None)
        The indices of channels used for the estimation. If None
        all channels are used.

    Returns
    -------
    s_noise : float
        Estimated standard deviation
    '''

    # TODO: gestire meglio i picks (consentire una scrittura tipo picks = 'grad')
    # if time_in is None:
    #     ist_in = 0
    # else:
    #     ist_in = np.argmin(np.abs(evoked.times - time_in * 0.001))
    # if time_fin is None:
    #     ist_fin = evoked.data.shape[1] - 1
    # else:
    #     ist_fin = np.argmin(np.abs(evoked.times - time_fin * 0.001))
    ist_in = time_in
    ist_fin = time_fin

    if isinstance(evoked, mne.evoked.Evoked):
        _data = evoked.data
    else:
        _data = evoked

    if picks is None:
        prestimulus = _data[:, ist_in:ist_fin + 1]
    else:
        prestimulus = _data[picks, ist_in:ist_fin + 1]

    s_noise = np.mean(np.std(prestimulus, axis=1))

    #prova = np.mean(stats.tstd(prestimulus, axis=1))
    #print('ciao ' + str(prova))

    return s_noise
