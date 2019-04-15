# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np


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
        s = 'location : {0}'.format(self.loc)
        return '<Dipole  |  {0}>'.format(s)

    # TODO: decidere se aggiungere momento di dipolo e (forse) coordinate del vertice


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

        # self.add_dipole(n_verts, np.random.poisson(lam))
        # self.compute_prior(lam)

    def __repr__(self):
        s = 'n_dips : {0}'.format(self.n_dips)
        for i_dip, dip in enumerate(self.dipoles):
            s += ', dipole {0} : {1}' .format(i_dip+1, dip)
        s += ', prior : {0}'.format(self.prior)
        return '<Particle  |  {0}>'.format(s)

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
            while loc in [dip.loc for dip in self.dipoles]:
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
            idx = np.ravel([[3*dip.loc, 3*dip.loc+1, 3*dip.loc+2] for dip in self.dipoles])
            Gc = lead_field[:, idx]
            # 1b: compute the variance
            sigma = (s_q / s_noise)**2 * np.dot(Gc, np.transpose(Gc)) + \
                np.eye(n_sens)

        # Step 2: compute inverse and determinant of the variance
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)

        # Step 3: compute the log_likelihood
        self.loglikelihood_unit = - (n_ist * s_noise**2) * np.log(det_sigma)
        for ist in range(n_ist):
            self.loglikelihood_unit -= \
                np.transpose(r_data[:, ist]).dot(inv_sigma).dot(r_data[:, ist])
        return self.loglikelihood_unit
