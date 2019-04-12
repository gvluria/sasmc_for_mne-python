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
