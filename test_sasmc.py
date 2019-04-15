# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#
# License: BSD (3-clause)

from sasmc import Dipole


dip = Dipole(5)

assert(dip.loc == 5)

print(dip)