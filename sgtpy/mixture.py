from __future__ import division, print_function, absolute_import
import numpy as np
from copy import copy
from .saft_forcefield import saft_forcefield
from .constants import kb


class component(object):
    '''
    Creates an object with pure component info

    Parameters
    ----------
    name : str
        Name of the component
    ms : float
        Chain lenght
    sigma : float
        diameter
    eps : float
        Mie Potential energy
    lambda_a : float
        Atractive exponent of Mie Potential
    lambda_r : float
        Repulsive exponent of Mie Potential
    eAB : float
        Association Energy
    rcAB : float
        Association rc
    rdAB : float
        Association rd
    sites = list
        Association sites [Bipolar, Positive, Negative]
    mupol : float
        dipolar moment in Debye
    npol : float
        numer of polar sites de GV polar term
    Tc : float
        Critical temperature
    Pc : float
        Critical Pressure
    Zc : float
        critical compresibility factor
    Vc : float
        critical volume
    w  : float
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT

    Methods
    -------
    ci :  evaluates influence parameter polynomial
    '''

    def __init__(self, name='None', ms=1., sigma=0., eps=0., lambda_r=12.,
                 lambda_a=6., eAB=0., rcAB=1., rdAB=0.4, sites=[0, 0, 0],
                 mupol=0, npol=0., ring=0., Tc=0., Pc=0., Zc=0., Vc=0.,
                 w=0., cii=0.):

        self.name = name
        self.Tc = Tc  # Critical Temperature in K
        self.Pc = Pc  # Critical Pressure in bar
        self.Zc = Zc  # Critical compresibility factor
        self.Vc = Vc  # Critical volume in cm3/mol
        self.w = w  # Acentric Factor
        self.cii = cii  # Influence factor SGT, list or array
        self.nc = 1

        # Saft Parameters

        self.ms = ms
        self.sigma = sigma * 1e-10
        self.eps = eps * kb
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a

        # For ring molecules
        self.ring = ring

        # Association Parameters
        self.eAB = eAB * kb
        self.rcAB = rcAB * self.sigma
        self.rdAB = rdAB * self.sigma
        self.sites = sites

        # Polar parameters
        self.mupol = mupol
        self.npol = npol

    def ci(self, T):
        """
        Method that evaluates the polynomial for cii coeffient of SGT
        cii must be in J m^5 / mol and T in K.

        Parameters
        ----------
        T : float
            absolute temperature in K

        Returns
        -------
        ci : float
            influence parameter at given temperature

        """

        return np.polyval(self.cii, T)

    def saftvrmie_forcefield(self, ms, rhol07):
        out = saft_forcefield(ms, self.Tc, self.w, rhol07)
        lambda_r, lambda_a, ms, eps, sigma = out
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a
        self.ms = ms
        self.sigma = sigma
        self.eps = eps


class mixture(object):
    '''
    class mixture
    Creates an object that cointains info about a mixture.

    Parameters
    ----------
    component1 : object
        component created with component class
    component2 : object
        component created with component class

    Attributes
    ----------
    name : list
        Name of the component
    ms : list
        list of chain lenght
    sigma : list
        list of Mie diameter
    eps : list
        List of Mie Potential energy
    lambda_a : list
        List of Atractive exponent of Mie Potential
    lambda_r : list
        List Repulsive exponent of Mie Potential
    eAB : list
        List of Association Energy
    rcAB : list
        List of Association rc
    rdAB : float
        List of Association rd
    sites = list
        Association sites [Bipolar, Positive, Negative]
    mupol : list
        List of dipolar moment in Debye
    npol : list
        List of numer of polar sites de GV polar term
    Tc : list
        Critical temperature
    Pc : list
        Critical Pressure
    Zc : list
        critical compresibility factor
    Vc : list
        critical volume
    w  : list
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT

    Methods
    -------
    add_component : adds a component to the mixture
    copy: returns a copy of the object
    kij_saft : add kij matrix for SAFT-VR-Mie
    ci : computes cij matrix at T for SGT
    '''

    def __init__(self, component1, component2):

        self.names = [component1.name, component2.name]
        self.Tc = [component1.Tc, component2.Tc]
        self.Pc = [component1.Pc, component2.Pc]
        self.Zc = [component1.Zc, component2.Zc]
        self.w = [component1.w, component2.w]
        self.Vc = [component1.Vc, component2.Vc]
        self.cii = [component1.cii, component2.cii]
        self.nc = 2

        self.lr = [component1.lambda_r, component2.lambda_r]
        self.la = [component1.lambda_a, component2.lambda_a]
        self.sigma = [component1.sigma, component2.sigma]
        self.eps = [component1.eps, component2.eps]
        self.ms = [component1.ms, component2.ms]
        self.ring = [component1.ring, component2.ring]
        self.eAB = [component1.eAB, component2.eAB]
        self.rc = [component1.rcAB, component2.rcAB]
        self.rd = [component1.rdAB, component2.rdAB]
        self.sitesmix = [component1.sites, component2.sites]

        self.mupol = [component1.mupol, component2.mupol]
        self.npol = [component1.npol, component2.npol]

    def add_component(self, component):
        """
        Method that add a component to the mixture
        """
        self.names.append(component.name)
        self.Tc.append(component.Tc)
        self.Pc.append(component.Pc)
        self.Zc.append(component.Zc)
        self.Vc.append(component.Vc)
        self.w.append(component.w)
        self.cii.append(component.cii)

        self.lr.append(component.lambda_r)
        self.la.append(component.lambda_a)
        self.sigma.append(component.sigma)
        self.eps.append(component.eps)
        self.ms.append(component.ms)
        self.ring.append(component.ring)
        self.eAB.append(component.eAB)
        self.rc.append(component.rcAB)
        self.rd.append(component.rdAB)
        self.sitesmix.append(component.sites)

        self.mupol.append(component.mupol)
        self.npol.append(component.npol)

        self.nc += 1

    def kij_saft(self, kij):
        self.KIJsaft = kij

    def lij_saft(self, lij):
        self.LIJsaft = lij

    def ci(self, T):
        """
        Method that computes the matrix of cij interaction parameter for SGT at
        T.
        beta is a modification to the interaction parameters and must be added
        as a symmetrical matrix with main diagonal set to zero.

        Parameters
        ----------
        T : float
            absolute temperature in K

        Returns
        -------
        ci : array_like
            influence parameter matrix at given temperature

        """

        n = len(self.cii)
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i], T)
        self.cij = np.sqrt(np.outer(ci, ci))
        return self.cij

    def copy(self):
        """
        Method that return a copy of the mixture

        Returns
        -------
        mix : object
            returns a copy a of the mixture
        """
        return copy(self)
