from __future__ import division, print_function, absolute_import
import numpy as np
from copy import copy
from .saft_forcefield import saft_forcefield
from .constants import kb
# used in saftgammamie
from .config_asso import asso_aux
from .secondorder import secondorder47_1, secondorder47_2, secondorder19_14
from .secondorder import secondorder21, secondorder22, secondorder23
from .database import database


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
        diameter, input in [Ángstrom], stored in [m]
    eps : float
        Mie Potential energy, input in [k], stored in [J]
    lambda_a : float
        Atractive exponent of Mie Potential
    lambda_r : float
        Repulsive exponent of Mie Potential
    eAB : float
        Association Energy, input in [k], stores in [J]
    rcAB : float
        Association rc, input in [sigma], stored in [m]
    rdAB : float
        Association rd, input in [sigma], stored in [m]
    sites : list
        Association sites [Bipolar, Positive, Negative]
    mupol : float
        dipolar moment [Debye]
    npol : float
        numer of polar sites de GV polar term
    ring : float
        geometric factor for ring molecules
        (see Langmuir 2017, 33, 11518-11529, Table I.)
    Tc : float
        Critical temperature [K]
    Pc : float
        Critical Pressure [Pa]
    Zc : float
        critical compresibility factor
    Vc : float
        critical volume [m^3/mol]
    w  : float
        acentric factor
    Mw : float
        molar weight [g/mol]
    cii : list
        polynomial coefficient for influence parameter in SGT [J m^5 / mol^2]

    Methods
    -------
    ci :  evaluates influence parameter polynomial
    saftvrmie_forcefield: use corresponding state principle to correlate
        sigma, eps and lambda_a.
    '''

    def __init__(self, name='None', ms=1., sigma=0., eps=0., lambda_r=12.,
                 lambda_a=6., eAB=0., rcAB=1., rdAB=0.4, sites=[0, 0, 0],
                 mupol=0, npol=0., ring=0., Tc=0., Pc=0., Zc=0., Vc=0.,
                 w=0., Mw=1., cii=0., GC=None):

        self.name = name
        self.Tc = Tc  # Critical Temperature in K
        self.Pc = Pc  # Critical Pressure in Pa
        self.Zc = Zc  # Critical compresibility factor
        self.Vc = Vc  # Critical volume in m3/mol
        self.w = w  # Acentric Factor
        self.cii = cii  # Influence factor SGT, list or array
        self.Mw = Mw  # molar weight in g/mol
        self.nc = 1
        self.GC = GC  # Dict, Group contribution info

        # Saft Parameters
        self.ms = ms
        self.sigma = sigma * 1e-10  # meters
        self.eps = eps * kb  # Joule
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a

        # For ring molecules (see Langmuir 2017, 33, 11518-11529, Table I.)
        self.ring = ring

        # Association Parameters
        self.eAB = eAB * kb  # Joule
        self.rcAB = rcAB * self.sigma  # meters
        self.rdAB = rdAB * self.sigma  # meters
        self.sites = sites

        # Polar parameters
        self.mupol = mupol  # Debye
        self.npol = npol

    def __add__(self, component2):
        '''
        Methods to add two components and create a mixture with them.
        '''
        return mixture(self, component2)

    def ci(self, T):
        """
        Method that evaluates the polynomial for cii coeffient of SGT
        cii must be in J m^5 / mol^2 and T in K.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        ci : float
            influence parameter at given temperature [J m^5 / mol^2]

        """

        return np.polyval(self.cii, T)

    def saftvrmie_forcefield(self, ms, rhol07, ring_type=None):
        """
        saftvrmie_forcefield(ms, rhol07)

        Method that uses corresponding state principle to correlate sigma,
        eps and lambda_r using critical temperature, acentric factor and
        liquid density (Tr=0.7)

        This method requires that critical temperature and acentric factor of
        the fluid has been set.

        This method overwrites the following attributes: ms, sigma, eps,
        lambda_r, lambda_a.

        Parameters
        ----------
        ms : integrer
            number of chain segments
        rhol07: float
            liquid density at reduced temperature = 0.7 [mol/m^3]
        ring_type : None or string
            ring type, 'None' for chain molecules, ring: 'a' for ms=3,
            'b' for ms=4, 'c' or 'd' for ms=5 and 'e' or 'f' for ms=7.
            See Table 1 of Langmuir 2017 33 (42), 11518-11529 or
            Table S2, Supporting information 2 of J. Chem. Inf. Model.
            2021, 61, 3, 1244–1250 for more information about ring geometries.

        Returns
        -------
        sigma : float,
            size parameter for Mie potential [m]
        eps : float
            well depth for Mie potential [K]
        lambda_r : float
            repulsive exponent for Mie potential [Adim]
        ring : float
            geometric factor for ring molecules

        """
        out = saft_forcefield(ms, self.Tc, self.w, rhol07, ring_type)
        lambda_r, lambda_a, ms, eps, sigma, ring = out
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a
        self.ms = ms
        self.sigma = sigma
        self.eps = eps
        self.ring = ring
        return sigma, eps/kb, lambda_r, ring

    def saftgammamie(self, database=database):
        """
        saftgammamie method

        Method that reads the SAFT-Gamma-Mie database and set the necessary
        interactions parameters to use the equation of state.

        Parameters
        ----------
        database: Object
            database object
        """
        df_groups = database.df_groups
        df_mie_kl = database.df_mie_kl
        df_asso_kl = database.df_asso_kl
        df_secondorder = database.df_secondorder

        GC = self.GC
        nc = self.nc

        subgroups = np.asarray(list(GC.keys()))
        ngroups = len(subgroups)
        vki = np.asarray(list(GC.values()))
        ngtotal = np.sum(ngroups)
        groups_index = np.zeros(ngtotal, dtype=np.int64)
        group_indexes = [[0, ngtotal]]

        group_filter = df_groups.loc[subgroups]

        vk = np.asarray(group_filter['vk*'])
        Sk = np.asarray(group_filter['Sk'])

        sigma_kk = np.asarray(group_filter['sigma_kk'])
        eps_kk = np.array(group_filter['eps_kk'])
        lr_kk = np.asarray(group_filter['lr_kk'])
        la_kk = np.asarray(group_filter['la_kk'])

        Nst_kk = np.asarray(group_filter['Nst_kk'])
        sites_kk = np.array(group_filter[['nH_kk', 'ne1_kk', 'ne2_kk']])

        charge_kk = np.array(group_filter['charge_kk'])
        sigma_born_kk = np.array(group_filter['sigma_born_kk'])

        mw_kk = np.array(group_filter['mw_kk'])

        self.vki = vki
        self.subgroups = subgroups
        self.ngroups = ngroups
        self.groups_index = groups_index
        self.groups_indexes = group_indexes

        self.vk = vk
        self.Sk = Sk
        self.sigma_kk = sigma_kk
        self.eps_kk = eps_kk
        self.lr_kk = lr_kk
        self.la_kk = la_kk

        self.Nst_kk = Nst_kk
        self.sites_kk = sites_kk

        self.charge_kk = charge_kk
        self.sigma_born_kk = sigma_born_kk

        self.mw_kk = mw_kk
        self.Mw = np.dot(vki, mw_kk)

        sigma_kk3 = sigma_kk**3
        sigma_kl = np.add.outer(sigma_kk, sigma_kk)/2
        sigma_kl3 = sigma_kl**3
        la_kl = np.sqrt(np.multiply.outer(la_kk-3., la_kk-3.)) + 3
        lr_kl = np.sqrt(np.multiply.outer(lr_kk-3., lr_kk-3.)) + 3

        eps_kl = np.sqrt(np.multiply.outer(eps_kk, eps_kk))
        eps_kl *= np.sqrt(np.multiply.outer(sigma_kk3, sigma_kk3))
        eps_kl /= sigma_kl3

        for k, groupK in enumerate(subgroups):
            for l in range(k, ngtotal):
                groupL = subgroups[l]

                bool_kk = df_mie_kl.group_k == groupK
                bool_ll = df_mie_kl.group_l == groupL
                bool_kl = df_mie_kl.group_k == groupL
                bool_lk = df_mie_kl.group_l == groupK

                df1 = df_mie_kl[bool_kk & bool_ll]
                len1 = df1.shape[0]

                df2 = df_mie_kl[bool_kl & bool_lk]
                len2 = df2.shape[0]

                if len1 == 1:
                    df = df1
                    n = len1
                elif len2 == 1:
                    df = df2
                    n = len2
                else:
                    n = 0

                if n == 1:
                    _, _, eps, lr = df.values[0]

                    if eps != 'CR':
                        eps_kl[k, l] = eps
                        eps_kl[l, k] = eps
                    if lr != 'CR':
                        lr_kl[k, l] = lr
                        lr_kl[l, k] = lr

        # second order modifications
        secondorder47_1(nc, group_indexes, subgroups, vki, eps_kl,
                        df_secondorder)
        secondorder47_2(nc, group_indexes, subgroups, vki, eps_kl,
                        df_secondorder)

        asso_bool = np.any(Nst_kk > 0)
        self.asso_bool = asso_bool
        if asso_bool:
            values = asso_aux(Nst_kk, sites_kk, groups_index, subgroups,
                              df_asso_kl)
            kAB_kl, epsAB_kl, sites_asso, group_asso_index = values[0:4]
            nsites = values[4]
            DIJ = np.zeros([nsites, nsites])
            DIJ[:] = sites_asso[np.nonzero(sites_asso)]
            DIJ[epsAB_kl == 0.] = 0
            self.S = sites_asso[np.nonzero(sites_asso)]
            self.sites_asso = sites_asso
            self.kAB_kl = kAB_kl
            self.epsAB_kl = epsAB_kl
            self.DIJ = DIJ
            self.diagasso = np.diag_indices(nsites)
            self.nsites = len(sites_asso[np.nonzero(sites_asso)])
            self.group_asso_index = group_asso_index

        self.sigma_kl = sigma_kl
        self.eps_kl = eps_kl
        self.lr_kl = lr_kl
        self.la_kl = la_kl


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
        list of Mie diameter [m]
    eps : list
        List of Mie Potential energy [J]
    lambda_a : list
        List of Atractive exponent of Mie Potential
    lambda_r : list
        List Repulsive exponent of Mie Potential
    eAB : list
        List of Association Energy [J]
    rcAB : list
        List of Association rc [m]
    rdAB : float
        List of Association rd [m]
    sites = list
        Association sites [Bipolar, Positive, Negative]
    mupol : list
        List of dipolar moment [Debye]
    npol : list
        List of numer of polar sites de GV polar term
    Tc : list
        Critical temperature [K]
    Pc : list
        Critical Pressure [Pa]
    Zc : list
        critical compresibility factor
    Vc : list
        critical volume [m^3/mol]
    w  : list
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT [J m^5/mol^2]

    Methods
    -------
    add_component : adds a component to the mixture
    copy: returns a copy of the object
    kij_saft : add kij matrix for SAFT-VR-Mie
    lij_saft : add lij matrix for SAFT-VR-Mie
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
        self.Mw = [component1.Mw, component2.Mw]
        self.GC = [component1.GC,  component2.GC]
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
        add_component(component)

        Method that add a component to the mixture

        Parameters
        ----------
        component : object
            pure fluid created with component function
        """
        self.names.append(component.name)
        self.Tc.append(component.Tc)
        self.Pc.append(component.Pc)
        self.Zc.append(component.Zc)
        self.Vc.append(component.Vc)
        self.w.append(component.w)
        self.cii.append(component.cii)
        self.Mw.append(component.Mw)
        self.GC.append(component.GC)

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

    def __add__(self, new_component):
        if isinstance(new_component, component):
            self.add_component(new_component)
        else:
            raise Exception('You can only add components objects to an existing mixture')
        return self

    def kij_saft(self, kij):
        r"""
        kij_saft(kij)

        Method that adds kij correction for Mie potential well depth.

        .. math::
            \epsilon_{ij} = (1-k_{ij}) \frac{\sqrt{\sigma_i^3 \sigma_j^3}}{\sigma_{ij}^3} \sqrt{\epsilon_i \epsilon_j}

        """
        nc = self.nc
        KIJ = np.asarray(kij)
        shape = KIJ.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(KIJ, KIJ.T)

        if isSquare and isSymmetric:
            self.KIJsaft = kij
        else:
            raise Exception('kij matrix is not square or symmetric')

    def lij_saft(self, lij):
        r"""
        lij_saft(lij)

        Method that adds lij correction for association energy.

        .. math::
            \epsilon_{ij}^{AB} = (1 - l_{ij})\sqrt{\epsilon_{ii}^{AB} \epsilon_{jj}^{AB}}

        """

        nc = self.nc
        LIJ = np.asarray(lij)
        shape = LIJ.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(LIJ, LIJ.T)

        if isSquare and isSymmetric:
            self.LIJsaft = lij
        else:
            raise Exception('kij matrix is not square or symmetric')

    def ci(self, T):
        """
        Method that computes the matrix of cij interaction parameter for SGT at
        given temperature.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        ci : array_like
            influence parameter matrix at given temperature [J m^5 / mol^2]

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

    def saftgammamie(self, database=database):
        """
        saftgammamie method

        Method that reads the SAFT-Gamma-Mie database and set the necessary
        interactions parameters to use the equation of state.

        Parameters
        ----------
        database: Object
            database object
        """
        df_groups = database.df_groups
        df_mie_kl = database.df_mie_kl
        df_asso_kl = database.df_asso_kl
        df_secondorder = database.df_secondorder
        df_secondasso = database.df_secondasso

        GC = self.GC
        nc = self.nc

        subgroups = []
        ngroups = []
        vki = []
        for i in range(nc):
            group_names = list(GC[i].keys())
            group_vki = list(GC[i].values())
            subgroups += group_names
            vki += group_vki
            ngroups.append(len(group_names))

        subgroups = np.asarray(subgroups)
        vki = np.asarray(vki)

        ngtotal = np.sum(ngroups)
        groups_index = np.zeros(ngtotal, dtype=np.int64)
        group_indexes = []

        index0 = 0
        for i in range(nc):
            indexf = ngroups[i] + index0
            group_indexes.append([index0, indexf])
            groups_index[index0:indexf] = i
            index0 = indexf

        group_filter = df_groups.loc[subgroups]

        vk = np.asarray(group_filter['vk*'])
        Sk = np.asarray(group_filter['Sk'])

        sigma_kk = np.asarray(group_filter['sigma_kk'])
        eps_kk = np.array(group_filter['eps_kk'])
        lr_kk = np.asarray(group_filter['lr_kk'])
        la_kk = np.asarray(group_filter['la_kk'])

        Nst_kk = np.asarray(group_filter['Nst_kk'])
        sites_kk = np.array(group_filter[['nH_kk', 'ne1_kk', 'ne2_kk']])

        charge_kk = np.array(group_filter['charge_kk'])
        sigma_born_kk = np.array(group_filter['sigma_born_kk'])

        mw_kk = np.array(group_filter['mw_kk'])

        sigma_kk3 = sigma_kk**3
        sigma_kl = np.add.outer(sigma_kk, sigma_kk)/2
        sigma_kl3 = sigma_kl**3
        la_kl = np.sqrt(np.multiply.outer(la_kk-3., la_kk-3.)) + 3
        lr_kl = np.sqrt(np.multiply.outer(lr_kk-3., lr_kk-3.)) + 3

        eps_kl = np.sqrt(np.multiply.outer(eps_kk, eps_kk))
        eps_kl *= np.sqrt(np.multiply.outer(sigma_kk3, sigma_kk3))
        eps_kl /= sigma_kl3

        for k, groupK in enumerate(subgroups):
            for l in range(k, ngtotal):
                groupL = subgroups[l]

                bool_kk = df_mie_kl.group_k == groupK
                bool_ll = df_mie_kl.group_l == groupL
                bool_kl = df_mie_kl.group_k == groupL
                bool_lk = df_mie_kl.group_l == groupK

                df1 = df_mie_kl[bool_kk & bool_ll]
                len1 = df1.shape[0]

                df2 = df_mie_kl[bool_kl & bool_lk]
                len2 = df2.shape[0]

                if len1 == 1:
                    df = df1
                    n = len1
                elif len2 == 1:
                    df = df2
                    n = len2
                else:
                    n = 0

                if n == 1:
                    _, _, eps, lr = df.values[0]

                    if eps != 'CR':
                        eps_kl[k, l] = eps
                        eps_kl[l, k] = eps
                    if lr != 'CR':
                        lr_kl[k, l] = lr
                        lr_kl[l, k] = lr

        self.vki = vki
        self.subgroups = subgroups
        self.ngroups = ngroups
        self.ngtotal = ngtotal
        self.groups_index = groups_index
        self.groups_indexes = group_indexes

        self.vk = vk
        self.Sk = Sk
        self.sigma_kk = sigma_kk
        self.eps_kk = eps_kk
        self.lr_kk = lr_kk
        self.la_kk = la_kk

        self.Nst_kk = Nst_kk
        self.sites_kk = sites_kk

        self.charge_kk = charge_kk
        self.sigma_born_kk = sigma_born_kk

        self.mw_kk = mw_kk

        Mw_i = np.zeros(nc)
        for i in range(nc):
            i0 = group_indexes[i][0]
            i1 = group_indexes[i][1]
            Mw_i[i] = np.dot(vki[i0:i1], mw_kk[i0:i1])
        self.Mw = Mw_i

        # second order modifications
        secondorder47_1(nc, group_indexes, subgroups, vki, eps_kl,
                        df_secondorder)
        secondorder47_2(nc, group_indexes, subgroups, vki, eps_kl,
                        df_secondorder)

        asso_bool = np.any(Nst_kk > 0)
        self.asso_bool = asso_bool
        if asso_bool:
            values = asso_aux(Nst_kk, sites_kk, groups_index, subgroups,
                              df_asso_kl)
            kAB_kl, epsAB_kl, sites_asso, group_asso_index = values[0:4]
            nsites, molecule_id_index_sites = values[4:6]
            indexAB_id, indexABij, subgroup_id_asso = values[6:9]
            molecule_id_index_asso, sites_cumsum = values[9:11]

            secondorder19_14(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                             df_secondorder, subgroup_id_asso,
                             molecule_id_index_asso, sites_cumsum, epsAB_kl,
                             kAB_kl, df_secondasso)

            secondorder21(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                          df_secondorder, subgroup_id_asso,
                          molecule_id_index_asso, sites_cumsum, epsAB_kl,
                          kAB_kl, df_secondasso)

            secondorder22(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                          df_secondorder, subgroup_id_asso,
                          molecule_id_index_asso, sites_cumsum, epsAB_kl,
                          kAB_kl, df_secondasso)

            secondorder23(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                          df_secondorder, subgroup_id_asso,
                          molecule_id_index_asso, sites_cumsum, epsAB_kl,
                          kAB_kl, df_secondasso)

            DIJ = np.zeros([nsites, nsites])
            DIJ[:] = sites_asso[np.nonzero(sites_asso)]
            DIJ[epsAB_kl == 0.] = 0
            self.S = sites_asso[np.nonzero(sites_asso)]
            self.sites_asso = sites_asso
            self.kAB_kl = kAB_kl
            self.epsAB_kl = epsAB_kl
            self.DIJ = DIJ
            self.diagasso = np.diag_indices(nsites)
            self.nsites = nsites
            self.group_asso_index = group_asso_index
            self.molecule_id_index_sites = molecule_id_index_sites
            self.indexAB_id = indexAB_id
            self.indexABij = indexABij
            self.vki_asso = vki[self.group_asso_index]

            dxjdx = np.zeros([nc, nsites])
            for i in range(nc):
                dxjdx[i] = molecule_id_index_sites == i
                dxjdx[i] *= self.vki_asso
            self.dxjasso_dx = dxjdx

        self.sigma_kl = sigma_kl
        self.eps_kl = eps_kl
        self.lr_kl = lr_kl
        self.la_kl = la_kl
