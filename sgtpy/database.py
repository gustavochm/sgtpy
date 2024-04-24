from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import os
from copy import copy

databasepath = os.path.join(os.path.dirname(__file__), 'database')
databasepath += '/saftgamma_database.xlsx'

file = pd.ExcelFile(databasepath, engine='openpyxl')
df_groups = pd.read_excel(file, 'groups', index_col='groups')
df_mie_kl = pd.read_excel(file, 'unlikemie_kl')
df_asso_kl = pd.read_excel(file, 'unlikeasso_kl')
df_secondorder = pd.read_excel(file, 'secondmie')
df_secondasso = pd.read_excel(file, 'secondasso')


class GCdatabase(object):
    '''
    SAFT-Gamma-Mie group contribution database Object

    This object have implemeted methods to modify the database published
    groups and interactions parameters for SAFT-Gamma-Mie EoS.

    The included parameters are obtained from J. Chem. Eng. Data 2020,
    65, 12, 5862–5890. https://doi.org/10.1021/acs.jced.0c00746

    Parameters
    ----------
    df_groups: DataFrame
        DataFrame that includes group information
    df_mie_kl: DataFrame
        DataFrame that includes unlike Mie parameters
    df_asso_kl: DataFrame
        DataFrame that includes unlike association Parameters
    df_secondorder: DataFrame
        Dataframe that includes second order modification to Mie interactions
    df_scondasso: DataFrame
        Dataframe that includes second order modification to association
        interactions

    Attributes
    ----------
    group_list: List of groups on the database
    df_groups: DataFrame that includes group information
    df_mie_kl: DataFrame that includes unlike Mie parameters
    df_asso_kl: DataFrame that includes unlike association Parameters
    df_secondorder: Dataframe that includes second order modification
    to Mie interactions
    df_scondasso: Dataframe that includes second order modification to
    association interactions

    Methods
    -------
    add_group: method to add a new group
    new_interaction_mie: method to set unlike Mie interactions
    new_interaction_asso: method to add association parameters
    get_interactions: method to get the interactions between two groups
    restore_database: method to restore the database to its initial value
    '''
    def __init__(self, df_groups=df_groups, df_mie_kl=df_mie_kl,
                 df_asso_kl=df_asso_kl, df_secondorder=df_secondorder,
                 df_secondasso=df_secondasso):

        self.df_groups = df_groups
        self.df_mie_kl = df_mie_kl
        self.df_asso_kl = df_asso_kl
        self.df_secondorder = df_secondorder
        self.df_secondasso = df_secondasso

        self.df_groups_backup = copy(df_groups)
        self.df_mie_kl_backup = copy(df_mie_kl)
        self.df_asso_kl_backup = copy(df_asso_kl)
        self.df_secondorder_backup = copy(df_secondorder)
        self.df_secondasso_backup = copy(df_secondasso)

        self.group_list = list(self.df_groups.index)

    def add_group(self, name, vk=1., Sk=1., sigma=0., eps=0., lr=12., la=6.,
                  nH=0, ne1=0, ne2=0, charge=0., sigma_born=0., mw=0.,
                  author_key='author', doi='doi',
                  overwrite=False):
        """
        add_group method

        Method that adds a new group to the database

        Parameters
        ----------
        name: string
            name of the group
        vk: float
            number os sites used by the group
        Sk: float
            shape factor of the group
        sigma: float
            lenght scale of the group, used in Mie potential [Amstrong]
        eps: float
            energy scale of the group, used in Mie potential [K]
        lr: float
            repulsive exponent of the group, used in Mie potential
        la: float
            attractive exponent of the group, used in Mie potential
        nH: int
            number of Hidrogen associative sites
        ne1: int
            number of e1 associative sites
        ne2: int
            number of e2 associative sites
        charge: int
            charge of the group (electron charge) [Adim.]
        sigma_born: float
            diameter used in Born contribution [Amstrong]
        mw: float
            molar weight of the group [g/mol]
        author_key: string
            key of the author that provided the parameters
        doi: string
            doi of the publication that provided the parameters
        overwrite: bool, optional
            whether to overwrite or not current parameters of the database
        """
        Nst = np.count_nonzero([nH, ne1, ne2])
        group_included = name in self.group_list

        if group_included and not overwrite:
            raise Exception('group {} already included in database, set'.format(name) +
                            ' overwrite=True to overwrite the current parameters')
        elif group_included and overwrite:
            new_parameters = [vk, Sk, sigma, eps, lr, la, Nst, nH,
                              ne1, ne2, charge, sigma_born, mw,
                              author_key, doi]

            self.df_groups.loc[name] = new_parameters
        else:
            new_group = {'groups': name, 'vk*': vk, 'Sk': Sk,
                         'sigma_kk': sigma, 'eps_kk': eps,
                         'lr_kk': lr, 'la_kk': la, 'Nst_kk': Nst, 'nH_kk': nH,
                         'ne1_kk': ne1, 'ne2_kk': ne2, 'charge_kk': charge,
                         'sigma_born_kk': sigma_born, 'mw_kk': mw,
                         'author_key': author_key, 'doi': doi}
            groups_aux = self.df_groups.reset_index()
            groups_new = groups_aux.append(new_group, ignore_index=True)
            groups_new.set_index('groups', inplace=True)
            self.df_groups = groups_new
            self.group_list = list(self.df_groups.index)

    def new_interaction_mie(self, group_k, group_l, eps_kl=0., lr_kl='CR',
                            author_key='author', doi='doi',
                            overwrite=False):
        """
        new_interaction_mie method

        Method that adds the unlike mie interactions between group k and
        group l

        Parameters
        ----------
        group_k: string
            name of the group k
        group_l: string
            name of the group l
        eps_kl: float
            unlike energy scale between the groups, used in Mie potential [K]
        lr_kl: float
            unlike repulsive exponent between the groups, used in Mie potential
        author_key: string
            key of the author that provided the parameters
        doi: string
            doi of the publication that provided the parameters
        overwrite: bool, optional
            whether to overwrite or not current parameters of the database
        """
        bool_kk = self.df_mie_kl.group_k == group_k
        bool_ll = self.df_mie_kl.group_l == group_l
        bool_kl = self.df_mie_kl.group_k == group_l
        bool_lk = self.df_mie_kl.group_l == group_k

        index1 = self.df_mie_kl.index[bool_kk & bool_ll]
        len1 = index1.shape[0]

        index2 = self.df_mie_kl.index[bool_kl & bool_lk]
        len2 = index2.shape[0]

        if len1 == 1:
            index = index1
            n = len1
        elif len2 == 1:
            index = index2
            n = len2
        else:
            n = 0
        already_in = n > 0
        if already_in and not overwrite:
            raise Exception('Interaction parameters between group {} '.format(group_k) +
                            ' and group {}'.format(group_l) + ' are already in the database' +
                            ' set overwrite=True to overwrite the current parameters')

        elif already_in and overwrite:
            self.df_mie_kl.iloc[index, [2, 3, 4, 5]] = [eps_kl, lr_kl, author_key, doi]
        else:
            new_interaction = {'group_k': group_k, 'group_l': group_l,
                               'eps_kl': eps_kl, 'lr_kl': lr_kl,
                               'author_key': author_key, 'doi': doi}
            df_mie_new = self.df_mie_kl.append(new_interaction, ignore_index=True)
            self.df_mie_kl = df_mie_new

    def new_interaction_asso(self, group_k, group_l, site_k, site_l,
                             epsAB_kl=0., kAB_kl=0,
                             author_key='author', doi='doi',
                             overwrite=False):
        """
        new_interaction_mie method

        Method that adds the unlike mie interactions between group k and
        group l

        Parameters
        ----------
        group_k: string
            name of the group k
        group_l: string
            name of the group l
        site_k: string
            name of the site k, available optiones are 'H', 'e1' and 'e2'
        site_l: string
            name of the site l, available optiones are 'H', 'e1' and 'e2'
        epsAB_kl: float
            unlike association energy between the the groups [K]
        kAB_kl: float
            unlike association volume between the groups [Amstrong^3]
        author_key: string
            key of the author that provided the parameters
        doi: string
            doi of the publication that provided the parameters
        overwrite: bool, optional
            whether to overwrite or not current parameters of the database
        """
        bool_kk = self.df_asso_kl.group_k == group_k
        bool_ll = self.df_asso_kl.group_l == group_l
        boolsite_kk = self.df_asso_kl.iloc[:, 1] == site_k
        boolsite_ll = self.df_asso_kl.iloc[:, 3] == site_l

        bool_kl = self.df_asso_kl.group_k == group_l
        bool_lk = self.df_asso_kl.group_l == group_k
        boolsite_kl = self.df_asso_kl.iloc[:, 1] == site_l
        boolsite_lk = self.df_asso_kl.iloc[:, 3] == site_k

        index1 = self.df_asso_kl.index[bool_kk & bool_ll & boolsite_kk & boolsite_ll]
        len1 = index1.shape[0]

        index2 = self.df_asso_kl.index[bool_kl & bool_lk & boolsite_kl & boolsite_lk]
        len2 = index2.shape[0]

        if len1 == 1:
            index = index1
            n = len1
        elif len2 == 1:
            index = index2
            n = len2
        else:
            n = 0

        already_in = n > 0
        if already_in and not overwrite:
            raise Exception('Interaction parameters between site{} of'.format(site_k) +
                            ' of group {} '.format(group_k) + 'and site {}'.format(site_l) +
                            ' of group {}'.format(group_l) + 'are already in the database' +
                            ' set overwrite=True to overwrite the current parameters')

        elif already_in and overwrite:
            self.df_asso_kl.iloc[index, [4, 5, 6, 7]] = [epsAB_kl, kAB_kl, author_key, doi]
        else:
            new_interaction = {'group_k': group_k,
                               'site\xa0a\xa0of group\xa0k': site_k,
                               'group_l': group_l,
                               'site\xa0b\xa0of group\xa0l': site_l,
                               'epsAB_kl': epsAB_kl, 'KAB_kl': kAB_kl,
                               'author_key': author_key, 'doi': doi}
            df_asso_new = self.df_asso_kl.append(new_interaction, ignore_index=True)
            self.df_asso_kl = df_asso_new

    def get_interactions(self, group_k, group_l):
        """
        get_interactions method

        Method that outputs the available interactions between group k and
        group l

        Parameters
        ----------
        group_k: string
            name of the group k
        group_l: string
            name of the group l

        Returns
        -------
        df_group : DataFrame
             parameters of each group
        df_mie : DataFrame
             unlike Mie Parameters
        df_asso : DataFrame
             unlike association parameters
        """
        df_group = self.df_groups.loc[[group_l, group_k]]

        bool_kk = self.df_mie_kl.group_k == group_k
        bool_ll = self.df_mie_kl.group_l == group_l
        bool_kl = self.df_mie_kl.group_k == group_l
        bool_lk = self.df_mie_kl.group_l == group_k

        df1 = self.df_mie_kl[bool_kk & bool_ll]
        len1 = df1.shape[0]

        df2 = self.df_mie_kl[bool_kl & bool_lk]
        len2 = df2.shape[0]

        if len1 == 1:
            df_mie = df1
        elif len2 == 1:
            df_mie = df2
        else:
            df_mie = 'There are no custom interaction parameters set for group {}'.format(group_k)
            df_mie += 'and group {}'.format(group_l)

        bool_kk = self.df_asso_kl.group_k == group_k
        bool_ll = self.df_asso_kl.group_l == group_l
        bool_kl = self.df_asso_kl.group_k == group_l
        bool_lk = self.df_asso_kl.group_l == group_k

        df1 = self.df_asso_kl[bool_kk & bool_ll]
        len1 = df1.shape[0]

        df2 = self.df_asso_kl[bool_kl & bool_lk]
        len2 = df2.shape[0]

        if len1 >= 1:
            df_asso = df1
        elif len2 >= 1:
            df_asso = df2
        else:
            df_asso = 'There are no association parameters set for group {}'.format(group_k)
            df_asso += ' and group {}'.format(group_l)

        return df_group, df_mie, df_asso

    def restore_database(self):
        """
        restore_database method

        Method that restores the database of its initial state.
        This method will erase any custom groups or interactions
        set up by the user.
        """
        self.df_groups = copy(self.df_groups_backup)
        self.df_mie_kl = copy(self.df_mie_kl_backup)
        self.df_asso_kl = copy(self.df_asso_kl_backup)
        self.df_secondorder = copy(self.df_secondorder_backup)
        self.df_secondasso = copy(self.df_secondasso_backup)

        self.group_list = list(self.df_groups.index)


database = GCdatabase()
