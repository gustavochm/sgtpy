from __future__ import division, print_function, absolute_import
import numpy as np


def asso_aux(Nst_kk, sites_kk, groups_index, subgroups, df_asso_kl):

    index_asso = np.where(Nst_kk > 0)
    molecule_id_index_asso = groups_index[index_asso]
    subgroup_id_asso = subgroups[index_asso]
    sites_asso = sites_kk[index_asso]
    n_sites_molecule = Nst_kk[index_asso]

    molecule_id_index_sites = []
    for i, j in enumerate(sites_asso):
        molecule_id_index_sites += [molecule_id_index_asso[i]] * np.count_nonzero(j)
    molecule_id_index_sites = np.asarray(molecule_id_index_sites)

    group_asso_index = []
    for i, j in enumerate(sites_kk):
        group_asso_index += np.count_nonzero(j) * [i]
    group_asso_index = np.asarray(group_asso_index)

    nsites = len(group_asso_index)
    epsAB_kl = np.zeros([nsites, nsites])
    kAB_kl = np.zeros([nsites, nsites])

    sites_cumsum = np.cumsum(np.hstack([0., n_sites_molecule[:-1]]), dtype=np.int64)

    ngroups = sites_asso.shape[0]
    move_pos_asso = np.array(ngroups*[np.arange(0, 3)])
    where_0e1 = sites_asso[:, 0] == 0
    move_pos_asso[where_0e1] = np.array([0, 0, 1])
    where_0e2 = sites_asso[:, 1] == 0
    move_pos_asso[where_0e2] = np.array([0, 0, 0])

    # index of associating molecule number
    indexABij1 = []
    indexABij2 = []
    # index of interacting site numbers
    indexAB_id1 = []
    indexAB_id2 = []
    for k, groupK in enumerate(subgroup_id_asso):
        for l in range(k, ngroups):
            groupL = subgroup_id_asso[l]

            bool_kk = df_asso_kl.group_k == groupK
            bool_ll = df_asso_kl.group_l == groupL

            bool_kl = df_asso_kl.group_k == groupL

            bool_lk = df_asso_kl.group_l == groupK

            df1 = df_asso_kl[bool_kk & bool_ll]
            len1 = df1.shape[0]

            df2 = df_asso_kl[bool_kl & bool_lk]
            len2 = df2.shape[0]

            if len1 >= len2:
                df = df1
                n2 = len1
            elif len2 > len1:
                df = df2
                n2 = len2

            for j in range(n2):
                values = df.iloc[j].values[0:6]
                groupK2, siteK, groupL2, siteL, epsAB, kAB = values

                if groupK2 != groupK:
                    siteK, siteL = siteL, siteK
                    groupK2, groupL2 = groupL2, groupK2

                if siteK == 'H':
                    moveK = move_pos_asso[k, 0]
                elif siteK == 'e1':
                    moveK = move_pos_asso[k, 1]
                elif siteK == 'e2':
                    moveK = move_pos_asso[k, 2]

                if siteL == 'H':
                    moveL = move_pos_asso[l, 0]
                elif siteL == 'e1':
                    moveL = move_pos_asso[l, 1]
                elif siteL == 'e2':
                    moveL = move_pos_asso[l, 2]

                index0 = sites_cumsum[k] + moveK
                indexf = sites_cumsum[l] + moveL
                epsAB_kl[index0, indexf] = epsAB
                epsAB_kl[indexf, index0] = epsAB
                kAB_kl[index0, indexf] = kAB
                kAB_kl[indexf, index0] = kAB

                # the lenght of the list is the number of interactions
                # they should have the same length as the number of interacting sites
                indexAB_id1.append(molecule_id_index_sites[index0])
                indexAB_id2.append(molecule_id_index_sites[indexf])
                indexABij1.append(index0)
                indexABij2.append(indexf)

                molecule_id0 = molecule_id_index_sites[index0]
                molecule_idf = molecule_id_index_sites[indexf]
                self_site_associating = molecule_id0 == molecule_idf

                if siteK != siteL:
                    indexAB_id1.append(molecule_id_index_sites[indexf])
                    indexAB_id2.append(molecule_id_index_sites[index0])
                    indexABij1.append(indexf)
                    indexABij2.append(index0)
                elif siteK == siteL and not self_site_associating:
                    indexAB_id1.append(molecule_id_index_sites[indexf])
                    indexAB_id2.append(molecule_id_index_sites[index0])
                    indexABij1.append(indexf)
                    indexABij2.append(index0)

            if len1 == 0 and len2 == 0:
                # mixing rule if both groups self-associate
                dfkk = df_asso_kl[bool_kk & bool_lk]
                lenkk = dfkk.shape[0]
                dfll = df_asso_kl[bool_kl & bool_ll]
                lenll = dfll.shape[0]
                epsAB = 0.
                kAB = 0.

                if lenkk == 1 and lenll == 1:
                    values = dfkk.iloc[0].values[0:6]
                    _, siteK_kk, _, siteL_kk, epsAB_kk, kAB_kk = values

                    values = dfll.iloc[0].values[0:6]
                    _, siteK_ll, _, siteL_ll, epsAB_ll, kAB_ll = values

                    epsAB = np.sqrt(epsAB_kk * epsAB_ll)
                    kAB = ((np.cbrt(kAB_kk) + np.cbrt(kAB_ll))/2)**3

                    bool_k1 = [siteK_kk, siteL_kk] == ['H', 'e1']
                    bool_k2 = [siteL_kk, siteK_kk] == ['H', 'e1']
                    bool_k3 = [siteL_kk, siteK_kk] == ['H', 'H']
                    bool_k4 = sites_asso[k, 1] != 0
                    bool_k5 = sites_asso[k, 2] != 0

                    bool_l1 = [siteK_ll, siteL_ll] == ['H', 'e1']
                    bool_l2 = [siteL_ll, siteK_ll] == ['H', 'e1']
                    bool_l3 = [siteL_ll, siteK_ll] == ['H', 'H']
                    bool_l4 = sites_asso[l, 1] != 0
                    bool_l5 = sites_asso[l, 2] != 0

                    bool_k = bool_k1 or bool_k2
                    bool_l = bool_l1 or bool_l2

                    bool_aux1 = bool_l and bool_k3
                    bool_aux2 = bool_k and bool_l3

                    if bool_k and bool_l:
                        # siteK = 'H' and siteL = 'e1' or  siteK = 'e1' and siteL = 'H'
                        index0 = sites_cumsum[k] + move_pos_asso[k, 0]
                        indexf = sites_cumsum[l] + move_pos_asso[l, 1]
                        epsAB_kl[index0, indexf] = epsAB
                        epsAB_kl[indexf, index0] = epsAB
                        kAB_kl[index0, indexf] = kAB
                        kAB_kl[indexf, index0] = kAB

                        indexAB_id1.append(molecule_id_index_sites[index0])
                        indexAB_id2.append(molecule_id_index_sites[indexf])
                        indexABij1.append(index0)
                        indexABij2.append(indexf)
                        indexAB_id1.append(molecule_id_index_sites[indexf])
                        indexAB_id2.append(molecule_id_index_sites[index0])
                        indexABij1.append(indexf)
                        indexABij2.append(index0)
                    if bool_aux1:
                        if bool_l4:
                            # siteK = 'H' and siteL = 'e1'
                            index0 = sites_cumsum[k] + move_pos_asso[k, 0]
                            indexf = sites_cumsum[l] + move_pos_asso[l, 1]
                            epsAB_kl[index0, indexf] = epsAB
                            epsAB_kl[indexf, index0] = epsAB
                            kAB_kl[index0, indexf] = kAB
                            kAB_kl[indexf, index0] = kAB

                            indexAB_id1.append(molecule_id_index_sites[index0])
                            indexAB_id2.append(molecule_id_index_sites[indexf])
                            indexABij1.append(index0)
                            indexABij2.append(indexf)
                            indexAB_id1.append(molecule_id_index_sites[indexf])
                            indexAB_id2.append(molecule_id_index_sites[index0])
                            indexABij1.append(indexf)
                            indexABij2.append(index0)

                        if bool_l5:
                            # siteK = 'H' and siteL = 'e2'
                            index0 = sites_cumsum[k] + move_pos_asso[k, 0]
                            indexf = sites_cumsum[l] + move_pos_asso[l, 2]
                            epsAB_kl[index0, indexf] = epsAB
                            epsAB_kl[indexf, index0] = epsAB
                            kAB_kl[index0, indexf] = kAB
                            kAB_kl[indexf, index0] = kAB

                            indexAB_id1.append(molecule_id_index_sites[index0])
                            indexAB_id2.append(molecule_id_index_sites[indexf])
                            indexABij1.append(index0)
                            indexABij2.append(indexf)
                            indexAB_id1.append(molecule_id_index_sites[indexf])
                            indexAB_id2.append(molecule_id_index_sites[index0])
                            indexABij1.append(indexf)
                            indexABij2.append(index0)

                    if bool_aux2:
                        if bool_k4:
                            # siteK = 'e1' and siteL = 'H'
                            index0 = sites_cumsum[k] + move_pos_asso[k, 1]
                            indexf = sites_cumsum[l] + move_pos_asso[l, 0]
                            epsAB_kl[index0, indexf] = epsAB
                            epsAB_kl[indexf, index0] = epsAB
                            kAB_kl[index0, indexf] = kAB
                            kAB_kl[indexf, index0] = kAB

                            indexAB_id1.append(molecule_id_index_sites[index0])
                            indexAB_id2.append(molecule_id_index_sites[indexf])
                            indexABij1.append(index0)
                            indexABij2.append(indexf)
                            indexAB_id1.append(molecule_id_index_sites[indexf])
                            indexAB_id2.append(molecule_id_index_sites[index0])
                            indexABij1.append(indexf)
                            indexABij2.append(index0)

                        if bool_k5:
                            # siteK = 'e2' and siteL = 'H'
                            index0 = sites_cumsum[k] + move_pos_asso[k, 2]
                            indexf = sites_cumsum[l] + move_pos_asso[l, 0]
                            epsAB_kl[index0, indexf] = epsAB
                            epsAB_kl[indexf, index0] = epsAB
                            kAB_kl[index0, indexf] = kAB
                            kAB_kl[indexf, index0] = kAB

                            indexAB_id1.append(molecule_id_index_sites[index0])
                            indexAB_id2.append(molecule_id_index_sites[indexf])
                            indexABij1.append(index0)
                            indexABij2.append(indexf)
                            indexAB_id1.append(molecule_id_index_sites[indexf])
                            indexAB_id2.append(molecule_id_index_sites[index0])
                            indexABij1.append(indexf)
                            indexABij2.append(index0)

                    if bool_k3 and bool_l3:
                        # siteK = 'H' and siteL = 'H'
                        index0 = sites_cumsum[k] + move_pos_asso[k, 0]
                        indexf = sites_cumsum[l] + move_pos_asso[l, 0]
                        epsAB_kl[index0, indexf] = epsAB
                        epsAB_kl[indexf, index0] = epsAB
                        kAB_kl[index0, indexf] = kAB
                        kAB_kl[indexf, index0] = kAB

                        indexAB_id1.append(molecule_id_index_sites[index0])
                        indexAB_id2.append(molecule_id_index_sites[indexf])
                        indexABij1.append(index0)
                        indexABij2.append(indexf)

    indexABij1 = np.hstack([indexABij1])
    indexABij2 = np.hstack([indexABij2])

    indexAB_id1 = np.hstack([indexAB_id1])
    indexAB_id2 = np.hstack([indexAB_id2])

    # make sure association indeces are integers
    indexABij1 = indexABij1.astype(int)
    indexABij2 = indexABij2.astype(int)
    indexAB_id1 = indexAB_id1.astype(int)
    indexAB_id2 = indexAB_id2.astype(int)

    indexAB_id = (indexAB_id1, indexAB_id2)
    indexABij = (indexABij1, indexABij2)

    out = [kAB_kl, epsAB_kl, sites_asso, group_asso_index, nsites,
           molecule_id_index_sites, indexAB_id, indexABij, subgroup_id_asso,
           molecule_id_index_asso, sites_cumsum]
    return out
