import numpy as np


def secondorder47_1(nc, group_indexes, subgroups, vki, eps_kl, secondorder):
    # condition between group 'COO-' and 'CH3'
    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond1 = 'CH3' in subgroups_id_molecule and 'COO-' in subgroups_id_molecule
        cond10 = subgroups_id_molecule.shape[0] == 2
        if cond1 and cond10:

            cond1_in1 = np.where(subgroups_id_molecule == 'CH3')
            bool11 = vki_molecule[cond1_in1] == 1

            cond1_in2 = np.where(subgroups_id_molecule == 'COO-')
            bool12 = vki_molecule[cond1_in2] == 1

            if bool11 and bool12:

                index1 = cond1_in1[0] + index0

                index47 = np.where(subgroups == 47)[0]

                boolk = secondorder.group_k == 'CH3'
                booll = secondorder.group_l == 'COO-'
                eps147 = secondorder[boolk & booll]['eps_kl'].values

                eps_kl[index1, index47] = eps147
                eps_kl[index47, index1] = eps147

            del bool11, bool12


def secondorder47_2(nc, group_indexes, subgroups, vki, eps_kl, secondorder):
    # condition between group 'COO-' and 'CH2'
    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond2 = 'CH3' in subgroups_id_molecule and 'COO-' in subgroups_id_molecule and 'CH2' in subgroups_id_molecule
        cond20 = subgroups_id_molecule.shape[0] == 3
        if cond2 and cond20:

            cond2_in1 = np.where(subgroups_id_molecule == 'CH3')
            bool21 = vki_molecule[cond2_in1] == 1

            cond2_in2 = np.where(subgroups_id_molecule == 'CH2')
            bool22 = vki_molecule[cond2_in2] == 1

            cond2_in3 = np.where(subgroups_id_molecule == 'COO-')
            bool23 = vki_molecule[cond2_in3] == 1

            if bool21 and bool22 and bool23:

                index2 = cond2_in2[0] + index0

                index47 = np.where(subgroups == 'COO-')[0]
                boolk = secondorder.group_k == 'CH2'
                booll = secondorder.group_l == 'COO-'
                eps247 = secondorder[boolk & booll]['eps_kl'].values

                eps_kl[index2, index47] = eps247
                eps_kl[index47, index2] = eps247

            del bool21, bool22, bool23


def secondorder19_14(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                     secondorder, subgroup_id_asso, molecule_id_index_asso,
                     sites_cumsum, epsAB_kl, kAB_kl, secondasso):
    # condition between group 'CH2OH' and 'H2O'
    bool19 = []

    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond3 = 'CH2OH' in subgroups_id_molecule

        cond30 = subgroups_id_molecule.shape[0] == 3

        cond31 = subgroups_id_molecule.shape[0] == 2

        cond32 = 'CH2' in subgroups_id_molecule

        if cond3 and cond30 and cond32:

            cond3_in19 = np.where(subgroups_id_molecule == 'CH2OH')
            bool319 = vki_molecule[cond3_in19] == 1

            cond3_in2 = np.where(subgroups_id_molecule == 'CH2')
            bool32 = vki_molecule[cond3_in2] in [1, 2]

            whereR = np.logical_and(subgroups_id_molecule != 'CH2OH', subgroups_id_molecule != 'CH2')
            R_id = subgroups_id_molecule[whereR]

            if R_id in ['CH3', 'NH2', 'NH', 'N']:
                bool33 = vki_molecule[whereR] == 1
            else:
                bool33 = False

            if R_id == 'CH3' and vki_molecule[cond3_in2] == 2:
                # for butanol
                bool34 = False
            else:
                bool34 = True

        elif cond3 and cond31:

            cond3_in19 = np.where(subgroups_id_molecule == 'CH2OH')
            bool319 = vki_molecule[cond3_in19] == 1

            bool32 = True
            whereR = np.where(subgroups_id_molecule != 'CH2OH')
            R_id = subgroups_id_molecule[whereR]
            bool34 = True

            if R_id in ['CH3', 'NH2', 'NH', 'N']:
                bool33 = vki_molecule[whereR] == 1
            else:
                bool33 = False

        else:
            bool319 = False
            bool32 = False
            bool33 = False
            bool34 = False

        bool19.append(bool319 and bool32 and bool33 and bool34)

        if bool319 and bool32 and bool33 and bool34:

            index19 = cond3_in19[0] + index0

            index14 = np.where(subgroups == 'H2O')[0]

            boolk = secondorder.group_k == 'CH2OH'
            booll = secondorder.group_l == 'H2O'
            eps1914 = secondorder[boolk & booll]['eps_kl'].values

            eps_kl[index19, index14] = eps1914
            eps_kl[index14, index19] = eps1914

    bool19 = np.hstack(bool19)
    cond_asso19_0 = np.any(bool19)
    cond_asso19_1 = np.any(subgroup_id_asso == 'CH2OH')

    if cond_asso19_0 and cond_asso19_1:
        where14 = np.where(subgroup_id_asso == 'H2O')

        where19all = np.where(subgroup_id_asso == 'CH2OH')
        where19all_id = molecule_id_index_asso[where19all]
        where19second = np.array(bool19)[where19all_id]
        where19 = where19all[0][where19second]
        ncond19 = len(where19)

        boolk = secondasso.group_k == 'CH2OH'
        booll = secondasso.group_l == 'H2O'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]

        for k in range(ncond19):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values

                if siteK == 'H' and siteL == 'e1':
                    indexH = sites_cumsum[where14] + 1
                    indexOH = sites_cumsum[where19[k]]

                    epsAB_kl[indexH, indexOH] = epsAB
                    epsAB_kl[indexOH, indexH] = epsAB
                    kAB_kl[indexH, indexOH] = kAB
                    kAB_kl[indexOH, indexH] = kAB
                elif siteK == 'e1' and siteL == 'H':

                    indexH = sites_cumsum[where14]
                    indexOH = sites_cumsum[where19[k]] + 1

                    epsAB_kl[indexH, indexOH] = epsAB
                    epsAB_kl[indexOH, indexH] = epsAB
                    kAB_kl[indexH, indexOH] = kAB
                    kAB_kl[indexOH, indexH] = kAB


def secondorder21(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                  secondorder, subgroup_id_asso, molecule_id_index_asso,
                  sites_cumsum, epsAB_kl, kAB_kl, secondasso):
    # condition between groups 'NH2'-'H2O' and 'NH2'-'CO2'

    bool21 = []
    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond4 = 'NH2' in subgroups_id_molecule and 'CH2' in subgroups_id_molecule and 'CH2OH' in subgroups_id_molecule
        cond40 = subgroups_id_molecule.shape[0] == 3

        bool419, bool421, bool42 = False, False, False

        if cond4 and cond40:

            cond4_in19 = np.where(subgroups_id_molecule == 'CH2OH')
            bool419 = vki_molecule[cond4_in19] == 1

            cond4_in21 = np.where(subgroups_id_molecule == 'NH2')
            bool421 = vki_molecule[cond4_in21] == 1

            cond4_in2 = np.where(subgroups_id_molecule == 'CH2')
            bool42 = vki_molecule[cond4_in2] in [1, 2]

        bool21.append(bool419 and bool421 and bool42)
        if bool419 and bool421 and bool42:

            index21 = cond4_in21[0] + index0

            index14 = np.where(subgroups == 'H2O')[0]

            index17 = np.where(subgroups == 'CO2')[0]

            boolNH2 = secondorder.group_k == 'NH2'
            boolH2O = secondorder.group_l == 'H2O'
            boolCO2 = secondorder.group_l == 'CO2'
            eps2114 = secondorder[boolNH2 & boolH2O]['eps_kl'].values
            eps2117 = secondorder[boolNH2 & boolCO2]['eps_kl'].values
            lr2117 = secondorder[boolNH2 & boolCO2]['lr_kl'].values

            eps_kl[index21, index14] = eps2114
            eps_kl[index14, index21] = eps2114

            eps_kl[index17, index21] = eps2117
            eps_kl[index21, index17] = eps2117

            lr_kl[index17, index21] = lr2117
            lr_kl[index21, index17] = lr2117

        del bool419, bool421, bool42

    bool21 = np.hstack(bool21)
    cond_asso21_0 = np.any(bool21)
    cond_asso21_14 = np.any(subgroup_id_asso == 'H2O')
    cond_asso21_17 = np.any(subgroup_id_asso == 'CO2')
    cond_asso21_19 = np.any(subgroup_id_asso == 'CH2OH')

    where21all = np.where(subgroup_id_asso == 'NH2')
    where21all_id = molecule_id_index_asso[where21all]
    where21second = bool21[where21all_id]
    where21 = where21all[0][where21second]
    ncond21 = len(where21)

    if cond_asso21_0 and cond_asso21_14:

        where14 = np.where(subgroup_id_asso == 'H2O')

        boolk = secondasso.group_k == 'NH2'
        booll = secondasso.group_l == 'H2O'
        df = secondasso[boolk & booll]

        len1 = df.shape[0]

        for k in range(ncond21):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values
                if siteK == 'H' and siteL == 'e1':
                    indexH = sites_cumsum[where14] + 1
                    indexNH2 = sites_cumsum[where21[k]]

                    epsAB_kl[indexH, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexH] = epsAB
                    kAB_kl[indexH, indexNH2] = kAB
                    kAB_kl[indexNH2, indexH] = kAB

                elif siteK == 'e1' and siteL == 'H':

                    indexH = sites_cumsum[where14]
                    indexNH2 = sites_cumsum[where21[k]] + 1

                    epsAB_kl[indexH, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexH] = epsAB
                    kAB_kl[indexH, indexNH2] = kAB
                    kAB_kl[indexNH2, indexH] = kAB


    if cond_asso21_0 and cond_asso21_17:

        where17 = np.where(subgroup_id_asso == 'CO2')

        boolk = secondasso.group_k == 'NH2'
        booll = secondasso.group_l == 'CO2'
        df = secondasso[boolk & booll]

        len1 = df.shape[0]

        for k in range(ncond21):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values
                if siteK == 'e1' and siteL == 'e1':
                    indexCO2 = sites_cumsum[where17]
                    indexNH2 = sites_cumsum[where21[k]] + 1

                    epsAB_kl[indexCO2, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexNH2] = kAB
                    kAB_kl[indexNH2, indexCO2] = kAB
                elif siteK == 'e1' and siteL == 'e2':
                    indexCO2 = sites_cumsum[where17] + 1
                    indexNH2 = sites_cumsum[where21[k]] + 1

                    epsAB_kl[indexCO2, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexNH2] = kAB
                    kAB_kl[indexNH2, indexCO2] = kAB

    if cond_asso21_0 and cond_asso21_19:

        where19 = np.where(subgroup_id_asso == 'CH2OH')

        boolk = secondasso.group_k == 'NH2'
        booll = secondasso.group_l == 'CH2OH'
        df = secondasso[boolk & booll]

        len1 = df.shape[0]

        for k in range(ncond21):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values

                if siteK == 'H' and siteL == 'e1':
                    indexOH = sites_cumsum[where19] + 1
                    indexNH2 = sites_cumsum[where21[k]]

                    epsAB_kl[indexOH, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexOH] = epsAB
                    kAB_kl[indexOH, indexNH2] = kAB
                    kAB_kl[indexNH2, indexOH] = kAB
                elif siteK == 'e1' and siteL == 'H':

                    indexOH = sites_cumsum[where19]
                    indexNH2 = sites_cumsum[where21[k]] + 1

                    epsAB_kl[indexOH, indexNH2] = epsAB
                    epsAB_kl[indexNH2, indexOH] = epsAB
                    kAB_kl[indexOH, indexNH2] = kAB
                    kAB_kl[indexNH2, indexOH] = kAB


def secondorder22(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                  secondorder, subgroup_id_asso, molecule_id_index_asso,
                  sites_cumsum, epsAB_kl, kAB_kl, secondasso):
    # condition between groups 'NH'-'H2O' and 'NH'-'CO2'
    bool22 = []
    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond6 = 'NH' in subgroups_id_molecule and 'CH2' in subgroups_id_molecule and 'CH2OH' in subgroups_id_molecule
        cond60 = subgroups_id_molecule.shape[0] == 3

        bool622, bool62, bool619 = False, False, False

        if cond6 and cond60:

            cond6_in22 = np.where(subgroups_id_molecule == 'NH')
            bool622 = vki_molecule[cond6_in22] == 1

            cond6_in2 = np.where(subgroups_id_molecule == 'CH2')
            bool62 = vki_molecule[cond6_in2] == 2

            cond6_in19 = np.where(subgroups_id_molecule == 'CH2OH')
            bool619 = vki_molecule[cond6_in19] == 2

        bool22.append(bool622 and bool62 and bool619)
        if bool622 and bool62 and bool619:
            # second order mie between NH and CO2
            index22 = cond6_in22[0] + index0

            index17 = np.where(subgroups == 'CO2')[0]

            boolNH = secondorder.group_k == 'NH'
            boolCO2 = secondorder.group_l == 'CO2'
            eps2217 = secondorder[boolNH & boolCO2]['eps_kl'].values
            lr2217 = secondorder[boolNH & boolCO2]['lr_kl'].values

            eps_kl[index17, index22] = eps2217
            eps_kl[index22, index17] = eps2217

            lr_kl[index17, index22] = lr2217
            lr_kl[index22, index17] = lr2217

            # second order mie between NH and H2O
            index14 = np.where(subgroups == 'H2O')[0]
            boolH2O = secondorder.group_l == 'H2O'
            eps2214 = secondorder[boolNH & boolH2O]['eps_kl'].values
            lr2214 = secondorder[boolNH & boolH2O]['lr_kl'].values

            eps_kl[index14, index22] = eps2214
            eps_kl[index22, index14] = eps2214
            lr_kl[index14, index22] = lr2214
            lr_kl[index22, index14] = lr2214


        del bool622, bool62, bool619

    bool22 = np.hstack(bool22)
    cond_asso22_0 = np.any(bool22)
    cond_asso22_14 = np.any(subgroup_id_asso == 'H2O')
    cond_asso22_17 = np.any(subgroup_id_asso == 'CO2')
    cond_asso22_19 = np.any(subgroup_id_asso == 'CH2OH')

    where22all = np.where(subgroup_id_asso == 'NH')
    where22all_id = molecule_id_index_asso[where22all]
    where22second = bool22[where22all_id]
    where22 = where22all[0][where22second]
    ncond22 = len(where22)

    if cond_asso22_0 and cond_asso22_14:
        where14 = np.where(subgroup_id_asso == 'H2O')
        boolk = secondasso.group_k == 'NH'
        booll = secondasso.group_l == 'H2O'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]
        for k in range(ncond22):
            for j in range(len1):
                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values
                if siteK == 'e1' and siteL == 'H':
                    indexH2O = sites_cumsum[where14]
                    indexNH = sites_cumsum[where22[k]] + 1
                    epsAB_kl[indexH2O, indexNH] = epsAB
                    epsAB_kl[indexNH, indexH2O] = epsAB
                    kAB_kl[indexH2O, indexNH] = kAB
                    kAB_kl[indexNH, indexH2O] = kAB
                elif siteK == 'H' and siteL == 'e1':
                    indexH2O = sites_cumsum[where14] + 1
                    indexNH = sites_cumsum[where22[k]]
                    epsAB_kl[indexH2O, indexNH] = epsAB
                    epsAB_kl[indexNH, indexH2O] = epsAB
                    kAB_kl[indexH2O, indexNH] = kAB
                    kAB_kl[indexNH, indexH2O] = kAB

    if cond_asso22_0 and cond_asso22_17:

        where17 = np.where(subgroup_id_asso == 'CO2')

        boolk = secondasso.group_k == 'NH'
        booll = secondasso.group_l == 'CO2'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]

        for k in range(ncond22):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values
                if siteK == 'e1' and siteL == 'e1':
                    indexCO2 = sites_cumsum[where17]
                    indexNH = sites_cumsum[where22[k]] + 1

                    epsAB_kl[indexCO2, indexNH] = epsAB
                    epsAB_kl[indexNH, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexNH] = kAB
                    kAB_kl[indexNH, indexCO2] = kAB

                elif siteK == 'e1' and siteL == 'e2':

                    indexCO2 = sites_cumsum[where17] + 1
                    indexNH = sites_cumsum[where22[k]] + 1

                    epsAB_kl[indexCO2, indexNH] = epsAB
                    epsAB_kl[indexNH, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexNH] = kAB
                    kAB_kl[indexNH, indexCO2] = kAB

    if cond_asso22_0 and cond_asso22_19:

        where19 = np.where(subgroup_id_asso == 'CH2OH')

        boolk = secondasso.group_k == 'NH'
        booll = secondasso.group_l == 'CH2OH'
        df = secondasso[boolk & booll]

        len1 = df.shape[0]

        for k in range(ncond22):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values
                if siteK == 'H' and siteL == 'e1':
                    indexOH = sites_cumsum[where19] + 1
                    indexNH = sites_cumsum[where22[k]]

                    epsAB_kl[indexOH, indexNH] = epsAB
                    epsAB_kl[indexNH, indexOH] = epsAB
                    kAB_kl[indexOH, indexNH] = kAB
                    kAB_kl[indexNH, indexOH] = kAB
                elif siteK == 'e1' and siteL == 'H':

                    indexOH = sites_cumsum[where19]
                    indexNH = sites_cumsum[where22[k]] + 1

                    epsAB_kl[indexOH, indexNH] = epsAB
                    epsAB_kl[indexNH, indexOH] = epsAB
                    kAB_kl[indexOH, indexNH] = kAB
                    kAB_kl[indexNH, indexOH] = kAB


def secondorder23(nc, group_indexes, subgroups, vki, eps_kl, lr_kl,
                  secondorder, subgroup_id_asso, molecule_id_index_asso,
                  sites_cumsum, epsAB_kl, kAB_kl, secondasso):
    # condition between groups 'N'-'H2O' and 'N'-'CO2'
    bool23 = []
    for i in range(nc):
        index0 = group_indexes[i][0]
        indexf = group_indexes[i][1]
        subgroups_id_molecule = subgroups[index0:indexf]
        vki_molecule = vki[index0:indexf]

        cond7 = 'N' in subgroups_id_molecule and 'CH2' in subgroups_id_molecule and 'CH2OH' in subgroups_id_molecule and 'CH3' in subgroups_id_molecule
        cond70 = subgroups_id_molecule.shape[0] == 4
        cond71 = 4 <= np.sum(vki_molecule) <= 6

        bool723, bool719, bool72, bool73, bool71 = False, False, False, False, False

        if cond7 and cond70 and cond71:

            cond7_in23 = np.where(subgroups_id_molecule == 'N')
            bool723 = vki_molecule[cond7_in23] == 1

            cond7_in19 = np.where(subgroups_id_molecule == 'CH2OH')
            bool719 = vki_molecule[cond7_in19] in [1, 2]

            cond7_in2 = np.where(subgroups_id_molecule == 'CH2')
            bool72 = vki_molecule[cond7_in2] in [1, 2]

            bool73 = vki_molecule[cond7_in19] == vki_molecule[cond7_in2]

            cond7_in1 = np.where(subgroups_id_molecule == 'CH3')
            bool71 = vki_molecule[cond7_in1] in [1, 2]

        bool23.append(bool723 and bool719 and bool72 and bool73 and bool71)
        if bool723 and bool719 and bool72 and bool73 and bool71:

            index23 = cond7_in23[0] + index0

            index14 = np.where(subgroups == 'H2O')[0]

            boolN = secondorder.group_k == 'N'
            boolH2O = secondorder.group_l == 'H2O'
            eps2314 = secondorder[boolN & boolH2O]['eps_kl'].values
            lr2314 = secondorder[boolN & boolH2O]['lr_kl'].values

            index17 = np.where(subgroups == 'CO2')[0]

            boolN = secondorder.group_k == 'N'
            boolCO2 = secondorder.group_l == 'CO2'
            eps2317 = secondorder[boolN & boolCO2]['eps_kl'].values

            eps_kl[index23, index14] = eps2314
            eps_kl[index14, index23] = eps2314

            lr_kl[index14, index23] = lr2314
            lr_kl[index23, index14] = lr2314

            eps_kl[index17, index23] = eps2317
            eps_kl[index23, index17] = eps2317

        del bool723, bool719, bool72, bool73, bool71

    bool23 = np.hstack(bool23)
    cond_asso23_0 = np.any(bool23)
    cond_asso23_14 = np.any(subgroup_id_asso == 'H2O')
    cond_asso23_17 = np.any(subgroup_id_asso == 'CO2')
    cond_asso23_19 = np.any(subgroup_id_asso == 'CH2OH')

    where23all = np.where(subgroup_id_asso == 'N')
    where23all_id = molecule_id_index_asso[where23all]
    where23second = bool23[where23all_id]
    where23 = where23all[0][where23second]
    ncond23 = len(where23)

    if cond_asso23_0 and cond_asso23_14:

        where14 = np.where(subgroup_id_asso == 'H2O')

        boolk = secondasso.group_k == 'N'
        booll = secondasso.group_l == 'H2O'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]

        for k in range(ncond23):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values

                if siteK == 'e1' and siteL == 'H':

                    indexH = sites_cumsum[where14]
                    indexN = sites_cumsum[where23[k]]

                    epsAB_kl[indexH, indexN] = epsAB
                    epsAB_kl[indexN, indexH] = epsAB
                    kAB_kl[indexH, indexN] = kAB
                    kAB_kl[indexN, indexH] = kAB

    if cond_asso23_0 and cond_asso23_17:

        where17 = np.where(subgroup_id_asso == 'CO2')
        boolk = secondasso.group_k == 'N'
        booll = secondasso.group_l == 'CO2'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]

        for k in range(ncond23):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values

                if siteK == 'e1' and siteL == 'e1':
                    indexCO2 = sites_cumsum[where17]
                    indexN = sites_cumsum[where23[k]]

                    epsAB_kl[indexCO2, indexN] = epsAB
                    epsAB_kl[indexN, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexN] = kAB
                    kAB_kl[indexN, indexCO2] = kAB

                elif siteK == 'e1' and siteL == 'e2':

                    indexCO2 = sites_cumsum[where17] + 1
                    indexN = sites_cumsum[where23[k]]

                    epsAB_kl[indexCO2, indexN] = epsAB
                    epsAB_kl[indexN, indexCO2] = epsAB
                    kAB_kl[indexCO2, indexN] = kAB
                    kAB_kl[indexN, indexCO2] = kAB

    if cond_asso23_0 and cond_asso23_19:

        where19 = np.where(subgroup_id_asso == 'CH2OH')
        boolk = secondasso.group_k == 'N'
        booll = secondasso.group_l == 'CH2OH'
        df = secondasso[boolk & booll]
        len1 = df.shape[0]

        for k in range(ncond23):
            for j in range(len1):

                values = df.iloc[j].values[0:7]
                groupK2, siteK, groupL2, siteL, _, epsAB, kAB = values

                if siteK == 'e1' and siteL == 'H':

                    indexOH = sites_cumsum[where19]
                    indexN = sites_cumsum[where23[k]]

                    epsAB_kl[indexOH, indexN] = epsAB
                    epsAB_kl[indexN, indexOH] = epsAB
                    kAB_kl[indexOH, indexN] = kAB
                    kAB_kl[indexN, indexOH] = kAB
