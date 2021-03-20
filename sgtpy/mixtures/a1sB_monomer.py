import numpy as np
from .a1s_monomer import a1s, da1s_dxhi00, d2a1s_dxhi00, d3a1s_dxhi00
from .B_monomer import B, dB_dxhi00, d2B_dxhi00, d3B_dxhi00
from .a1s_monomer import da1s_dx_dxhi00_dxxhi, da1s_dx_d2xhi00_dxxhi
from .B_monomer import dB_dx_dxhi00_dxxhi, dB_dx_d2xhi00_dxxhi


def a1sB(xhi00, xhix, xhix_vec, xm, Ilam, Jlam, cictes, a1vdw, a1vdw_cte):
    a1 = a1s(xhi00, xhix_vec, xm, cictes, a1vdw)
    b = B(xhi00, xhix, xm, Ilam, Jlam, a1vdw_cte)
    return a1 + b


def da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, Ilam, Jlam, cictes, a1vdw,
                 a1vdw_cte, dxhix_dxhi00):

    a1, da1 = da1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw, dxhix_dxhi00)
    b, db = dB_dxhi00(xhi00, xhix, xm, Ilam, Jlam, a1vdw_cte, dxhix_dxhi00)

    return a1 + b, da1 + db


def d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, Ilam, Jlam, cictes,
                  a1vdw, a1vdw_cte, dxhix_dxhi00):
    a1, da1, d2a1 = d2a1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw,
                                 dxhix_dxhi00)
    b, db, d2b = d2B_dxhi00(xhi00, xhix, xm, Ilam, Jlam, a1vdw_cte,
                            dxhix_dxhi00)
    return a1 + b, da1 + db, d2a1 + d2b


def d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, Ilam, Jlam, cictes, a1vdw,
                  a1vdw_cte, dxhix_dxhi00):
    a1, da1, d2a1, d3a1 = d3a1s_dxhi00(xhi00, xhix_vec, xm, cictes, a1vdw,
                                       dxhix_dxhi00)
    b, db, d2b, d3b = d3B_dxhi00(xhi00, xhix, xm, Ilam, Jlam, a1vdw_cte,
                                 dxhix_dxhi00)
    return a1 + b, da1 + db, d2a1 + d2b, d3a1 + d3b


def da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_ij,
                          J_ij, cctesij, a1vdwij, a1vdw_cteij,
                          dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):
    out = da1s_dx_dxhi00_dxxhi(xhi00, xhix_vec, xm, ms, cctesij, a1vdwij,
                               dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    a1, da1, da1x, da1xxhi = out
    out = dB_dx_dxhi00_dxxhi(xhi00, xhix, xm, ms, I_ij, J_ij, a1vdw_cteij,
                             dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    b, db, dbx, dbxxhi = out
    return a1+b, da1+db, da1x+dbx, da1xxhi+dbxxhi


def da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_ij,
                           J_ij, cctesij, a1vdwij, a1vdw_cteij,
                           dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):
    out = da1s_dx_d2xhi00_dxxhi(xhi00, xhix_vec, xm, ms, cctesij, a1vdwij,
                                dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    a1, da1, d2a1, da1x, da1xxhi = out
    out = dB_dx_d2xhi00_dxxhi(xhi00, xhix, xm, ms, I_ij, J_ij, a1vdw_cteij,
                              dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    b, db, d2b, dbx, dbxxhi = out
    return a1+b, da1+db, d2a1+d2b, da1x+dbx, da1xxhi+dbxxhi


def a1sB_eval(xhi00, xhix, xhix_vec, xm, I_lambdasij, J_lambdasij, cctesij,
              a1vdwij, a1vdw_cteij):

    # laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdasij
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdasij

    a1sb_a = a1sB(xhi00, xhix, xhix_vec, xm, I_la, J_la, cctes_laij,
                  a1vdw_laij, a1vdw_cteij)
    a1sb_r = a1sB(xhi00, xhix, xhix_vec, xm, I_lr, J_lr, cctes_lrij,
                  a1vdw_lrij, a1vdw_cteij)
    a1sb_2a = a1sB(xhi00, xhix, xhix_vec, xm, I_2la, J_2la, cctes_2laij,
                   a1vdw_2laij, a1vdw_cteij)
    a1sb_2r = a1sB(xhi00, xhix, xhix_vec, xm, I_2lr, J_2lr, cctes_2lrij,
                   a1vdw_2lrij, a1vdw_cteij)
    a1sb_ar = a1sB(xhi00, xhix, xhix_vec, xm, I_lar, J_lar, cctes_larij,
                   a1vdw_larij, a1vdw_cteij)

    a1sb_a1 = np.array([a1sb_a, a1sb_r])
    a1sb_a2 = np.array([a1sb_2a, a1sb_ar, a1sb_2r])
    return a1sb_a1, a1sb_a2


def da1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xm, I_lambdasij, J_lambdasij,
                      cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00):

    # laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdasij
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdasij

    a1sb_a, da1sb_a = da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_la, J_la,
                                   cctes_laij, a1vdw_laij, a1vdw_cteij,
                                   dxhix_dxhi00)

    a1sb_r, da1sb_r = da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_lr, J_lr,
                                   cctes_lrij, a1vdw_lrij, a1vdw_cteij,
                                   dxhix_dxhi00)

    a1sb_2a, da1sb_2a = da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_2la,
                                     J_2la, cctes_2laij, a1vdw_2laij,
                                     a1vdw_cteij, dxhix_dxhi00)

    a1sb_2r, da1sb_2r = da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_2lr,
                                     J_2lr, cctes_2lrij, a1vdw_2lrij,
                                     a1vdw_cteij, dxhix_dxhi00)

    a1sb_ar, da1sb_ar = da1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_lar,
                                     J_lar, cctes_larij, a1vdw_larij,
                                     a1vdw_cteij, dxhix_dxhi00)

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r]])
    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r]])

    return a1sb_a1, a1sb_a2


def d2a1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xm, I_lambdasij, J_lambdasij,
                       cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00):

    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdasij
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdasij

    out = d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm,  I_la, J_la,
                        cctes_laij, a1vdw_laij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_a, da1sb_a, d2a1sb_a = out

    out = d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_lr, J_lr,
                        cctes_lrij, a1vdw_lrij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_r, da1sb_r, d2a1sb_r = out

    out = d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm,  I_2la, J_2la,
                        cctes_2laij, a1vdw_2laij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_2a, da1sb_2a, d2a1sb_2a = out

    out = d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm,  I_2lr, J_2lr,
                        cctes_2lrij, a1vdw_2lrij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_2r, da1sb_2r, d2a1sb_2r = out

    out = d2a1sB_dxhi00(xhi00, xhix, xhix_vec, xm,  I_lar, J_lar,
                        cctes_larij, a1vdw_larij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_ar, da1sb_ar, d2a1sb_ar = out

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r],
                       [d2a1sb_a, d2a1sb_r]])

    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r],
                       [d2a1sb_2a, d2a1sb_ar, d2a1sb_2r]])
    return a1sb_a1, a1sb_a2


def d3a1sB_dxhi00_eval(xhi00, xhix, xhix_vec, xm, I_lambdasij, J_lambdasij,
                       cctesij, a1vdwij, a1vdw_cteij, dxhix_dxhi00):

    # laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_la, I_lr, I_2la, I_2lr, I_lar = I_lambdasij
    J_la, J_lr, J_2la, J_2lr, J_lar = J_lambdasij

    out = d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_la, J_la, cctes_laij,
                        a1vdw_laij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_a, da1sb_a, d2a1sb_a, d3a1sb_a = out

    out = d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_lr, J_lr, cctes_lrij,
                        a1vdw_lrij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_r, da1sb_r, d2a1sb_r, d3a1sb_r = out

    out = d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_2la, J_2la,
                        cctes_2laij, a1vdw_2laij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_2a, da1sb_2a, d2a1sb_2a, d3a1sb_2a = out

    out = d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_2lr, J_2lr,
                        cctes_2lrij, a1vdw_2lrij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_2r, da1sb_2r, d2a1sb_2r, d3a1sb_2r = out

    out = d3a1sB_dxhi00(xhi00, xhix, xhix_vec, xm, I_lar, J_lar,
                        cctes_larij, a1vdw_larij, a1vdw_cteij, dxhix_dxhi00)
    a1sb_ar, da1sb_ar, d2a1sb_ar, d3a1sb_ar = out

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r],
                       [d2a1sb_a, d2a1sb_r],
                       [d3a1sb_a, d3a1sb_r]])

    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r],
                       [d2a1sb_2a, d2a1sb_ar, d2a1sb_2r],
                       [d3a1sb_2a, d3a1sb_ar, d3a1sb_2r]])

    return a1sb_a1, a1sb_a2


def da1sB_dx_dxhi00_dxxhi_eval(xhi00, xhix, xhix_vec, xm, ms, I_lambdasij,
                               J_lambdasij, cctesij, a1vdwij, a1vdw_cteij,
                               dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):

    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_laij, I_lrij, I_2laij, I_2lrij, I_larij = I_lambdasij
    J_laij, J_lrij, J_2laij, J_2lrij, J_larij = J_lambdasij

    out_la = da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_laij,
                                   J_laij, cctes_laij, a1vdw_laij, a1vdw_cteij,
                                   dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    a1sb_a, da1sb_a, da1sb_ax, da1sb_axxhi = out_la
    out_lr = da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_lrij,
                                   J_lrij, cctes_lrij, a1vdw_lrij, a1vdw_cteij,
                                   dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    a1sb_r, da1sb_r, da1sb_rx, da1sb_rxxhi = out_lr
    out_2la = da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_2laij,
                                    J_2laij, cctes_2laij, a1vdw_2laij,
                                    a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                    dxhix_dx_dxhi00)
    a1sb_2a, da1sb_2a, da1sb_2ax, da1sb_2axxhi = out_2la
    out_2lr = da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_2lrij,
                                    J_2lrij, cctes_2lrij, a1vdw_2lrij,
                                    a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                    dxhix_dx_dxhi00)
    a1sb_2r, da1sb_2r, da1sb_2rx, da1sb_2rxxhi = out_2lr
    out_lar = da1sB_dx_dxhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_larij,
                                    J_larij, cctes_larij, a1vdw_larij,
                                    a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                    dxhix_dx_dxhi00)
    a1sb_ar, da1sb_ar, da1sb_arx, da1sb_arxxhi = out_lar

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r]])

    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r]])

    a1sb_a1x = np.array([da1sb_ax, da1sb_rx])
    a1sb_a2x = np.array([da1sb_2ax, da1sb_arx, da1sb_2rx])

    a1sb_a1xxhi = np.array([da1sb_axxhi, da1sb_rxxhi])
    a1sb_a2xxhi = np.array([da1sb_2axxhi, da1sb_arxxhi, da1sb_2rxxhi])

    return a1sb_a1, a1sb_a2, a1sb_a1x, a1sb_a2x, a1sb_a1xxhi, a1sb_a2xxhi


def da1sB_dx_d2xhi00_dxxhi_eval(xhi00, xhix, xhix_vec, xm, ms, I_lambdasij,
                                J_lambdasij, cctesij, a1vdwij, a1vdw_cteij,
                                dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00):

    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctesij
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdwij
    I_laij, I_lrij, I_2laij, I_2lrij, I_larij = I_lambdasij
    J_laij, J_lrij, J_2laij, J_2lrij, J_larij = J_lambdasij

    out_la = da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_laij,
                                    J_laij, cctes_laij, a1vdw_laij,
                                    a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                    dxhix_dx_dxhi00)
    a1sb_a, da1sb_a, d2a1sb_a, da1sb_ax, da1sb_axxhi = out_la
    out_lr = da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_lrij,
                                    J_lrij, cctes_lrij, a1vdw_lrij,
                                    a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                    dxhix_dx_dxhi00)
    a1sb_r, da1sb_r, d2a1sb_r, da1sb_rx, da1sb_rxxhi = out_lr
    out_2la = da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_2laij,
                                     J_2laij, cctes_2laij, a1vdw_2laij,
                                     a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                     dxhix_dx_dxhi00)
    a1sb_2a, da1sb_2a, d2a1sb_2a, da1sb_2ax, da1sb_2axxhi = out_2la
    out_2lr = da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_2lrij,
                                     J_2lrij, cctes_2lrij, a1vdw_2lrij,
                                     a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                     dxhix_dx_dxhi00)
    a1sb_2r, da1sb_2r, d2a1sb_2r, da1sb_2rx, da1sb_2rxxhi = out_2lr
    out_lar = da1sB_dx_d2xhi00_dxxhi(xhi00, xhix, xhix_vec, xm, ms, I_larij,
                                     J_larij, cctes_larij, a1vdw_larij,
                                     a1vdw_cteij, dxhix_dxhi00, dxhix_dx,
                                     dxhix_dx_dxhi00)
    a1sb_ar, da1sb_ar, d2a1sb_ar, da1sb_arx, da1sb_arxxhi = out_lar

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r],
                        [d2a1sb_a, d2a1sb_r]])

    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r],
                       [d2a1sb_2a, d2a1sb_ar, d2a1sb_2r]])

    a1sb_a1x = np.array([da1sb_ax, da1sb_rx])
    a1sb_a2x = np.array([da1sb_2ax, da1sb_arx, da1sb_2rx])

    a1sb_a1xxhi = np.array([da1sb_axxhi, da1sb_rxxhi])
    a1sb_a2xxhi = np.array([da1sb_2axxhi, da1sb_arxxhi, da1sb_2rxxhi])

    return a1sb_a1, a1sb_a2, a1sb_a1x, a1sb_a2x, a1sb_a1xxhi, a1sb_a2xxhi


def x0lambda_eval(x0, la, lr, lar, laij, lrij, larij, diag_index):
    x0la = x0**laij
    x0lr = x0**lrij
    x02la = x0**(2*laij)
    x02lr = x0**(2*lrij)
    x0lar = x0**larij

    # To be used for a1 and a2 of monomer
    x0_a1 = np.array([x0la, -x0lr])
    x0_a2 = np.array([x02la, -2*x0lar, x02lr])

    # To be used in g1 and g2 of chain
    x0_g1 = np.array([la * x0la[diag_index], -lr*x0lr[diag_index]])
    x0_g2 = np.array([la * x02la[diag_index], -lar*x0lar[diag_index],
                     lr * x02lr[diag_index]])

    return x0_a1, x0_a2, x0_g1, x0_g2
