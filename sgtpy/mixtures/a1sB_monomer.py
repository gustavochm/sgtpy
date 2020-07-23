import numpy as np
from .a1s_monomer import a1s, da1s_dxhi00, d2a1s_dxhi00, d3a1s_dxhi00, da1s_dx, da1s_dx_dxhi00
from .B_monomer import B, dB_dxhi00, d2B_dxhi00, d3B_dxhi00, dB_dx, dB_dx_dxhi00

def a1sB(xhi00, xhix, xm, x0, lam, cictes, a1vdw, a1vdw_cte):
    a1 = a1s(xhi00, xhix, xm, cictes, a1vdw)
    b = B(xhi00, xhix, xm, x0, lam, a1vdw_cte)
    return a1 + b

def da1sB_dxhi00(xhi00, xhix, xm, x0, lam, cictes,
                 a1vdw, a1vdw_cte, dxhix_dxhi00):
    
    a1, da1 = da1s_dxhi00(xhi00, xhix, xm, cictes,
                                 a1vdw, dxhix_dxhi00)
    
    b, db = dB_dxhi00(xhi00, xhix, xm, x0,  lam ,
                      a1vdw_cte, dxhix_dxhi00)
    
    return a1 + b, da1 + db

def d2a1sB_dxhi00(xhi00, xhix, xm, x0, lam, cictes,
                  a1vdw, a1vdw_cte, dxhix_dxhi00):
    
    a1, da1, d2a1 = d2a1s_dxhi00(xhi00, xhix, xm, cictes,
                                 a1vdw, dxhix_dxhi00)
    b, db, d2b = d2B_dxhi00(xhi00, xhix, xm, x0,  lam ,
                            a1vdw_cte, dxhix_dxhi00)
    return a1 + b, da1 + db, d2a1 + d2b

def d3a1sB_dxhi00(xhi00, xhix, xm, x0, lam, cictes,
                  a1vdw, a1vdw_cte, dxhix_dxhi00):
    
    a1, da1, d2a1, d3a1 = d3a1s_dxhi00(xhi00, xhix, xm, cictes,
                                       a1vdw, dxhix_dxhi00)
    
    b, db, d2b, d3b = d3B_dxhi00(xhi00, xhix, xm, x0, lam ,
                                 a1vdw_cte, dxhix_dxhi00)
    return a1 + b, da1 + db, d2a1 + d2b, d3a1 + d3b

#d3a1s_dxhi00(xhi00, xhix, xm, cictes, a1vdw, dxhix_dxhi00)
#d3B_dxhi00(xhi00, xhix, xm, x0, lam, a1vdw_cte, dxhix_dxhi00)

def da1sB_dx(xhi00, xhix, xm, ms, x0, lam, cictes,
             a1vdw, a1vdw_cte, dxhix_dx):
    a1 = da1s_dx(xhi00, xhix, xm, ms, cictes, a1vdw, dxhix_dx)
    
    b = dB_dx(xhi00, xhix, xm, ms, x0, lam, a1vdw_cte,  dxhix_dx)
    return a1 + b

def da1sB_dx_dxhi00(xhi00, xhix, xm, ms, x0, lam, cictes,
                    a1vdw, a1vdw_cte, dxhix_dxhi00,
                    dxhix_dx,  dxhix_dx_dxhi00):
    
    a1 = da1s_dx_dxhi00(xhi00, xhix, xm, ms, cictes, a1vdw,
                 dxhix_dxhi00, dxhix_dx,  dxhix_dx_dxhi00)
    
    b = dB_dx_dxhi00(xhi00, xhix, xm, ms, x0, lam, a1vdw_cte,
                     dxhix_dxhi00, dxhix_dx, dxhix_dx_dxhi00)
    
    return a1 + b

#xhi00, xhix, xm, x0, lam, cictes, a1vdw, a1vdw_cte
def a1sB_eval(xhi00, xhix, x0, xm, lambdas, cctes, a1vdw, a1vdw_cte):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw
    
    a1sb_a = a1sB(xhi00, xhix, xm, x0,  laij, cctes_laij, a1vdw_laij, a1vdw_cte)
    a1sb_r = a1sB(xhi00, xhix, xm, x0,  lrij, cctes_lrij, a1vdw_lrij, a1vdw_cte)
    a1sb_2a = a1sB(xhi00, xhix, xm, x0,  2*laij, cctes_2laij, a1vdw_2laij, a1vdw_cte)
    a1sb_2r = a1sB(xhi00, xhix, xm, x0,  2*lrij, cctes_2lrij, a1vdw_2lrij, a1vdw_cte)
    a1sb_ar = a1sB(xhi00, xhix, xm, x0,  larij, cctes_larij, a1vdw_larij, a1vdw_cte)
    
    a1sb_a1 = np.array([a1sb_a, a1sb_r])
    a1sb_a2 = np.array([a1sb_2a, a1sb_ar, a1sb_2r])
    return a1sb_a1, a1sb_a2

def da1sB_dxhi00_eval(xhi00, xhix, x0, xm, lambdas, cctes, a1vdw, a1vdw_cte, dxhix_dxhi00 ):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw 
    
    a1sb_a, da1sb_a = da1sB_dxhi00(xhi00, xhix, xm,  x0, laij, cctes_laij, 
                                   a1vdw_laij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_r, da1sb_r = da1sB_dxhi00(xhi00, xhix, xm,  x0, lrij, cctes_lrij,
                                   a1vdw_lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2a, da1sb_2a = da1sB_dxhi00(xhi00, xhix, xm,  x0, 2*laij, cctes_2laij,
                                     a1vdw_2laij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2r, da1sb_2r = da1sB_dxhi00(xhi00, xhix, xm,  x0, 2*lrij, cctes_2lrij,
                                     a1vdw_2lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_ar, da1sb_ar = da1sB_dxhi00(xhi00, xhix, xm, x0, larij, cctes_larij,
                                     a1vdw_larij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r]])
    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r]])
    
    return a1sb_a1, a1sb_a2

def d2a1sB_dxhi00_eval(xhi00, xhix, x0, xm, lambdas, cctes, a1vdw, a1vdw_cte, dxhix_dxhi00 ):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw
    
    a1sb_a, da1sb_a, d2a1sb_a = d2a1sB_dxhi00(xhi00, xhix, xm, x0, laij, cctes_laij,
                                              a1vdw_laij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_r, da1sb_r, d2a1sb_r = d2a1sB_dxhi00(xhi00, xhix, xm,  x0, lrij, cctes_lrij,
                                              a1vdw_lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2a, da1sb_2a, d2a1sb_2a = d2a1sB_dxhi00(xhi00, xhix, xm,  x0, 2*laij, cctes_2laij,
                                                 a1vdw_2laij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2r, da1sb_2r, d2a1sb_2r = d2a1sB_dxhi00(xhi00, xhix, xm, x0, 2*lrij, cctes_2lrij,
                                                 a1vdw_2lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_ar, da1sb_ar, d2a1sb_ar = d2a1sB_dxhi00(xhi00, xhix, xm, x0, larij, cctes_larij,
                                                 a1vdw_larij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r],
                       [d2a1sb_a, d2a1sb_r]])
    
    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r],
                       [d2a1sb_2a, d2a1sb_ar, d2a1sb_2r]])
    return a1sb_a1, a1sb_a2

def d3a1sB_dxhi00_eval(xhi00, xhix, x0, xm, lambdas, cctes, a1vdw, a1vdw_cte, dxhix_dxhi00 ):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw
    
    a1sb_a, da1sb_a, d2a1sb_a, d3a1sb_a = d3a1sB_dxhi00(xhi00, xhix, xm,  x0, laij, cctes_laij,
                                                        a1vdw_laij, a1vdw_cte, dxhix_dxhi00)
                                                        
    
    a1sb_r, da1sb_r, d2a1sb_r, d3a1sb_r = d3a1sB_dxhi00(xhi00, xhix, xm,  x0, lrij, cctes_lrij,
                                                        a1vdw_lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2a, da1sb_2a, d2a1sb_2a, d3a1sb_2a = d3a1sB_dxhi00(xhi00, xhix, xm, x0, 2*laij, cctes_2laij,
                                                            a1vdw_2laij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_2r, da1sb_2r, d2a1sb_2r, d3a1sb_2r = d3a1sB_dxhi00(xhi00, xhix, xm, x0, 2*lrij, cctes_2lrij,
                                                            a1vdw_2lrij, a1vdw_cte, dxhix_dxhi00)
    
    a1sb_ar, da1sb_ar, d2a1sb_ar, d3a1sb_ar = d3a1sB_dxhi00(xhi00, xhix, xm,  x0, larij, cctes_larij,
                                                            a1vdw_larij, a1vdw_cte, dxhix_dxhi00)

    a1sb_a1 = np.array([[a1sb_a, a1sb_r],
                        [da1sb_a, da1sb_r],
                       [d2a1sb_a, d2a1sb_r],
                       [d3a1sb_a, d3a1sb_r]])
    
    a1sb_a2 = np.array([[a1sb_2a, a1sb_ar, a1sb_2r],
                       [da1sb_2a, da1sb_ar, da1sb_2r],
                       [d2a1sb_2a, d2a1sb_ar, d2a1sb_2r],
                       [d3a1sb_2a, d3a1sb_ar, d3a1sb_2r]])
    
    return a1sb_a1, a1sb_a2

def da1sB_dx_eval(xhi00, xhix, x0, xm, ms, lambdas, cctes, a1vdw, a1vdw_cte, dxhix_dx):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw
    
    #da1sB_dx(xhi00, xhix, xm, ms, x0, lam, cictes,
    #         a1vdw, a1vdw_cte, dxhix_dx)
             
    a1sb_a = da1sB_dx(xhi00, xhix, xm, ms,  x0, laij, cctes_laij,
                    a1vdw_laij, a1vdw_cte, dxhix_dx)
        
    a1sb_r = da1sB_dx(xhi00, xhix, xm, ms, x0, lrij, cctes_lrij,
                    a1vdw_lrij, a1vdw_cte, dxhix_dx)
                                              
    a1sb_2a = da1sB_dx(xhi00, xhix, xm, ms, x0, 2*laij, cctes_2laij,
                    a1vdw_2laij, a1vdw_cte, dxhix_dx)
                                              
    a1sb_2r = da1sB_dx(xhi00, xhix, xm, ms, x0, 2*lrij, cctes_2lrij,
                    a1vdw_2lrij, a1vdw_cte, dxhix_dx)
                                              
    a1sb_ar = da1sB_dx(xhi00, xhix, xm, ms, x0, larij, cctes_larij,
                    a1vdw_larij, a1vdw_cte, dxhix_dx)
                                              
    a1sb_a1 = np.array([a1sb_a, a1sb_r])
                                              
    a1sb_a2 = np.array([a1sb_2a, a1sb_ar, a1sb_2r])
    
    return a1sb_a1, a1sb_a2

def da1sB_dx_dxhi00_eval(xhi00, xhix, x0, xm, ms, lambdas, cctes, a1vdw,
                         a1vdw_cte, dxhix_dxhi00,  dxhix_dx, dxhix_dx_dxhi00):
    
    laij, lrij, larij = lambdas
    cctes_laij, cctes_lrij, cctes_2laij, cctes_2lrij, cctes_larij = cctes
    a1vdw_laij, a1vdw_lrij, a1vdw_2laij, a1vdw_2lrij, a1vdw_larij = a1vdw
    
    #da1sB_dx(xhi00, xhix, xm, ms, x0, lam, cictes,
    #         a1vdw, a1vdw_cte, dxhix_dx)
        
    a1sb_a = da1sB_dx_dxhi00(xhi00, xhix, xm, ms,  x0, laij, cctes_laij,
                               a1vdw_laij, a1vdw_cte, dxhix_dxhi00,
                             dxhix_dx,  dxhix_dx_dxhi00)
             
    a1sb_r = da1sB_dx_dxhi00(xhi00, xhix, xm, ms, x0, lrij, cctes_lrij,
                               a1vdw_lrij, a1vdw_cte, dxhix_dxhi00,
                             dxhix_dx,  dxhix_dx_dxhi00)
             
    a1sb_2a = da1sB_dx_dxhi00(xhi00, xhix, xm, ms, x0, 2*laij, cctes_2laij,
                                a1vdw_2laij, a1vdw_cte, dxhix_dxhi00,
                              dxhix_dx,  dxhix_dx_dxhi00)
             
    a1sb_2r = da1sB_dx_dxhi00(xhi00, xhix, xm, ms, x0, 2*lrij, cctes_2lrij,
                                a1vdw_2lrij, a1vdw_cte, dxhix_dxhi00,
                              dxhix_dx,  dxhix_dx_dxhi00)
             
    a1sb_ar = da1sB_dx_dxhi00(xhi00, xhix, xm, ms, x0, larij, cctes_larij,
                                a1vdw_larij, a1vdw_cte, dxhix_dxhi00,
                              dxhix_dx, dxhix_dx_dxhi00)
             
    a1sb_a1 = np.array([a1sb_a, a1sb_r])
             
    a1sb_a2 = np.array([a1sb_2a, a1sb_ar, a1sb_2r])
             
    return a1sb_a1, a1sb_a2


def x0lambda_eval(x0, la, lr, lar, laij, lrij, larij, diag_index):
    x0la = x0**laij
    x0lr = x0**lrij
    x02la = x0**(2*laij)
    x02lr = x0**(2*lrij)
    x0lar = x0**larij
    
    #Utilizados para evaluar a1 y a2 del monomero
    x0_a1 = np.array([x0la, -x0lr])
    x0_a2 = np.array([x02la, -2*x0lar, x02lr])
    
    #utilizados para evaluar g1 y g2 de la cadena
    x0_g1 = np.array([la * x0la[diag_index], -lr* x0lr[diag_index]])
    x0_g2 = np.array([la * x02la[diag_index], -lar * x0lar[diag_index], lr * x02lr[diag_index]])
    
    return x0_a1, x0_a2, x0_g1, x0_g2
