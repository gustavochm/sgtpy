from __future__ import division, print_function, absolute_import
import numpy as np
from .constants import kb, Na
from warnings import warn

# parameters for linear molecules
ms1 = np.array([[14.8359, 22.2019, 7220.9599, 23193.475, -6207.4663, 1732.96],
               [0., -6.9630, 468.7358, -983.6038, 914.3608, -1383.4441],
               [0.1284, 1.6772, 0., 0., 0., 0.],
               [0., 0.4049, -0.1592, 0., 0., 0.],
               [1.8966, -6.9808, 10.6330, -9.2041, 4.2503, 0.],
               [0., -1.6205, -0.8019, 1.7086, -0.5333, 1.0536]])

ms2 = np.array([[8.0034, -22.5111, 3.5750, 60.3129, 0., 0.],
               [0., -5.2669, 10.2299, -6.4860, 0., 0.],
               [0.1125, 1.5404, -5.8769, 5.2427, 0., 0.],
               [0., -3.1964, 2.5174, 0.3518, -0.1654, 0.],
               [-0.0696, -1.9440, 6.2575, -5.4431, 0.8731, 0.],
               [0., -10.5646, 25.4914, -20.5091, 3.6753, 0.]])

ms3 = np.array([[6.9829, -13.7097, -1.9604, 17.3237, 0, 0],
               [0., -3.869, 5.2519, -2.3637, 0., 0.],
               [-0.2092, 4.2672, -9.7703, 4.8661, -0.1950, 4.2125],
               [0., -1.3778, -2.4836, 3.5280, 0.7918, -0.1246],
               [0.0656, -1.4630, 3.6991, -2.5081, 0., 0.],
               [0., -8.9309, 18.9584, -11.6668, -0.2561, 0.]])

ms4 = np.array([[6.4159, -34.3656, 59.6108, -21.6579, -35.8210, 27.2358],
               [0, -6.9751, 19.2063, -26.0583, 17.4222, -4.5757],
               [0.1350, 1.3115, -10.1437, 24.0729, -24.8084, 9.7109],
               [0., -5.8540, 13.3411, -14.3302, 6.7309, -0.7830],
               [0.1025, -1.1948, 2.8448, -1.9519, 0., 0.],
               [0., -8.1077, 16.7865, -9.6354, -1.2390, 0.]])

ms5 = np.array([[6.1284, -9.1568, -0.2229, 4.5311, 0., 0.],
               [0., -2.8486, 2.7828, -0.9030, 0., 0.],
               [0.1107, 1.9807, -6.6720, 5.4841, 0., 0.],
               [0, -3.1341, 2.7657, -0.2737, -0.0431, 0.],
               [0.1108, -0.9900, 2.2187, -1.5027, 0., 0.],
               [0., -7.3749, 14.5313, -7.4967, -1.9209, 0.]])

ms6 = np.array([[5.9217, -8.0711, 0.4264, 2.5600, 0., 0.],
               [0., -2.5291, 2.1864, -0.6298, 0., 0.],
               [0.1302, 1.9357, -6.4591, 5.1864, 0., 0.],
               [0, -3.1078, 2.8058, -0.4375, 0., 0.],
               [0.2665, -0.4268, -0.2732, 0.6486, 0., 0.],
               [0., 1.7499, -10.1370, 9.4381, 0., 0.]])

# chain like molecules
ms3_a = np.array([[6.9725, -24.8851, 119.5032, -232.6325, 177.9879, 0.],
                 [0., -5.2703, 22.6810, -54.7374, 62.6967, -26.6382],
                 [-0.1963, 3.8429, -15.5963, 32.6737, -39.1147, 20.9199],
                 [0., -5.0496, 12.6694, -17.6981, 11.2147, -1.1138],
                 [0.0991, -1.5927, 4.6296, -4.6918, 1.4552, 0.],
                 [0., -9.1703, 23.2400, -22.1380, 6.6497, 0.]])

ms4_b = np.array([[6.5137, -14.8541, 8.2147, 2.3634, 0., 0.],
                 [0., -3.6723, 4.9632, -2.7691, 0.5017, 0.],
                 [0.4901, -1.3513, -5.1730, 18.0352, -13.9409, 0.],
                 [0., -6.8977, 15.1764, -12.8773, 3.3297, -0.4160],
                 [0.1281, -1.2349, 3.0728, -2.7357, 0.6759, 0.],
                 [0., -7.8857, 18.1466, -15.3313, 3.5466, 0.]])

ms5_c = np.array([[5.9974, -25.4418, 35.1304, -8.7779, -17.4560, 10.8151],
                 [0., -5.6278, 12.6813, -14.2839, 8.0207, -1.7843],
                 [-0.0574, 3.1522, -14.6517, 28.4219, -27.0356, 10.9765],
                 [0., -5.0723, 10.5536, -11.0175, 5.3725, -0.5705],
                 [0.0975, -1.0255, 3.0131, -3.6701, 1.9253, -0.3682],
                 [0., -8.5586, 23.1998, -26.9186, 13.5840, -2.5091]])

ms5_d = np.array([[5.9870, -26.3670, 38.9070, -13.7206, -15.7141, 11.2597],
                 [0., -5.7497, 13.2677, -15.3524, 8.8875, -2.0466],
                 [0.4786, -1.0367, -5.8418, 17.9103, -13.1351, 0.],
                 [0., -6.5275, 13.9620, -11.7622, 3.2367, -0.4277],
                 [0.1148, -1.0737, 3.0574, -3.7133, 1.9717, -0.3814],
                 [0., -8.2345, 22.1442, -25.9044, 13.3190, -2.4978]])

ms7_e = np.array([[5.2601, -18.0152, 21.6901, -6.0489, -6.7239, 4.0247],
                 [0., -4.6291, 8.9896, -9.0082, 4.6002, -0.9466],
                 [0.4847, -0.6085, 6.8858, 17.6535, -11.6816, 0.],
                 [0., -5.9592, 11.9474, -9.6061, 2.6874, -0.3546],
                 [0.0960, -0.7875, 2.1158, -2.5024, 1.3246, -0.2650],
                 [0., -7.7202, 20.0622, -23.1345, 11.9742, -2.3639]])

ms7_f = np.array([[5.3804, -19.2987, 22.5961, -4.5702, -8.5162, 4.4222],
                 [0., -4.7382, 9.0450, -8.6820, 4.1780, -0.8024],
                 [0.4746, -0.5674, -7.0132, 17.9753, -11.9759, 0.],
                 [0., -5.9993, 12.1102, -9.7991, 2.7465, -0.3637],
                 [0.0930, -0.7801, 2.1084, -2.4943, 1.3220, -0.2691],
                 [0., -7.8190, 20.3990, -23.5642, 12.2818, -2.4951]])
"""
a = np.array([[14.8359, 22.2019, 7220.9599, 23193.475, -6207.4663, 1732.96],
              [8.0034, -22.5111, 3.5750, 60.3129, 0., 0.],
              [6.9829, -13.7097, -1.9604, 17.3237, 0, 0],
              [6.4159, -34.3656, 59.6108, -21.6579, -35.8210, 27.2358],
              [6.1284, -9.1568, -0.2229, 4.5311, 0., 0.],
              [5.9217, -8.0711, 0.4264, 2.5600, 0., 0.]])

b = np.array([[0, -6.9630, 468.7358, -983.6038, 914.3608, -1383.4441],
              [0, -5.2669, 10.2299, -6.4860, 0., 0.],
              [0, -3.869, 5.2519, -2.3637, 0., 0.],
              [0, -6.9751, 19.2063, -26.0583, 17.4222, -4.5757],
              [0., -2.8486, 2.7828, -0.9030, 0., 0.],
              [0, -2.5291, 2.1864, -0.6298, 0., 0.]])

c = np.array([[0.1284, 1.6772, 0., 0., 0., 0.],
             [0.1125, 1.5404, -5.8769, 5.2427, 0., 0.],
             [-0.2092, 4.2672, -9.7703, 4.8661, -0.1950, 4.2125],
             [0.1350, 1.3115, -10.1437, 24.0729, -24.8084, 9.7109],
             [0.1107, 1.9807, -6.6720, 5.4841, 0., 0.],
             [0.1302, 1.9357, -6.4591, 5.1864, 0., 0.]])

d = np.array([[0., 0.4049, -0.1592, 0., 0., 0.],
             [0., -3.1964, 2.5174, 0.3518, -0.1654, 0.],
             [0, -1.3778, -2.4836, 3.5280, 0.7918, -0.1246],
             [0., -5.8540, 13.3411, -14.3302, 6.7309, -0.7830],
             [0, -3.1341, 2.7657, -0.2737, -0.0431, 0.],
             [0, -3.1078, 2.8058, -0.4375, 0., 0.]])

j = np.array([[1.8966, -6.9808, 10.6330, -9.2041, 4.2503, 0.],
              [-0.0696, -1.9440, 6.2575, -5.4431, 0.8731, 0.],
              [0.0656, -1.4630, 3.6991, -2.5081, 0., 0.],
              [0.1025, -1.1948, 2.8448, -1.9519, 0., 0.],
              [0.1108, -0.9900, 2.2187, -1.5027, 0., 0.],
              [0.2665, -0.4268, -0.2732, 0.6486, 0., 0.]])

k = np.array([[0, -1.6205, -0.8019, 1.7086, -0.5333, 1.0536],
             [0, -10.5646, 25.4914, -20.5091, 3.6753, 0.],
             [0., -8.9309, 18.9584, -11.6668, -0.2561, 0.],
             [0, -8.1077, 16.7865, -9.6354, -1.2390, 0.],
             [0., -7.3749, 14.5313, -7.4967, -1.9209, 0.],
             [0., 1.7499, -10.1370, 9.4381, 0., 0.]])
"""


# Force Fields for Coarse-Grained Molecular Simulations
# from a Corresponding States Correlation. Mejia et al, 2014
def saft_forcefield(ms, Tc, w, rhol07, ring_type):
    lambda_a = 6.
    if ms == 1 and ring_type is None:
        wmin = -0.0847
        wmax = 0.2387
        matrix = ms1
        ring = 0.
    elif ms == 2 and ring_type is None:
        wmin = 0.0489
        wmax = 0.5215
        matrix = ms2
        ring = 0.
    elif ms == 3 and ring_type is None:
        wmin = 0.1326
        wmax = 0.7372
        matrix = ms3
        ring = 0.
    elif ms == 4 and ring_type is None:
        wmin = 0.2054
        wmax = 0.9125
        matrix = ms4
        ring = 0.
    elif ms == 5 and ring_type is None:
        wmin = 0.2731
        wmax = 1.0635
        matrix = ms5
        ring = 0.
    elif ms == 6 and ring_type is None:
        wmin = 0.3378
        wmax = 1.1974
        matrix = ms6
        ring = 0.
    elif ms == 3 and ring_type == 'a':
        wmin = 0.0745
        wmax = 0.7060
        matrix = ms3_a
        ring = 1.4938
    elif ms == 4 and ring_type == 'b':
        wmin = 0.1424
        wmax = 0.8533
        matrix = ms4_b
        ring = 2.9833
    elif ms == 5 and ring_type == 'c':
        wmin = 0.2061
        wmax = 1.0427
        matrix = ms5_c
        ring = 3.2222
    elif ms == 5 and ring_type == 'd':
        wmin = 0.2105
        wmax = 0.9738
        matrix = ms5_d
        ring = 4.7188
    elif ms == 7 and ring_type == 'e':
        wmin = 0.3493
        wmax = 1.1271
        matrix = ms7_e
        ring = 9.0958
    elif ms == 7 and ring_type == 'f':
        wmin = 0.3486
        wmax = 1.1467
        matrix = ms7_f
        ring = 8.8640
    else:
        raise Exception('ms and ring_type not supported')

    if w < wmin or w > wmax:
        warn('Acentric factor must be within '+str(wmin)+' and '+str(wmax)+' for ms = '+str(ms) + ' ring = ' + str(ring))

    ai, bi, ci, di, ji, ki = matrix
    """
    index = np.int(ms-1)
    ai = a[index]
    bi = b[index]
    ci = c[index]
    di = d[index]
    ji = j[index]
    ki = k[index]
    """
    exponent = np.arange(6)

    # Eq 13 lambda_r
    wi = w**exponent
    lambda_r = np.dot(ai, wi) / (1 + np.dot(bi, wi))

    # Eq 14 alpha VdW
    alpha = (lambda_r/(lambda_r-6))*(lambda_r/6)**(6/(lambda_r-6))*(1/3-1/(lambda_r-3))

    # Eq 15 T*c
    alphai = alpha**exponent
    Tc_r = np.dot(ci, alphai) / (1 + np.dot(di, alphai))

    # Eq 16 eps
    eps = Tc / Tc_r * kb

    # Eq 18 rho*c
    rhoc_r = np.dot(ji, alphai) / (1 + np.dot(ki, alphai))

    # Eq 17 sigma
    sigma = (rhoc_r / rhol07 / Na)**(1/3)
    return lambda_r, lambda_a, ms, eps, sigma, ring
