import math
import os
import sys

import new_spline as sp

sys.path.append("numpy_path")
import shutil
import time
import numpy
from scipy.optimize import minimize
from shutil import copyfile

folder_name = str(int(time.time()))
print(folder_name)
try:
    os.makedirs(folder_name)
except:
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)

copyfile(__file__, folder_name + '/code')

# eps_perp=6.35	#gd
# eps_a=-2.15
# chi_tab=0.0018
# M=1812.0
# eps_perp=6.0	#dy
# eps_a=-2.0
# chi_tab=0.0014
# M=1812.0

for iter in range(0, 9):
    print("=========================================================================\n iter = ", iter)
    os.makedirs(folder_name + '/' + str(iter))
    # eps_perp = 6.35  # gd
    # eps_a = -4.15
    # chi_tab = 0.001805
    # M = 1812.0
    eps_perp = 6.9  # 5CB
    eps_par = 20.04367
    eps_a = eps_par - eps_perp
    chi_tab = 0.000028427
    M = 249.36
    rho = 1.0
    phi = 4.65507 / 100 * 20  # 0.000465507 -> phi_n = 100000 at U=0.1V, d= 0.125

    H_max = 7000.0
    H_step = 50.0

    d = 0.5 * 0.025
    U = 0.00333564 * 0.1  # 0.00333564 = 1V
    chi = chi_tab * rho / M
    print('chi =', chi)
    E_0 = U / d
    f_E = abs(eps_a * E_0 * E_0 / (4 * math.pi))
    print('F_E =', f_E)
    Norm = f_E * d * d
    print('Norm =', Norm)

    theta_0 = (math.pi * 0.5) * iter / 9
    print('theta_0 = ', theta_0)

    g_type = ['perp', 'on']

    BOUND = [theta_0, theta_0]
    print('eps_p =' + str(eps_perp))
    print('eps_a =' + str(eps_a))
    print('BOUND = ' + str(BOUND))

    INTEGRATION_STEPS = 1000
    MAX_SPLINE_POINTS = 10
    lam = 0.1
    MIN_POINTS = 100

    K_n = 6.186816
    K_as = -0.138888889 * 0

    print('K_n = ' + str(K_n))
    print('K_as = ' + str(K_as))
    print('Final K = ', K_n * Norm)
    print('K_1 = ', K_n * Norm * (1.0 + K_as))
    print('K_3 = ', K_n * Norm * (1.0 - K_as))

    ee = []
    x = 0.0
    while x <= H_max:
        phi_n = phi * d / Norm
        print('phi_n =', phi_n)
        F = chi * x * x / f_E
        print(iter, ' H = ', x, 'f_H/f_E = ', F)
        #    if F<1.0:   F=1.0
        f_0 = [BOUND[0], math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, BOUND[1]]
        spx = []
        for i in range(len(f_0)):
            spx.append(float(i) / (len(f_0) - 1))
        spl = sp.functional(x=spx, f=f_0, bound=BOUND, F=F, K=K_n, K_as=K_as, eps_n=eps_perp / eps_a, type=g_type,
                            phi=phi_n)
        for k in range(3, MAX_SPLINE_POINTS):
            b = []
            for i in range(0, len(f_0)): b.append((0.0, math.pi * 0.5))
            b = tuple(b)
            solution = minimize(spl.wp, f_0, method='L-BFGS-B', bounds=b)
            if k < MAX_SPLINE_POINTS - 1:
                new_point = spl.get_new_point()
                if type(solution.x) == float or type(solution.x) == numpy.float64:
                    if new_point[0] == 1:
                        f_0.append(spl.get_value(new_point[1], new_point[0]))
                        f_0.append(solution.x)
                    else:
                        f_0.append(solution.x)
                        f_0.append(spl.get_value(new_point[1], new_point[0]))
                else:
                    f_0 = solution.x.tolist()
                    f_0.insert(new_point[0] - 1, spl.get_value(new_point[1], new_point[0]))
                spl.set_new_point(new_point[0], new_point[1])
            else:
                f = solution.x.tolist()
                spl.reset_spline(f)
                spl.graph(folder_name + '/' + str(iter) + '/theta' + str(x) + '.dat')
        eps = 0.0
        for i in range(0, INTEGRATION_STEPS):
            c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
            eps = eps + 1 / (INTEGRATION_STEPS * (eps_perp + eps_a * c * c))
        ee.append(1 / eps)
        x += H_step
    f = open(folder_name + '/eps_' + str(iter) + '.dat', 'w')
    i = 0
    f.write('H;eps\n')
    while i < len(ee):
        f.write(str(H_step * float(i)) + ';' + str(ee[i]) + '\n')
        i = i + 1
    f.close()
