#!/usr/bin/python
import math

import numpy
import spline as sp
from scipy.optimize import minimize

# eps_perp=6.35	#gd
# eps_a=-2.15
# chi_tab=0.0018
# M=1812.0
# eps_perp=6.0	#dy
# eps_a=-2.0
# chi_tab=0.0014
# M=1812.0
eps_perp = 6.9  # 5CB
eps_par = 20.04367
eps_a = eps_par - eps_perp
chi_tab = 0.000028427
M = 249.36
rho = 1.0

H_max = 7000.0
H_step = 20.0

d = 0.5 * 0.025
U = 0.00333564
chi = chi_tab * rho / M
print('chi =', chi)
E_0 = U / d
f_E = eps_a * E_0 * E_0 / (4 * math.pi)
print('F_E =', f_E)
Norm = f_E * d * d
print('Norm =', Norm)

theta_0 = 0.05 * (math.pi * 0.5)
print('theta_0 = ', theta_0)

BOUND = [theta_0, theta_0]
print('eps_p =' + str(eps_perp))
print('eps_a =' + str(eps_a))
print('BOUND = ' + str(BOUND))

INTEGRATION_STEPS = 1000
MAX_SPLINeps_perpOINTS = 10
lam = 0.1
MIN_POINTS = 100

K_n = 6.0
K_as = -0.14

print('K_n = ' + str(K_n))
print('K_as = ' + str(K_as))
print('Final K = ', K_n * Norm)
print('K_1 = ', K_n * Norm * (1.0 + K_as))
print('K_3 = ', K_n * Norm * (1.0 - K_as))

ee = []
x = 0.0
while x <= H_max:
    F = chi * x * x / f_E
    print('H = ', x, 'f_H/f_E = ', F)
    #    if F<1.0:   F=1.0
    f_0 = [BOUND[0], math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, BOUND[1]]
    spx = []
    for i in range(len(f_0)):
        spx.append(float(i) / (len(f_0) - 1))
    spl = sp.functional(x=spx, f=f_0, bound=BOUND, F=F, K=K_n, K_as=K_as, type='perp')
    for k in range(3, MAX_SPLINeps_perpOINTS):
        b = []
        for i in range(0, len(f_0)): b.append((0.0, math.pi * 0.5))
        b = tuple(b)
        solution = minimize(spl.wp, f_0, method='L-BFGS-B', bounds=b)
        if k < MAX_SPLINeps_perpOINTS - 1:
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
            spl.graph(x)
    eps = 0.0
    for i in range(0, INTEGRATION_STEPS):
        c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
        eps = eps + 1 / (INTEGRATION_STEPS * (eps_perp + eps_a * c * c))
    ee.append(1 / eps)
    x += H_step
f = open('eps_teor_graph.dat', 'w')
i = 0
while i < len(ee):
    f.write(str(H_step * float(i)) + '\t' + str(ee[i]) + '\n')
    i = i + 1
f.close()
