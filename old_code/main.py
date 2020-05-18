#!/usr/bin/python
import logging
import math
import os
import shutil

import numpy
import spline as sp
from scipy.optimize import minimize

# data_file = '5CB.dat'
data_file = '5CB_new_perp.dat'
# data_file = 'gd.dat'
# data_file = 'Dy 17-17.dat'

g_type = ('perp', 'on')
# g_type='par'


global F
f = open(data_file, 'r')
lines = f.readlines()
H = []
eps_exp = []
for x in lines:
    H_n = float(x.split('\t')[0])
    try:
        if (g_type[0] == 'perp'):
            e_n = float(x.split('\t')[2])  # perp
        elif (g_type[0] == 'par'):
            e_n = float(x.split('\t')[1])  # par
        H.append(H_n)
        eps_exp.append(e_n)
    except:
        ()
#        print('blank line' )
f.close()

try:
    os.makedirs('graph')
except:
    shutil.rmtree('graph')
    os.makedirs('graph')

best_eps_error = numpy.inf

# e_perp=6.35	#gd
# e_a=-2.15
# chi_tab=0.0018
# M=1812.0
# e_perp=6.0	#dy
# e_a=-2.0
# chi_tab=0.0014
# M=1812.0
e_perp = 6.9  # 5CB
e_par = 20.04367
e_a = e_par - e_perp
chi_tab = 0.000028427
M = 249.36
rho = 1.0

d = 0.025 / 2
U = 0.000333564
chi = chi_tab * rho / M
print('chi =', chi)
E_0 = U / d
f_E = e_a * E_0 * E_0 / (4 * math.pi)
print('F_E =', f_E)
Norm = f_E * d * d
print('Norm =', Norm)
theta_0 = math.acos(math.sqrt((eps_exp[0] - e_perp) / e_a))
print('theta_0 = ', theta_0)
BOUND = [theta_0, theta_0]

print('H =\n' + str(H))
print('eps exp =\n' + str(eps_exp))
print('eps_perp = ', e_perp, ' eps_par = ', e_par, ' eps_a = ', e_a)
print('BOUND = ' + str(BOUND))

INTEGRATION_STEPS = 1000
MAX_SPLINE_POINTS = 10
lam = 0.1
MIN_POINTS = 100


def get_eps_error(K_vec):
    logging.info('New K = ' + str(K_vec))
    print('New K = ' + str(K_vec))
    if K_vec[0] < 0.0:
        return numpy.inf
    if K_vec[1] < -1.0:
        return numpy.inf
    if K_vec[1] > 1.0:
        return numpy.inf
    ee = []
    for x in H:
        F = chi * x * x / f_E
        f_0 = [math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499]
        spx = []
        for i in range(len(f_0)):
            spx.append(float(i) / (len(f_0) - 1))
        spl = sp.functional(x=spx, f=f_0, bound=BOUND, F=F, K=K_vec[0], K_as=K_vec[1], type=g_type)
        for k in range(3, MAX_SPLINE_POINTS):
            b = []
            for i in range(0, len(f_0)): b.append((0.0, math.pi * 0.5))
            b = tuple(b)
            solution = minimize(spl.wp, f_0, method='L-BFGS-B', bounds=b)
            # if F>50.0:
            #	print solution
            #	raw_input()
            # print solution.fun
            # print solution.x
            if k < MAX_SPLINE_POINTS - 1:
                new_point = spl.get_new_point()
                # print solution.x,type(solution.x)
                if type(solution.x) == float or type(solution.x) == numpy.float64:
                    f_0 = []
                    if new_point[0] == 1:
                        f_0.append(spl.get_value(new_point[1], new_point[0]))
                        f_0.append(solution.x)
                    else:
                        f_0.append(solution.x)
                        f_0.append(spl.get_value(new_point[1], new_point[0]))
                else:
                    f_0 = solution.x.tolist()
                    f_0.insert(new_point[0] - 1, spl.get_value(new_point[1], new_point[0]))
                # print x0
                spl.set_new_point(new_point[0], new_point[1])
            else:
                f = []
                f.append(BOUND[0])
                f.extend(solution.x.tolist())
                f.append(BOUND[1])
                spl.reset_spline(f)
            # print 'at F = ', F,' solution is:\n', f,'\nwith error = ', sp.min_value()
            # raw_input()
        eps = 0.0
        for i in range(0, INTEGRATION_STEPS):
            c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
            eps = eps + 1 / (INTEGRATION_STEPS * (e_perp + e_a * c * c))
        ee.append(1 / eps)
    i = 0
    eps_error = 0.0
    while i < len(ee):
        eps_error = eps_error + (ee[i] - eps_exp[i]) ** 2
        i = i + 1
    logging.info('with eps_error = ' + str(math.sqrt(eps_error)))
    print('with eps_error = ' + str(math.sqrt(eps_error)))
    global best_eps_error
    if math.sqrt(eps_error) < best_eps_error:
        best_eps_error = math.sqrt(eps_error)
    print('best eps_error = ' + str(best_eps_error))
    return math.sqrt(eps_error)


# solution = basinhopping(get_eps_error, [0.022, -0.14], niter=10, disp=True)
# K_n = solution.x[0]
# K_as = solution.x[1]

# K=0.46175836	#gd
# K_as=-0.04671133
# K=0.02115597	#Dy
# K_as=-0.14912428
# K_n=6.0	#5CB 0.02398339 -0.18651024
# K_as=-0.14*0

# K_n=0.1
# K_as=0.0

K_n = 6.186816  # 5CB_id
K_as = -0.138888889

print('K_n = ' + str(K_n))
print('K_as = ' + str(K_as))
print('Final K = ', K_n * Norm)
print('K_1 = ', K_n * Norm * (1.0 + K_as))
print('K_3 = ', K_n * Norm * (1.0 - K_as))

ee = []
for x in H:
    F = chi * x * x / f_E
    print('H = ', x, 'f_H/f_E = ', F)
    #    if F<1.0:   F=1.0
    f_0 = [BOUND[0], math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, BOUND[1]]
    spx = []
    for i in range(len(f_0)):
        spx.append(float(i) / (len(f_0) - 1))
    spl = sp.functional(x=spx, f=f_0, bound=BOUND, F=F, K=K_n, K_as=K_as, type=g_type)
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
            spl.graph('graph/' + str(x))
    eps = 0.0
    for i in range(0, INTEGRATION_STEPS):
        c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
        eps = eps + 1 / (INTEGRATION_STEPS * (e_perp + e_a * c * c))
    ee.append(1 / eps)
i = 0
eps_error = 0.0
while i < len(ee):
    eps_error = eps_error + (ee[i] - eps_exp[i]) ** 2
    i = i + 1
print('final eps_error = ', math.sqrt(eps_error))
f = open('eps.dat', 'w')
i = 0
while i < len(H):
    f.write(str(H[i]) + '\t' + str(ee[i]) + '\n')
    i = i + 1
f.close()
