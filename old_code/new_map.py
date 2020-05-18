import math
import os
import sys

import new_spline as sp

sys.path.append("numpy_path")
import shutil
import numpy
from scipy.optimize import minimize
from shutil import copyfile

# data_file = '5CB.dat'
# data_file = '5CB_new_perp.dat'
# data_file = 'Gd 17-17_exp.dat'
# data_file = 'Dy 17-17.dat'
data_file = 'Er_17-17_exp.csv'

g_type = ['perp', 'on']
# g_type='par'

best_eps_error = numpy.inf

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

# folder_name="5CB map" #str(int(time.time()))
folder_name = "Er map"  # str(int(time.time()))
print(folder_name)
try:
    os.makedirs(folder_name)
except:
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)

copyfile(__file__, folder_name + '/code')

# eps_perp=6.35	#gd
# eps_par=3.91554#4.2
# chi_tab=0.0018
# M=1812.0
# phi = 0.465507*1000000
# eps_perp=6.0	#dy
# eps_a=-2.0
# chi_tab=0.0014
# M=1812.0
# eps_perp = 6.9  # 5CB
# eps_par=20.04367
# chi_tab = 0.000028427
# M = 249.36
# phi = 0.465507

eps_perp = 4.7  # 4.4472	#Er
eps_par = 3.3  # 3.54729
chi_tab = 0.009894
M = 1812.0
phi = 0.465507 * 1000000

rho = 1.0

eps_a = eps_par - eps_perp

d = 0.025 / 2
U = 0.000333564
chi = chi_tab * rho / M
print('chi =', chi)
E_0 = U / d
f_E = abs(eps_a * E_0 * E_0 / (4 * math.pi))
print('F_E =', f_E)
Norm = f_E * d * d
print('Norm =', Norm)
print((eps_exp[0] - eps_perp) / eps_a)
theta_0 = math.acos(math.sqrt((eps_exp[0] - eps_perp) / eps_a))
print('theta_0 = ', theta_0)
BOUND = [theta_0, theta_0]

print('H =\n' + str(H))
print('eps exp =\n' + str(eps_exp))
print('eps_perp = ', eps_perp, ' eps_par = ', eps_par, ' eps_a = ', eps_a)
print('BOUND = ' + str(BOUND))

INTEGRATION_STEPS = 1000
MAX_SPLINE_POINTS = 10
lam = 0.1
MIN_POINTS = 100

# K_n=6.186816
# K_as=-0.138888889

# K_n = 4853.55
# K_as = -0.7862595419847329

K_n = 25000
K_as = -0.8


def get_eps_error(K_vec):
    print('New K = ' + str(K_vec))
    if K_vec[0] < 0.0:
        return numpy.inf
    if K_vec[1] < -1.0:
        return numpy.inf
    if K_vec[1] > 1.0:
        return numpy.inf
    ee = []
    for x in H:
        phi_n = phi * d / Norm
        # print('phi_n =', phi_n)
        F = chi * x * x / (f_E)
        # print('H = ', x, 'f_H/f_E = ', F)
        #    if F<1.0:   F=1.0
        f_0 = [BOUND[0], math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, math.pi * 0.499, BOUND[1]]
        spx = []
        for i in range(len(f_0)):
            spx.append(float(i) / (len(f_0) - 1))
        spl = sp.functional(x=spx, f=f_0, bound=BOUND, F=F, K=K_vec[0], K_as=K_vec[1], eps_n=eps_perp / eps_a,
                            type=g_type,
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
                spl.graph('graph/' + str(x))
        eps = 0.0
        for i in range(0, INTEGRATION_STEPS):
            c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
            eps += 1 / (INTEGRATION_STEPS * (eps_perp + eps_a * c * c))
        ee.append(1 / eps)
        # print(x, ee[-1])
    i = 0
    eps_error = 0.0
    while i < len(ee):
        eps_error += (ee[i] - eps_exp[i]) ** 2
        i = i + 1
    print('with eps_error = ' + str(math.sqrt(eps_error) / abs(eps_a)))
    global best_eps_error
    if math.sqrt(eps_error) / abs(eps_a) < best_eps_error:
        best_eps_error = math.sqrt(eps_error) / abs(eps_a)
    print('best eps_error = ' + str(best_eps_error))

    return math.sqrt(eps_error) / abs(eps_a)


result = []
for x in numpy.arange(K_n * (1 + K_as) * 0.6, K_n * (1 + K_as) * 1.5, K_n * (1 + K_as) * 0.2):
    for y in numpy.arange(K_n * (1 - K_as) * 0.6, K_n * (1 - K_as) * 1.5, K_n * (1 - K_as) * 0.2):
        print("=========================================================================\n (", x * Norm, y * Norm, " )")
        result.append([x, y, get_eps_error([(x + y) * 0.5, (x - y) / (x + y)])])
        print(result[-1])
        print(str(result[-1][0] * Norm) + ';' + str(result[-1][1] * Norm) + ';' + str(result[-1][2]))
print(result)
f = open(folder_name + '/map.csv', 'w')
f.write('K1;K3;epserror\n')
for x in result:
    f.write(str(x[0] * Norm) + ';' + str(x[1] * Norm) + ';' + str(x[2]) + '\n')
f.close()
f = open(folder_name + '/map_N.csv', 'w')
f.write('K_n;K_as;epserror\n')
for x in result:
    f.write(str((x[0] + x[1]) * Norm / 2) + ';' + str((x[0] - x[1]) / (x[0] + x[1])) + ';' + str(x[2]) + '\n')
f.close()
