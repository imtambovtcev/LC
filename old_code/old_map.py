import math
import os
import sys

import spline_old as sp

sys.path.append("numpy_path")
import shutil
import numpy
from scipy.optimize import minimize, minimize_scalar
from shutil import copyfile

# data_file = '5CB.dat'
# data_file = '5CB_new_perp.dat'
# data_file = 'Gd 17-17_exp.dat'
# data_file = 'Dy 17-17.datп '
# data_file = 'Er_17-17_exp.csv'
data_file = 'newer.csv'
# data_file = 'erid.dat'

g_type = 'perp'
# g_type='par'

# folder_name="5CB hard_map" #str(int(time.time()))
# folder_name="Gd hard_map"
folder_name = "Er hard_map"
# folder_name="erid"

best_eps_error = numpy.inf

global F
f = open(data_file, 'r')
lines = f.readlines()
H = []
eps_exp = []
for x in lines:
    H_n = float(x.split('\t')[0])
    try:
        if (g_type == 'perp'):
            e_n = float(x.split('\t')[2])  # perp
        elif (g_type == 'par'):
            e_n = float(x.split('\t')[1])  # par
        H.append(H_n)
        eps_exp.append(e_n)
    except:
        ()
#        print('blank line' )
f.close()

print(folder_name)
try:
    os.makedirs(folder_name)
except:
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)

copyfile(__file__, folder_name + '/code')
# eps_perp=6.35	#gd
# eps_par=6.35-2.5
# chi_tab=0.0018
# M=1812.0
# eps_perp=6.0	#dy
# eps_a=-2.0
# chi_tab=0.0014
# M=1812.0
# eps_perp = 6.9  # 5CB
# eps_par=20.04367
# chi_tab = 0.000028427
# M = 249.36

eps_perp = 4.58  # Er
eps_par = 3.52
eps_a = eps_par - eps_perp
chi_tab = 0.009894
M = 1812.0

rho = 1.0
# phi = 0.465507

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

# K_n=5.739990400137485 #5CB
# K_as= -0.3502994012969987
# K_n = 4392.4627500000015 #Gd
# K_as = -0.6408839779005527
# K_n = 4392.4627500000015 #Gd
# K_as = -0.749
# New K = [4200.40231625625, -0.7375232020077905]
# K_n = 4853.55 #test
# K_as = -0.138888889

K_n = 35849.393690624995  # Er
K_as = -0.9967373506785254


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
        F = chi * x * x / f_E
        #        print('H = ', x, 'f_H/f_E = ', F)
        x0 = []
        x0.append(math.pi * 0.499)
        x0.append(math.pi * 0.499)
        x0.append(math.pi * 0.499)
        x0.append(math.pi * 0.499)
        x0.append(math.pi * 0.499)
        x0.append(math.pi * 0.499)
        spx = []
        spy = []
        for i in range(len(x0) + 2):
            spx.append(float(i) / (len(x0) + 1))
        spy.append(BOUND[0])
        spy.extend(x0)
        spy.append(BOUND[1])
        spl = sp.functional(x=spx, f=spy, bound=BOUND, F=F, K=K_vec[0], K_as=K_vec[1], g_type=g_type)
        for k in range(3, MAX_SPLINE_POINTS):
            if len(x0) == 1:
                solution = minimize_scalar(spl.wp, bounds=(0.0, math.pi * 0.5), method='bounded')
            else:
                b = []
                for i in range(0, len(x0)): b.append((0.0, math.pi * 0.5))
                b = tuple(b)
                solution = minimize(spl.wp, x0, method='L-BFGS-B', bounds=b)
            if k < MAX_SPLINE_POINTS - 1:
                new_point = spl.get_new_point()
                if type(solution.x) == float or type(solution.x) == numpy.float64:
                    if new_point[0] == 1:
                        x0.append(spl.get_value(new_point[1], new_point[0]))
                        x0.append(solution.x)
                    else:
                        x0.append(solution.x)
                        x0.append(spl.get_value(new_point[1], new_point[0]))
                else:
                    x0 = solution.x.tolist()
                    x0.insert(new_point[0] - 1, spl.get_value(new_point[1], new_point[0]))
                spl.set_new_point(new_point[0], new_point[1])
            else:
                f = []
                f.append(BOUND[0])
                f.extend(solution.x.tolist())
                f.append(BOUND[1])
                spl.reset_spline(f)
                spl.graph('graph/' + str(x))
        eps = 0.0
        for i in range(0, INTEGRATION_STEPS):
            c = math.cos(spl.get_value(float(i) / INTEGRATION_STEPS))
            eps = eps + 1 / (INTEGRATION_STEPS * (eps_perp + eps_a * c * c))
        ee.append(1 / eps)
    i = 0
    eps_error = 0.0
    while i < len(ee):
        eps_error = eps_error + (ee[i] - eps_exp[i]) ** 2
        i = i + 1
    print('final eps_error = ', math.sqrt(eps_error) / (abs(eps_a) * len(ee)))
    global best_eps_error
    if (math.sqrt(eps_error) / (abs(eps_a) * len(ee))) < best_eps_error:
        best_eps_error = math.sqrt(eps_error) / (abs(eps_a) * len(ee))
    print('best  eps_error = ' + str(best_eps_error))

    return math.sqrt(eps_error) / (abs(eps_a) * len(ee))


result = []
for x in numpy.linspace(K_n * (1 + K_as) * 0.1, K_n * (1 + K_as) * 1.9, 7, endpoint=True):
    for y in numpy.linspace(K_n * (1 - K_as) * 0.95, K_n * (1 - K_as) * 1.05, 7, endpoint=True):
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
