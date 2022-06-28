#!/usr/bin/python
import logging
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import spline_old as sp
from scipy.optimize import minimize, minimize_scalar

logging.basicConfig(filename="log.log", level=logging.INFO, filemode="w")

for iterator in range(0, 1):
    # data_file = '5CB.dat'
    # data_file = '5CB_new_perp.dat'
    # data_file = 'gd.dat'
    # data_file = 'Dy 17-17.dat'
    # data_file = 'Gd 17-17_exp.dat'
    # data_file = 'test.dat'
    # data_file = 'Er_17-17_exp.csv'
    data_file = 'newer.csv'
    # data_file = 'zeros.dat'

    g_type = 'perp'
    g_type = 'par'

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

    try:
        os.makedirs('graph')
    except:
        shutil.rmtree('graph')
        os.makedirs('graph')

    best_eps_error = np.inf

    # eps_perp=6.35	#gd
    # eps_a=-2.5
    # chi_tab=0.0018
    # M=1812.0

    # eps_perp=6.0	#dy
    # eps_a=-2.0
    # chi_tab=0.0014
    # M=1812.0

    # eps_perp = 6.9  # 5CB
    # eps_a = 13.14367
    # chi_tab = 0.000028427
    # M = 249.36

    eps_perp = 4.58  # Er
    eps_par = 3.52
    eps_a = eps_par - eps_perp
    chi_tab = 0.009894
    M = 1812.0

    rho = 1.0

    d = 0.5 * 0.025
    U = 0.000333564
    chi = chi_tab * rho / M
    print('chi =', chi)
    E_0 = U / d
    f_E = abs(eps_a * E_0 * E_0 / (4 * math.pi))
    print('F_E =', f_E)
    Norm = f_E * d * d
    print('Norm =', Norm)
    theta_0 = math.acos(math.sqrt(
        (eps_exp[0] - eps_perp) / eps_a))  # Gd: theta_0 =  0.3554307805330981 #Er: theta_0 = 0.4769000076041882
    # theta_0 = 0.4769000076041882
    # theta_0=iterator*math.pi/18.0    ##########################################
    print('theta_0 = ', theta_0)
    BOUND = [theta_0, theta_0]

    logging.info('H =\n' + str(H))
    logging.info('eps exp =\n' + str(eps_exp))
    logging.info('eps_p =' + str(eps_perp))
    logging.info('eps_a =' + str(eps_a))
    logging.info('BOUND = ' + str(BOUND))
    print('H =\n' + str(H))
    print('eps exp =\n' + str(eps_exp))
    print('eps_p =' + str(eps_perp))
    print('eps_pp =' + str(eps_perp + eps_a))
    print('eps_a =' + str(eps_a))
    print('BOUND = ' + str(BOUND))

    INTEGRATION_STEPS = 1000
    MAX_SPLINE_POINTS = 10
    lam = 0.1
    MIN_POINTS = 100


    def get_eps_error(K_vec):
        logging.info('New K = ' + str(K_vec))
        print('New K = ' + str(K_vec))
        if K_vec[0] < 0.0:
            return np.inf
        if K_vec[1] < -1.0:
            return np.inf
        if K_vec[1] > 1.0:
            return np.inf
        ee = []
        for x in H:
            F = chi * x * x / f_E
            x0 = []
            if (g_type == 'perp'):
                x0.append(math.pi * 0.499)
                x0.append(math.pi * 0.499)
                x0.append(math.pi * 0.499)
                x0.append(math.pi * 0.499)
                x0.append(math.pi * 0.499)
            else:
                x0.append(0)
                x0.append(0)
                x0.append(0)
                x0.append(0)
                x0.append(0)
            spx = []
            spy = []
            for i in range(len(x0) + 2):
                spx.append(float(i) / (len(x0) + 1))
            spy.append(BOUND[0])
            spy.extend(x0)
            spy.append(BOUND[1])
            spl = sp.functional(x=spx, f=spy, bound=BOUND, F=F, K=K_vec[0], K_as=K_vec[1], type='perp')
            for k in range(5, MAX_SPLINE_POINTS):
                if len(x0) == 1:
                    solution = minimize_scalar(spl.wp, bounds=(0.0, math.pi * 0.5), method='bounded')
                #	print solution
                else:
                    b = []
                    for i in range(0, len(x0)): b.append((0.0, math.pi * 0.5))
                    b = tuple(b)
                    # brds=b#tuple(b)
                    # print x0, len(x0)
                    # print b, len(b)
                    solution = minimize(spl.wp, x0, method='L-BFGS-B', bounds=b)
                # if F>50.0:
                #	print solution
                #	raw_input()
                # print solution.fun
                # print solution.x
                if k < MAX_SPLINE_POINTS - 1:
                    new_point = spl.get_new_point()
                    # print solution.x,type(solution.x)
                    if type(solution.x) == float or type(solution.x) == np.float64:
                        x0 = []
                        if new_point[0] == 1:
                            x0.append(spl.get_value(new_point[1], new_point[0]))
                            x0.append(solution.x)
                        else:
                            x0.append(solution.x)
                            x0.append(spl.get_value(new_point[1], new_point[0]))
                    else:
                        x0 = solution.x.tolist()
                        x0.insert(new_point[0] - 1, spl.get_value(new_point[1], new_point[0]))
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
                eps = eps + 1 / (INTEGRATION_STEPS * (eps_perp + eps_a * c * c))
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


    # solution = basinhopping(get_eps_error, [6.186816, -0.138888889], niter=10, disp=True)
    # K_n = solution.x[0]
    # K_as = solution.x[1]

    # K=0.02115597	#Dy
    # K_as=-0.14912428

    # K_n=5.739990400137485 #5CB
    # K_as= -0.3502994012969987

    # K_n=6.186816	#5CB_id
    # K_as=-0.138888889

    # K_n = 4392.4627500000015 #Gd
    # K_as = -0.749

    K_n = 35849.393690624995  # Er
    K_as = -0.9967373506785254

    print('K_n = ' + str(K_n))
    print('K_as = ' + str(K_as))
    print('Final K = ', K_n * Norm)
    print('K_1 = ', K_n * Norm * (1.0 + K_as))
    print('K_3 = ', K_n * Norm * (1.0 - K_as))

    ee = []
    for x in H:
        F = chi * x * x / abs(f_E)
        print('H = ', x, 'f_H/f_E = ', F)
        #    if F<1.0:   F=1.0
        x0 = np.linspace(0., 1., 7, endpoint=True)
        x0 = x0[1:-1]
        if g_type == 'perp':
            spy = sp.ini(7, BOUND[0], np.pi / 2.)
        else:
            spy = sp.ini(7, BOUND[0], 0.)
        spx = []
        for i in range(len(x0) + 2):
            spx.append(float(i) / (len(x0) + 1))
        spl = sp.functional(x=spx, f=spy, bound=BOUND, F=F, K=K_n, K_as=K_as, g_type=g_type)
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
                if type(solution.x) == float or type(solution.x) == np.float64:
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
        # plt.plot(np.linspace(0., 1., len(f)), f, '.')
        # plt.show()
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
    print('final eps_error = ', math.sqrt(eps_error) / (len(ee) * abs(eps_a)))
    f = open('erteor' + str(iterator) + '.dat', 'w')
    f.write('H;eps\n')
    i = 0
    while i < len(H):
        f.write(str(H[i]) + ';' + str(ee[i]) + '\n')
        i = i + 1
    f.close()
    plt.plot(H, ee)
    plt.plot(H, eps_exp, '.')
    plt.show()
