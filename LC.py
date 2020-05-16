from time import time

import numpy as np

import lcell as lc

GD = {
    'size': [1, 1, 2 * 0.025],
    'K1': 1.0e-5,
    'K2': 0.5e-5,
    'K3': 2.0e-5,
    'K1_grid': [0.1, 1],
    'K2_grid': [0.1, 1],
    'K3_grid': [0.1, 1],
    'eps_par': 3.85,
    'eps_perp': 6.35,
    'chi': 0.009894 / 1812,
    'U': 0.1 / 300,  # V*300 = 1 sgsV
    'N': 40,
    'anc': 500000 * 40 * np.array([1., 0.]),
    'directory': '/home/ivan/LC/GD/',
    'state_name': 'GD',
    'data': '/home/ivan/LC/Gd 17-17_exp.dat'
}
ER = {
    'size': [1, 1, 0.5 * 0.025],
    'K1': 1.0e-5,
    'K2': 0.5e-5,
    'K3': 2.0e-5,
    'K1_grid': [0.1, 1],
    'K2_grid': [0.1, 1],
    'K3_grid': [0.1, 1],
    'eps_par': 3.52,
    'eps_perp': 4.58,
    'chi': 0.009894 / 1812,
    'U': 0.000333564,
    'N': 40,
    'anc': 500000 * 40 * np.array([1., 0.1]),
    'directory': '/home/ivan/LC/ER/',
    'state_name': 'ER',
    'data': '/home/ivan/LC/Er_17-17_exp.csv'
}
CB = {
    'size': [1, 1, 0.5 * 0.025],
    'K1': 6.2e-7,
    'K2': 3.9e-7,
    'K3': 8.2e-7,
    'K1_grid': [0.1, 0],
    'K2_grid': [0.1, 0],
    'K3_grid': [0.1, 0],
    'eps_par': 13.14367 + 6.9,
    'eps_perp': 6.9,
    'chi': 0.000028427 / 249.36,
    'U': 0.000333564 * 0,
    'N': 40,
    'anc': 100000 * 40 * np.array([1., 0.]),
    'directory': '/home/ivan/LC/5CB2/',
    'state_name': '5CB',
    'data': '/home/ivan/LC/5CB_new_perp.dat'
}
SM = {
    'size': [1, 1, 0.5 * 0.025],
    'K1': 6.2e-4,
    'K2': 3.9e-4,
    'K3': 8.2e-4,
    'K1_grid': [0.1, 0],
    'K2_grid': [0.1, 0],
    'K3_grid': [0.1, 0],
    'eps_par': 6.10156,
    'eps_perp': 5.44307,
    'chi': -0.009894 / 1812,
    'U': 0.000333564 * 0,
    'N': 40,
    'anc': 100000 * 40 * np.array([1., 1.]),
    'directory': '/home/ivan/LC/SM/',
    'state_name': 'SM',
    'data': '/home/ivan/LC/Sm 17-1.dat'
}
EU = {
    'size': [1, 1, 0.5 * 0.025],
    'K1': 6.2e-4,
    'K2': 3.9e-4,
    'K3': 8.2e-4,
    'K1_grid': [0.1, 0],
    'K2_grid': [0.1, 0],
    'K3_grid': [0.1, 0],
    'eps_par': 5.2,
    'eps_perp': 4.3,
    'chi': -0.003981 / 1812,
    'U': 0.000333564,
    'N': 200,
    'anc': 100000 * 40 * np.array([1., 1.]),
    'directory': '/home/ivan/LC/EU/',
    'state_name': 'SM',
    'data': '/home/ivan/LC/Eu 17-1.dat'
}
'''
LCD = lc.LcMinimiser(GD)
LCD.plot_only_practics(show=True,save='GD_p.pdf')
'''
LCD = lc.LcMinimiser(EU)
t = time()
LCD.minimize(nodes=6)  # nodes>1 for linux only
print('t = {}'.format(time() - t))

LCD.save()

LCD.plot(show=True)
print(f'{LCD.diff() = }')

LCD = lc.LcMinimiser(load={'directory': EU['directory'], 'state_name': EU['state_name']})
LCD.plot(show=True, save=EU['directory'] + 'EU.pdf')
LCD.rediff()
LC = LCD.get_point([0, 0, 0], mode='perp')
LC.plot(show=True)

print(f'{LCD.diff() = }')
print(f'{LCD.best_K() = }')
LCD.plot([LCD.best_K()[0][1]], show=True)
LCD.diff_plot()
# LCD.diff_plot_K13()
