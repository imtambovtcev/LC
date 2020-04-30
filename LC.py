import numpy as np

import lcell as lc

GD = {
    'size': [1, 1, 0.5 * 0.025],
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
    'K1_grid': [0.1, 1],
    'K2_grid': [0.1, 1],
    'K3_grid': [0.1, 1],
    'eps_par': 13.14367 + 6.9,
    'eps_perp': 6.9,
    'chi': 0.000028427 / 249.36,
    'U': 0.000333564 * 0,
    'N': 40,
    'anc': 50000 * 40 * np.array([1., 0.]),
    'directory': '/home/ivan/LC/5CB/',
    'state_name': '5CB',
    'data': '/home/ivan/LC/5CB_new_perp.dat'
}
'''
LCD = lc.LcMinimiser(CB)
t = time()
LCD.minimize(nodes=6)  # nodes>1 for linux only
print('t = {}'.format(time() - t))

LCD.save()

# LCD.plot_maxangle(show=True)
LCD.plot(show=True)
print(f'{LCD.diff() = }')
'''
LCD = lc.LcMinimiser(load={'directory': CB['directory'], 'state_name': CB['state_name']})
# LCD.plot(show=True)
LCD.rediff()
print(f'{LCD.diff() = }')
print(f'{LCD.best_K() = }')
# LCD.diff_plot()
LCD.diff_plot_K13()
