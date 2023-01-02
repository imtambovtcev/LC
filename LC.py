import numpy as np
from time import time
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

CB5 = {
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
    'directory': '/home/ivan/LC/5CB/',
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
    'state_name': 'EU',
    'data': '/home/ivan/LC/Eu 17-1.dat'
}

# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/test.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/test.dat'),
#                           save_directory="/home/ivan/LC/test/",
#                           K1_list=1.e-5 * np.array([1.0, ]),
#                           K2_list=1.e-5 * np.array([1.0, ]),
#                           K3_list=1.e-5 * np.array([1.0]),
#                           N=40)

# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/5CB.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/5CB_new_perp.dat'),
#                           save_directory="/home/ivan/LC/5CB/",
#                           K1_list=6.2e-7 * np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
#                           K2_list=3.9e-7 * np.array([1.0, ]),
#                           K3_list=8.2e-7 * np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
#                           N=40)

# LCD = lc.Minimiser().load_from_directory(material=lc.Material().load('/home/ivan/LC/5CB.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/5CB_new_perp.dat'),
#                           save_directory="/home/ivan/LC/5CB/",)


# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/GD.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Gd 17-17_exp.dat'),
#                           save_directory="/home/ivan/LC/GD/",
#                           K1_list=0.00035 * np.linspace(0.5,1.5,11),
#                           K2_list=0.5e-4 * np.array([1.0,]),
#                           K3_list=0.0015 * np.linspace(0.5,1.5,11),
#                           N=40)

# LCD = lc.Minimiser().load_from_directory(material=lc.Material().load('/home/ivan/LC/GD.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Gd 17-17_exp.dat'),
#                           save_directory="/home/ivan/LC/GD/",)

# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/ER.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Er_17-17_exp.csv'),
#                           save_directory="/home/ivan/LC/ER/",
#                           K1_list=0.0001 * np.linspace(0.5,2,11),
#                           K2_list=0.5e-5 * np.array([1.0,]),
#                           K3_list=0.0009 * np.linspace(0.5,2,11),
#                           N=40)

# LCD = lc.Minimiser().load_from_directory(material=lc.Material().load('/home/ivan/LC/ER.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Er_17-17_exp.csv'),
#                           save_directory="/home/ivan/LC/ER/",)

# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/SM.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Sm 17-1.dat'),
#                           save_directory="/home/ivan/LC/SM/",
#                           K1_list=0.0005 * np.linspace(0.1,10,10),
#                           K2_list=0.0009 * np.linspace(0.1,10,3),
#                           K3_list=0.00025 * np.linspace(0.1,10,10),
#                           N=40)
#
LCD = lc.Minimiser().load_from_directory(material=lc.Material().load('/home/ivan/LC/SM.json'),
                          experiment=lc.Experiment('/home/ivan/LC/Sm 17-1.dat'),
                          save_directory="/home/ivan/LC/SM/",)

# LCD = lc.Minimiser().init(material=lc.Material().load('/home/ivan/LC/EU.json'),
#                           experiment=lc.Experiment('/home/ivan/LC/Eu 17-1.dat'),
#                           save_directory="/home/ivan/LC/EU/",
#                           K1_list=0.0005 * np.linspace(0.5,2,11),
#                           K2_list=0.000454 * np.linspace(0.5,2,11),
#                           K3_list=0.000126 * np.linspace(0.5,2,11),
#                           N=40)

# LCD = lc.Minimiser().load_from_directory(material=lc.Material().load('/home/ivan/LC/EU.json'),
#                                          experiment=lc.Experiment('/home/ivan/LC/Eu 17-1.dat'),
#                                          save_directory="/home/ivan/LC/EU/", )

# LCD.field.minimize(nodes=8)

print(f'{LCD = }')

# LCD.field.N=80

# LCD.field.minimize(nodes=8)
#
# print(f'{LCD = }')

# print(f'{LCD.field.nearest_perp([1e-05,4.7e-06,1.9e-05]) = }')

# print(f'{LCD.field.nearest_perp([9e-06,4.5e-06,1.8e-05]).lsr(LCD.experiment) = }')

print(f'{LCD.field.best(experiment=LCD.experiment)}')

# LCD.field.best_perp(experiment=experiment).plot_eps(show=True)

LCD.field.best_perp(experiment=LCD.experiment).plot(show=True,
                                                    save='/home/ivan/LC/' + LCD.material.state_name + '_perp.pdf')
LCD.field.best_par(experiment=LCD.experiment).plot(show=True,
                                                   save='/home/ivan/LC/' + LCD.material.state_name + '_par.pdf')

print(f'{LCD.field.best_perp(experiment=LCD.experiment).nearest(H=5000).energy_n() = }')

print(f'{LCD.field.best_perp(experiment=LCD.experiment).nearest(H=5000).energy() = } Erg')

# LCD.plot(show=True, points=[...,...,...])

LCD.plot_best(show=True, save='/home/ivan/LC/' + LCD.material.state_name + '.pdf')

LCD.save()

# LCD = lc.LcMinimiser(CB5)
# LCD.plot_only_practics(show=True)
# LCD.plot(show=True)
# # LCD.plot([LCD.best_K()[0][1]], show=True)
# LC = LCD.get_point([0, 0, 0], mode='perp')
# # LC.plot_maxangle(show=True)
# LC.plot(show=True)
# # LC.plot_one_by_one()41
#
# # LCD = lc.LcMinimiser(material)
# t = time()
# LCD.minimize(nodes=6)  # nodes>1 for linux only
# print('t = {}'.format(time() - t))
# #
# LCD.save()
#
# LCD.plot(show=True)
# print(f'{LCD.diff() = }')
#
#
#
# LCD = lc.LcMinimiser(load={'directory': material['directory'], 'state_name': material['state_name']})
# LCD.plot_smooth(show=True, save=material['directory'] + 'material_s.pdf')
# LCD.rediff()
# LC = LCD.get_point([0, 0, 0], mode='perp')
# # LC.plot(show=True)
#
# print(f'{LCD.diff() = }')
# print(f'{LCD.best_K() = }')
# LCD.plot([LCD.best_K()[0][1]], show=True)
# LCD.diff_plot()
# # LCD.diff_plot_K13()
