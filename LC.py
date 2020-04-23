import numpy as np
import lcell as lc

from time import time
import os

GD={
    'size':[1,1,0.5*0.025],
    'K1':1.,
    'K2':.5,
    'K3':2.,
    'eps_par':3.85,
    'eps_perp':6.35,
    'chi':0.009894/1812,
    'U':0.1/300, #V*300 = 1 sgsV
    'N':40,
    'anc':500*40*np.array([1.,0.]),
    'directory':'/home/ivan/LC/GD/',
    'state_name':'LC',
    'data':'/home/ivan/LC/Gd 17-17_exp.dat'
}
ER={
    'size':[1,1,0.5*0.025],
    'K1':1.,
    'K2':.5,
    'K3':2.,
    'eps_par':3.52,
    'eps_perp':4.58,
    'chi':0.009894/1812,
    'U':0.000333564,
    'N':40,
    'anc':500*40*np.array([1.,0.1]),
    'directory':'/home/ivan/LC/ER/',
    'state_name':'LC',
    'data':'/home/ivan/LC/Er_17-17_exp.csv'
}
CB={
    'size':[1,1,2*0.0025],
    'K1':6.2e-7,
    'K2':3.9e-7,
    'K3':8.2e-7,
    'eps_par':13.14367+6.9,
    'eps_perp':6.9,
    'chi':0.000028427/249.36,
    'U':0.000333564,
    'N':40,
    'anc':500000*40*np.array([1.,0.]),
    'directory':'/home/ivan/LC/5CB/',
    'state_name':'LC',
    'data':'/home/ivan/LC/5CB_new_perp.dat'
}



LCD = lc.LcMinimiser(CB)
t = time()
#LCD.simple_minimize()
LCD.complex_minimize(node=4)
print('t = {}'.format(time()-t))
LCD.save()
maxang=LCD.get_maxangle_dependence()
eps=LCD.get_eps_dependence()

LCD.plot_maxangle(show=True)
LCD.plot_eps(show=True)
#LCD.plot_exp(show=True)
print(f'{LCD.diff() = }')

LC=lc.LcDependence(CB['K1'],CB['K2'],CB['K3'],load=True,directory=CB['directory'],state_name=CB['state_name'])
LC.plot()