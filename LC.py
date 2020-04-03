import numpy as np
import lcell as lc

from time import time
import os

H=10
hpoints = 11
Hlist=np.linspace(0,H,hpoints,endpoint=True)



GD={
    'size':[1,1,0.5*0.025],
    'K1':1.,
    'K2':.5,
    'K3':2.,
    'eps_par':3.85,
    'eps_perp':6.35,
    'chi':0.009894/1812,
    'U':0.000333564,
    'N':40,
    'anc':500*40*np.array([1.,0.1]),
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
    'size':[1,1,0.5*0.025],
    'K1':1.,
    'K2':.5,
    'K3':2.,
    'eps_par':13.14367+6.9,
    'eps_perp':6.9,
    'chi':0.000028427/249.36,
    'U':0.000333564,
    'N':40,
    'anc':500*40*np.array([1.,0.1]),
    'directory':'/home/ivan/LC/5CB/',
    'state_name':'LC',
    'data':'/home/ivan/LC/5CB_new_perp.dat'
}


t = time()

LCD = lc.LcMinimiser(GD)
LCD.simple_minimize()
LCD.save()
maxang=LCD.get_maxtheta_dependence()
eps=LCD.get_eps_dependence()

print('t = {}'.format(time()-t))

LCD.plot_maxangle(show=True)
LCD.plot_eps(show=True)
#LCD.plot(show=True)
LCD.plot_exp(show=True)
print(f'{LCD.diff() = }')



'''
filelist=[file for file in os.listdir(directory) if file[-4:]=='.pkl']
filelist=sorted(filelist,key=lambda x: float(x.split('_')[1][:-4]))
for file in filelist:
    obj = lc.LcCell_from_file(filename=directory+file)
    print(obj)
'''