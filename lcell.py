import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize
import os
from multiprocessing import Pool
import math


class Director:
    def __init__(self, tp=None, N=100, tp0=np.array([[0.], [0.]])):
        tp0 = np.array(tp0).reshape(2, 1)
        if tp is None:
            self.tp = np.array([np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - tp0[0]) + tp0[0],
                                np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - tp0[1]) + tp0[1]])
        else:
            self.tp = np.array(tp).reshape(2, -1)
        self.N = self.tp.shape[1]
        self.rediff()

    def rediff(self):
        self.dtp = self.tp[:,2:]-self.tp[:,:-2]
        self.dtp = np.concatenate((2*(self.tp[:,1]-self.tp[:,0]).reshape(2,1),
                                   self.dtp,
                                   2*(self.tp[:,-1]-self.tp[:,-2]).reshape(2,1)),axis=1)
        self.dtp*=0.5* self.N

    def newtp(self, tp):
        self.tp = np.array(tp).reshape(2, -1)
        self.rediff()

    def get_max_theta(self):
        return self.tp[0,:].max()

    def get_max_phi(self):
        return self.tp[1,:].max()

    def plot(self,title=None,show=False,save=None):
        fig, ax1 = plt.subplots()
        if title is not None:
            plt.title(title)
        ax1.set_xlabel('z')
        ax1.set_ylabel('ang')
        ax1.plot(np.linspace(0., 1., self.N), self.tp[0], 'r', label='theta')
        ax1.plot(np.linspace(0., 1., self.N), self.tp[1], 'b', label='phi')
        fig.tight_layout()
        plt.legend()
        ax2 = ax1.twinx()
        ax2.set_ylabel('grad')
        ax2.plot(np.linspace(0., 1., self.N), self.dtp[0], 'r--', label='dtheta')
        ax2.plot(np.linspace(0., 1., self.N), self.dtp[1], 'b--', label='dphi')
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()
        plt.close('all')


class LcCell(Director):

    def __init__(self, K1, K2, K3, size=np.array([1.,1.,1.]), eps_par=0., eps_perp=0., chi=0., E=0., H=0., anc=np.array([0.,0.]), state=None, N=100,
                 tp0=np.array([[0.], [0.]])):
        self.size = np.array(size)
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.K3 = float(K3)
        self.K1_n = float(K1)/float(K3)
        self.K2_n = float(K2)/float(K3)
        self.eps_par = float(eps_par)
        self.eps_perp = float(eps_perp)
        self.chi = float(chi)
        self.E=E
        self.H=H
        self.E_n = (self.eps_par-self.eps_perp)*(float(E)**2)/self.K3
        self.H_n = self.chi*(float(H)**2)/self.K3
        self.anc = np.array(anc)
        self.tp0 = np.array(tp0).reshape(2,)
        super().__init__(tp=state, N=N, tp0=self.tp0)

    def k_energy_density(self):
        return 0.5*(
                    self.K1_n*np.sin(self.tp[0])**2 *self.dtp[0]**2+
                    self.K2_n*np.sin(self.tp[1])**4 *self.dtp[1]**2+
                    (np.sin(self.tp[0])**2*np.cos(self.tp[0])**2 *self.dtp[1]**2+
                             np.cos(self.tp[0])**2 *self.dtp[0]**2)
                    )

    def H_energy_density(self):
        return -self.H_n * (np.sin(self.tp[0])*np.cos(self.tp[1]))**2

    def E_energy_density(self):
        return -self.E_n* np.cos(self.tp[0])**2

    def boundary_energy(self):
 #       return self.anc*(np.linalg.norm(self.tp[:, 0]-self.tp0)**2+np.linalg.norm(self.tp[:, -1]-self.tp0)**2)  # с phi
        return np.linalg.norm(self.anc * ((self.tp[:, 0] - self.tp0) ** 2 + (self.tp[:, -1] - self.tp0) ** 2))

    def energy(self):
        #print('E = ',self.k_energy_density().sum() + self.H_energy_density().sum() + self.boundary_energy(),'Ek = ',self.k_energy_density().sum(),'Ef = ', self.H_energy_density().sum(),'Eb = ',self.boundary_energy())
        return self.k_energy_density().sum() + self.H_energy_density().sum()+ self.E_energy_density().sum() + self.boundary_energy()
        #return ((self.K1*np.sin(self.tp[0])**2 + self.K3*np.cos(self.tp[0])**2)*(self.dtp[0]/self.N)**2 - self.H*np.cos(self.tp[0])**2).sum()-self.anc*(np.cos(self.tp[0,0])+np.cos(self.tp[0,-1]))

    def energy_density_plot(self,title=None,show=False,save=None):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('z')
        ax1.set_ylabel('Energy')
        ax1.plot(np.linspace(0., 1., self.N), self.k_energy_density(), 'r', label='k')
        ax1.plot(np.linspace(0., 1., self.N), self.H_energy_density(), 'b', label='H')
        plt.legend()
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def weak_restate(self, state):
        self.newtp(state)
        #print(self)
        return self.energy()

    def strong_restate(self, state):
        self.newtp(state)
        return self.energy()

    def minimize_state(self):
        minimum = scipy.optimize.minimize(self.weak_restate, self.tp, method='CG', options={'gtol': 1e0, 'disp': True})
        self.weak_restate(minimum.x)
        return self.get_epsilon()

    def get_epsilon(self):
        return self.N/((1./(self.eps_perp+(self.eps_par-self.eps_perp)*(np.cos(self.tp[0])**2))).sum())

    def save(self,filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def __repr__(self):
        self.plot(title='H = {:.5f} Energy = {:.5f}'.format(self.H,self.energy()),show=True)
        self.energy_density_plot('H = {:.5f}, Energy = {:.5f}'.format(self.H, self.energy()),show=True)
        return 'LC_cell\n' \
               'K1={}\n' \
               'K2={}\n' \
               'K3={}\n' \
               'Kn1={}\n' \
               'Kn2={}\n' \
               'eps_par={}\n' \
               'eps_perp={}\n' \
               'chi={}\n' \
               'E={}\n' \
               'H={}\n' \
               'Energy={}\n' \
               'E_K={}\n' \
               'E_H={}\n' \
               'E_b={}\n' \
               'Energy_n={}\n' \
               'E_K_n={}\n' \
               'E_H_n={}\n' \
               'E_b_n={}\n'.format(self.K1, self.K2, self.K3,self.K1_n, self.K2_n, self.eps_par, self.eps_perp, self.chi, self.E, self.H,
                                   self.energy(), self.k_energy_density().sum(), self.H_energy_density().sum(),
                                   self.boundary_energy(), self.energy(), self.k_energy_density().sum(), self.H_energy_density().sum(),
                                    self.boundary_energy())

def LcCell_from_file(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def call(lc):
    return lc.minimize_state()



class LcDependence():
    def __init__(self,Hlist,size, K1, K2, K3,directory='./',state_name='LC',state=None, eps_par=0., eps_perp=0., chi=0., E=0., anc=np.array([0.,0.]),  N=100,
                 tp0=np.array([[0.], [0.]])):
        self.Hlist=np.array(Hlist)
        self.K1=K1
        self.K2=K2
        self.K3=K3
        if state:
            assert len(Hlist)==len(state)
            self.states=[LcCell(size=size,K1=K1,K2=K2,K3=K3,state=state[i],eps_par=eps_par,eps_perp=eps_perp,chi=chi,E=E,H=Hlist[i],anc=anc,N=N,tp0=tp0) for i in range(len(Hlist))]
        else:
            self.states=[LcCell(size=size,K1=K1,K2=K2,K3=K3,eps_par=eps_par,eps_perp=eps_perp,chi=chi,E=E,H=h,anc=anc,N=N,tp0=tp0) for h in Hlist]
        self.eps=np.array([lc.get_epsilon() for lc in self.states])
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory=directory
        self.state_name=state_name

    def simple_minimize(self):
        self.eps=np.array([lc.minimize_state() for lc in self.states])

    def cm(self,x):
        return x.minimize_state()
    def complex_minimize(self,node=4): #multiprocessing. Linux only
        with Pool(node) as p:
            self.eps = np.array(p.map(self.cm,self.states))
            #print(self.eps)

    def get_eps_dependence(self):
        return self.eps

    def plot(self,title=None,show=False,save=None):
        for lc in self.states:
            lc.plot(title=title,show=show,save=save)

    def plot_eps(self,title=None,show=False,save=None):
        plt.plot(self.Hlist, self.eps)
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def plot_maxangle(self,title=None,show=False,save=None):
        plt.plot(self.Hlist, self.get_maxtheta_dependence())
        plt.xlabel('H')
        plt.ylabel(r'max angle')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def get_maxtheta_dependence(self):
        return np.array([lc.get_max_theta() for lc in self.states])
    def get_maxphi_dependence(self):
        return np.array([lc.get_max_phi() for lc in self.states])
    def save(self):
        for idx,lc in enumerate(self.states):
            lc.save(self.directory+self.state_name + '_{:.5f}_{:.5f}_{:.5f}_{:.5f}.pkl'.format(self.K1,self.K2,self.K3,self.Hlist[idx]))
            np.savez(self.directory + self.state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(self.K1,self.K2,self.K3), H=self.Hlist, eps=self.eps,K1=self.K1,K2=self.K2,K3=self.K3)

class LcMinimiser(LcDependence):
    def __init__(self,system):
        self.full_Hlist,self.exp_eps,tp0=self.load_file(system['data'],system['eps_par'],system['eps_perp'])
        if True:
            Hlist=self.full_Hlist[np.invert(np.isnan(self.exp_eps[0]))]
            exp_eps = self.exp_eps[1][np.invert(np.isnan(self.exp_eps[1]))]
        if 'directory' in system:
            directory=system['directory']
        else: directory='./'
        if 'state_name' in system: state_name=system['state_name']
        else: state_name='LC'
        if 'size' in system: size=np.array(system['size'])
        else: size=np.array([1.,1.,1.])
        if 'state' in system: state=system['state']
        else: state=None
        if 'N' in system: N = int(system['N'])
        else: N = 100
        super().__init__(Hlist=Hlist, K1=float(system['K1']), K2=float(system['K2']), K3=float(system['K3']),
                         eps_par=float(system['eps_par']), eps_perp=float(system['eps_perp']), chi=float(system['chi']),E=float(system['U'])/system['size'][2],
                         anc=system['anc'], tp0=tp0, N=N,
                         directory=directory, state_name=state_name,size=size,state=state)

    def load_file(self,filename,eps_par,eps_perp):
        f = open(filename, 'r')
        lines = f.readlines()
        H = []
        eps_exp_par = []
        eps_exp_perp = []
        for x in lines:
            try:
                x=x.split('\t')
                H.append(float(x[0]))
                try:
                    eps_exp_par.append(float(x[1]))
                except:
                    eps_exp_par.append(np.nan)
                try:
                    eps_exp_perp.append(float(x[2]))
                except:
                    eps_exp_perp.append(np.nan)
            except: ()
        f.close()
        return np.array(H),np.array([eps_exp_par,eps_exp_perp]),np.array([[math.acos(math.sqrt((eps_exp_perp[0] - eps_perp) / (eps_par-eps_perp)))],[0.0]])

    def plot_exp(self,title=None,show=False,save=None):
        plt.plot(self.full_Hlist, self.exp_eps[0],'bx',label=r'$\varepsilon_{\parallel}$')
        plt.plot(self.full_Hlist, self.exp_eps[1],'rx',label=r'$\varepsilon_{\perp}$')
        plt.plot(self.Hlist,self.get_eps_dependence(),'.')
        plt.legend()
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def diff(self):
        return np.linalg.norm(self.exp_eps[1][np.invert(np.isnan(self.exp_eps[1]))]-self.get_eps_dependence())