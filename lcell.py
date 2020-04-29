import math
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


class Director:
    def __init__(self, tp=None, N=100, tp0=np.array([[0.], [0.]])):
        tp0 = np.array(tp0).reshape(2, 1)
        if tp is None:
            print('initial')
            self.tp = np.array([np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - tp0[0]) + tp0[0],
                                np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - tp0[1]) + tp0[1]])
        else:
            self.tp = np.array(tp).reshape(2, -1)
        self.N = self.tp.shape[1]
        self.rediff()

    def rediff(self):
        self.dtp = self.tp[:, 2:] - self.tp[:, :-2]
        self.dtp = np.concatenate((2 * (self.tp[:, 1] - self.tp[:, 0]).reshape(2, 1),
                                   self.dtp,
                                   2 * (self.tp[:, -1] - self.tp[:, -2]).reshape(2, 1)), axis=1)
        self.dtp *= 0.5 * self.N

    def newtp(self, tp):
        self.tp = np.array(tp).reshape(2, -1)
        self.rediff()

    def get_max_theta(self):
        return self.tp[0, :].max()

    def get_max_phi(self):
        return self.tp[1, :].max()

    def plot(self, title=None, show=False, save=None):
        fig, ax1 = plt.subplots()
        if title:
            plt.title(title)
        ax1.set_xlabel('z')
        ax1.set_ylabel('ang')
        ax1.plot(np.linspace(0., 1., self.N), self.tp[0] * 180 / np.pi, 'r', label='theta')
        ax1.plot(np.linspace(0., 1., self.N), self.tp[1] * 180 / np.pi, 'b', label='phi')
        fig.tight_layout()
        plt.legend()
        ax2 = ax1.twinx()
        ax2.set_ylabel('grad')
        ax2.plot(np.linspace(0., 1., self.N), self.dtp[0], 'r--', label='dtheta')
        ax2.plot(np.linspace(0., 1., self.N), self.dtp[1], 'b--', label='dphi')
        plt.tight_layout()
        if save:
            plt.savefig(save)
        if show:
            plt.show()
        plt.close('all')


class LcCell(Director):
    def __init__(self, K1, K2, K3, mode='perp', size=np.array([1., 1., 1.]), eps_par=0., eps_perp=0., chi=0., E=0.,
                 H=0.,
                 anc=np.array([0., 0.]), state=None, N=100, tp0=np.array([[0.], [0.]])):
        self.size = np.array(size)
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.K3 = float(K3)
        self.mode = mode
        self.K1_n = float(K1) / float(K3)
        self.K2_n = float(K2) / float(K3)
        self.eps_par = float(eps_par)
        self.eps_perp = float(eps_perp)
        self.chi = float(chi)
        self.E = E
        self.H = H
        self.E_n = (self.eps_par - self.eps_perp) * (float(E) ** 2) / self.K3
        self.H_n = self.chi * (float(H) ** 2) / self.K3
        self.anc = np.array(anc)
        self.tp0 = np.array(tp0).reshape(2, )
        super().__init__(tp=state, N=N, tp0=self.tp0)

    def k_energy_density(self):
        return 0.5 / self.size[2] ** 2 * (
            self.K1_n * np.sin(self.tp[0]) ** 2 * self.dtp[0] ** 2 +
            self.K2_n * np.sin(self.tp[1]) ** 4 * self.dtp[1] ** 2 +
            (np.sin(self.tp[0]) ** 2 * np.cos(self.tp[0]) ** 2 * self.dtp[1] ** 2 +
             np.cos(self.tp[0]) ** 2 * self.dtp[0] ** 2)
        )

    def H_energy_density(self):
        if self.mode == 'perp':
            return -self.H_n * (np.sin(self.tp[0]) * np.cos(self.tp[1])) ** 2
        else:
            return -self.H_n * np.cos(self.tp[0]) ** 2

    def E_energy_density(self):
        return -self.E_n * np.cos(self.tp[0]) ** 2

    def boundary_energy(self):
        #       return self.anc*(np.linalg.norm(self.tp[:, 0]-self.tp0)**2+np.linalg.norm(self.tp[:, -1]-self.tp0)**2)  # с phi
        return np.linalg.norm(self.anc * ((self.tp[:, 0] - self.tp0) ** 2 + (self.tp[:, -1] - self.tp0) ** 2))

    def energy(self):
        # print('E = ',self.k_energy_density().sum() + self.H_energy_density().sum() + self.boundary_energy(),
        # 'Ek = ',self.k_energy_density().sum(),'Ef = ', self.H_energy_density().sum(),'Eb = ',self.boundary_energy())
        return self.k_energy_density().sum() + self.H_energy_density().sum() + self.E_energy_density().sum() + self.boundary_energy()
        # return ((self.K1*np.sin(self.tp[0])**2 + self.K3*np.cos(self.tp[0])**2)*(self.dtp[0]/self.N)**2
        # - self.H*np.cos(self.tp[0])**2).sum()-self.anc*(np.cos(self.tp[0,0])+np.cos(self.tp[0,-1]))

    def energy_density_plot(self, title=None, show=False, save=None):
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
        # print(self)
        return self.energy()

    def strong_restate(self, state):
        self.newtp(state)
        return self.energy()

    def minimize_state(self):
        minimum = scipy.optimize.minimize(self.weak_restate, self.tp, method='CG', options={'gtol': 1e0, 'disp': True})
        self.weak_restate(minimum.x)
        return self

    def get_epsilon(self):
        return self.N / ((1. / (self.eps_perp + (self.eps_par - self.eps_perp) * (np.cos(self.tp[0]) ** 2))).sum())

    def save(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        self.plot(title='H = {:.10f} Energy = {:.10f}'.format(self.H, self.energy()), show=True)
        self.energy_density_plot('H = {:.10f}, Energy = {:.10f}'.format(self.H, self.energy()), show=True)
        return 'LC_cell\n' \
               'K1={}\n' \
               'K2={}\n' \
               'K3={}\n' \
               'mode={}\n' \
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
               'E_b_n={}\n'.format(self.K1, self.K2, self.K3, self.mode, self.K1_n, self.K2_n, self.eps_par,
                                   self.eps_perp,
                                   self.chi, self.E, self.H,
                                   self.energy(), self.k_energy_density().sum(), self.H_energy_density().sum(),
                                   self.boundary_energy(), self.energy(), self.k_energy_density().sum(),
                                   self.H_energy_density().sum(),
                                   self.boundary_energy())


def LcCell_from_file(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


class LcDependence():
    def __init__(self, K1, K2, K3, mode='perp', load=None, Hlist=None, size=None, state=None,
                 eps_par=0., eps_perp=0., chi=0., E=0., anc=np.array([0., 0.]), N=100,
                 tp0=np.array([[0.], [0.]])):

        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.mode = mode

        load_status = False
        if load:
            try:
                if self.load(directory=load['directory'], state_name=load['state_name']) == 1:
                    load_status = True
                else:
                    print('uncomplete load ' + load['directory'] + load['state_name'] + '_'
                          + self.mode + '_{:.10f}_{:.10f}_{:.10f}.npz'.format(self.K1, self.K2, self.K3))
                    print('starting from initialisation')
            except:
                print('load failure')
        if not load_status:
            self.Hlist = np.array(Hlist)
            if state:
                assert len(self.Hlist) == len(state)
                self.states = [LcCell(size=size, K1=self.K1, K2=self.K2, K3=self.K3, mode=self.mode,
                                      state=state[i], eps_par=eps_par, eps_perp=eps_perp, chi=chi, E=E,
                                      H=Hlist[i], anc=anc, N=N, tp0=tp0) for i in range(len(Hlist))]
            else:
                self.states = [LcCell(size=size, K1=K1, K2=K2, K3=K3, mode=self.mode,
                                      eps_par=eps_par, eps_perp=eps_perp, chi=chi, E=E,
                                      H=h, anc=anc, N=N, tp0=tp0) for h in Hlist]
        self.eps = np.array([lc.get_epsilon() for lc in self.states])

    def simple_minimize(self):
        self.states = [lc.minimize_state() for lc in self.states]
        self.eps = np.array([lc.get_epsilon() for lc in self.states])
        return self

    def cm(self, x):
        return x.minimize_state()

    def complex_minimize(self, node=4):  # multiprocessing. Linux only
        with Pool(node) as p:
            self.states = p.map(self.cm, self.states)
        self.eps = np.array([lc.get_epsilon() for lc in self.states])
        return self

    def get_eps_dependence(self):
        return self.eps

    def get_mode(self):
        return self.mode

    def plot(self, title='H = ', show=False, save=None):
        [print(lc) for lc in self.states]

    '''        for idx,lc in enumerate(self.states):
            if title == 'H = ':
                title='H = {:.10f}'.format(self.Hlist[idx])
            lc.plot(title=title,show=show,save=save)
    '''

    def plot_eps(self, title=None, show=False, save=None):
        plt.plot(self.Hlist, self.eps)
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def plot_maxangle(self, title=None, show=False, save=None):
        plt.plot(self.Hlist, self.get_maxangle_dependence()[:, 0], 'r', label=r'$\theta$')
        plt.plot(self.Hlist, self.get_maxangle_dependence()[:, 1], 'b', label=r'$\varphi$')
        plt.xlabel('H')
        plt.ylabel(r'max angle')
        plt.legend()
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def get_maxangle_dependence(self):
        return np.array([[lc.get_max_theta(), lc.get_max_phi()] for lc in self.states])

    def save(self, directory, state_name):
        for idx, lc in enumerate(self.states):
            lc.save(directory + state_name + '_' + self.mode +
                    '_{:.10f}_{:.10f}_{:.10f}_{:.10f}.pkl'.format(self.K1, self.K2, self.K3, self.Hlist[idx]))
        np.savez(
            directory + state_name + '_' + self.mode + '_{:.10f}_{:.10f}_{:.10f}.npz'.format(self.K1, self.K2, self.K3),
            H=self.Hlist, eps=self.eps, K1=self.K1, K2=self.K2, K3=self.K3, mode=self.mode)

    def load(self, directory, state_name):
        file = np.load(
            directory + state_name + '_' + self.mode + '_{:.10f}_{:.10f}_{:.10f}.npz'.format(self.K1, self.K2, self.K3))
        self.Hlist = file['H']
        self.eps = file['eps']
        self.K1 = file['K1']
        self.K2 = file['K2']
        self.K3 = file['K3']

        try:
            self.states = [LcCell_from_file(directory + state_name + '_' + self.mode +
                                            '_{:.10f}_{:.10f}_{:.10f}_{:.10f}.pkl'.format(self.K1, self.K2, self.K3, h))
                           for h in self.Hlist]
        except:
            return -1
        return 1


class LcMinimiser():
    def __init__(self, system=None, load=None):
        if system:
            self.exp_eps_perp, self.exp_eps_par, perp_tp0, par_tp0 = self.load_file(system['data'], system['eps_par'],
                                                                                    system['eps_perp'])
            if 'directory' in system:
                self.directory = system['directory']
            else:
                self.directory = './'
            if 'state_name' in system:
                self.state_name = system['state_name']
            else:
                self.state_name = 'LC'
            if 'size' in system:
                size = np.array(system['size'])
            else:
                size = np.array([1., 1., 1.])
            if 'state' in system:
                state = system['state']
            else:
                state = None
            if 'N' in system:
                N = int(system['N'])
            else:
                N = 100

            self.Kv = np.zeros(
                [2 * system['K1_grid'][1] + 1, 2 * system['K2_grid'][1] + 1, 2 * system['K3_grid'][1] + 1, 3])
            self.shape = self.Kv.shape[:3]
            self.perp_points = np.full(self.Kv.shape[:3], np.nan, dtype=object)
            self.perp_eps_diff = np.zeros(self.Kv.shape[:3])
            self.par_points = np.full(self.Kv.shape[:3], np.nan, dtype=object)
            self.par_eps_diff = np.zeros(self.Kv.shape[:3])
            for ct1, i in enumerate(range(-system['K1_grid'][1], system['K1_grid'][1] + 1)):
                for ct2, j in enumerate(range(-system['K2_grid'][1], system['K2_grid'][1] + 1)):
                    for ct3, k in enumerate(range(-system['K3_grid'][1], system['K3_grid'][1] + 1)):
                        self.Kv[ct1, ct2, ct3, :] = np.array([(1. + i * system['K1_grid'][0]) * system['K1'],
                                                              (1. + j * system['K2_grid'][0]) * system['K2'],
                                                              (1. + k * system['K3_grid'][0]) * system['K3']])
                        self.perp_points[ct1, ct2, ct3] = LcDependence(Hlist=self.exp_eps_perp[:, 0],
                                                                       K1=self.Kv[ct1, ct2, ct3, 0],
                                                                       K2=self.Kv[ct1, ct2, ct3, 1],
                                                                       K3=self.Kv[ct1, ct2, ct3, 2],
                                                                       mode='perp',
                                                                       eps_par=float(system['eps_par']),
                                                                       eps_perp=float(system['eps_perp']),
                                                                       chi=float(system['chi']),
                                                                       E=float(system['U']) / system['size'][2],
                                                                       anc=system['anc'], tp0=perp_tp0, N=N,
                                                                       size=size, state=state)
                        self.perp_eps_diff[ct1, ct2, ct3] = self.diff(self.perp_points[ct1, ct2, ct3])
                        self.par_points[ct1, ct2, ct3] = LcDependence(Hlist=self.exp_eps_par[:, 0],
                                                                      K1=self.Kv[ct1, ct2, ct3, 0],
                                                                      K2=self.Kv[ct1, ct2, ct3, 1],
                                                                      K3=self.Kv[ct1, ct2, ct3, 2],
                                                                      mode='par',
                                                                      eps_par=float(system['eps_par']),
                                                                      eps_perp=float(system['eps_perp']),
                                                                      chi=float(system['chi']),
                                                                      E=float(system['U']) / system['size'][2],
                                                                      anc=system['anc'], tp0=par_tp0, N=N,
                                                                      size=size, state=state)
                        self.par_eps_diff[ct1, ct2, ct3] = self.diff(self.par_points[ct1, ct2, ct3])
            print(f'{self.Kv = }')
            print(self.perp_points.shape)
            print(self.par_points.shape)
        if load:
            self.load(load)

    def load_file(self, filename, eps_par, eps_perp):
        f = open(filename, 'r')
        lines = f.readlines()
        eps_exp_par = []
        eps_exp_perp = []
        for x in lines:
            try:
                x = x.split('\t')
                try:
                    eps_exp_par.append([float(x[0]), float(x[1])])
                except:
                    eps_exp_par.append([np.nan, np.nan])
                try:
                    eps_exp_perp.append([float(x[0]), float(x[2])])
                except:
                    eps_exp_perp.append([np.nan, np.nan])
            except:
                print('broken file')
        f.close()
        eps_exp_perp = np.array(eps_exp_perp)
        eps_exp_par = np.array(eps_exp_par)
        eps_exp_perp = eps_exp_perp[np.invert(np.isnan(eps_exp_perp[:, 0]))]
        eps_exp_par = eps_exp_par[np.invert(np.isnan(eps_exp_par[:, 0]))]
        theta0_perp = math.acos(math.sqrt((eps_exp_perp[0, 1] - eps_perp) / (eps_par - eps_perp)))
        phi0_perp = 0 * np.pi / 2
        theta0_par = math.acos(math.sqrt((eps_exp_par[0, 1] - eps_perp) / (eps_par - eps_perp)))
        phi0_par = 0 * np.pi / 2
        print(f'{theta0_perp*180/np.pi = }\t{phi0_perp*180/np.pi = }')
        print(f'{theta0_par*180/np.pi = }\t{phi0_par*180/np.pi = }')
        return eps_exp_perp, eps_exp_par, np.array([[theta0_perp], [phi0_perp]]), np.array([[theta0_par], [phi0_par]])

    def minimize(self, nodes=1):
        assert isinstance(nodes, int) and nodes > 0
        self.perp_points = self.perp_points.reshape(-1)
        if nodes == 1:
            for i, p in enumerate(self.perp_points):
                self.perp_points[i] = p.simple_minimize()
                print(f'{i = }')
        else:
            for i, p in enumerate(self.perp_points):
                self.perp_points[i] = p.complex_minimize(node=nodes)
                print(f'{i = }')
        self.perp_points = self.perp_points.reshape(self.shape)
        shape = self.par_points.shape
        self.par_points = self.par_points.reshape(-1)
        if nodes == 1:
            for i, p in enumerate(self.perp_points):
                self.par_points[i] = p.simple_minimize()
                print(f'{i = }')
        else:
            for i, p in enumerate(self.par_points):
                self.par_points[i] = p.complex_minimize(node=nodes)
                print(f'{i = }')
        self.par_points = self.par_points.reshape(shape)
        return self

    def save(self):
        self.perp_points = self.perp_points.reshape(-1)
        for p in self.perp_points:
            p.save(directory=self.directory, state_name=self.state_name)
        self.perp_points = self.perp_points.reshape(self.shape)
        self.par_points = self.par_points.reshape(-1)
        for p in self.par_points:
            p.save(directory=self.directory, state_name=self.state_name)
        self.par_points = self.par_points.reshape(self.shape)
        np.savez(self.directory + self.state_name + '.npz', Kv=self.Kv,
                 perp_eps_diff=self.perp_eps_diff, par_eps_diff=self.par_eps_diff,
                 exp_eps_perp=self.exp_eps_perp, exp_eps_par=self.exp_eps_par)

    def load(self, l):
        self.directory = l['directory']
        self.state_name = l['state_name']
        file = np.load(self.directory + self.state_name + '.npz')
        self.Kv = file['Kv']
        self.exp_eps_perp = file['exp_eps_perp']
        self.exp_eps_par = file['exp_eps_par']
        self.perp_eps_diff = file['perp_eps_diff']
        self.par_eps_diff = file['par_eps_diff']
        self.perp_points = np.full(self.Kv.shape[:3], np.nan, dtype=object).reshape(-1)
        self.par_points = np.full(self.Kv.shape[:3], np.nan, dtype=object).reshape(-1)
        self.shape = self.Kv.shape[:3]
        self.Kv = self.Kv.reshape([-1, 3])
        for ct, K in enumerate(self.Kv):
            self.perp_points[ct] = LcDependence(*K, 'perp',
                                                load={'directory': self.directory, 'state_name': self.state_name})
            self.par_points[ct] = LcDependence(*K, 'par',
                                               load={'directory': self.directory, 'state_name': self.state_name})
        self.Kv = self.Kv.reshape(list(self.shape) + [3])
        self.perp_points = self.perp_points.reshape(self.shape)
        self.par_points = self.par_points.reshape(self.shape)

    def diff(self, lc='all'):
        if isinstance(lc, LcDependence):
            if lc.get_mode() == 'perp':
                return np.linalg.norm(self.exp_eps_perp[:, 1] - lc.get_eps_dependence())
            else:
                return np.linalg.norm(self.exp_eps_par[:, 1] - lc.get_eps_dependence())

        elif isinstance(lc, np.ndarray) or isinstance(lc, list):
            lc_perp_arr = self.perp_points[lc].reshape(-1)
            for ct, lc0 in enumerate(lc_perp_arr):
                lc_perp_arr[ct] = self.diff(lc0)
            lc_par_arr = self.par_points[lc].reshape(-1)
            for ct, lc0 in enumerate(lc_perp_arr):
                lc_par_arr[ct] = self.diff(lc0)
            return np.array([lc_par_arr, lc_perp_arr], dtype='float64')

        elif lc == 'all':
            return self.diff(np.full(self.perp_points.shape, True))

    def plot(self, title=None, show=False, save=None):
        plt.plot(self.exp_eps_par[:, 0], self.exp_eps_par[:, 1], 'bx', label=r'$\varepsilon_{\parallel}$')
        plt.plot(self.exp_eps_perp[:, 0], self.exp_eps_perp[:, 1], 'rx', label=r'$\varepsilon_{\perp}$')

        self.perp_points = self.perp_points.reshape(-1)
        for p in self.perp_points:
            plt.plot(self.exp_eps_perp[:, 0], p.get_eps_dependence())
        self.perp_points = self.perp_points.reshape(self.shape)

        self.par_points = self.par_points.reshape(-1)
        for p in self.par_points:
            plt.plot(self.exp_eps_par[:, 0], p.get_eps_dependence())
        self.par_points = self.par_points.reshape(self.shape)

        plt.legend()
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')


'''
    def plot_maxangle(self,title=None,show=False,save=None):
        shape = self.perp_points.shape
        self.perp_points = self.perp_points.reshape(-1)
        for p in self.perp_points:
            p.plot_maxangle()
        self.perp_points = self.perp_points.reshape(shape)
'''
