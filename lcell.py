import copy
import json
import math
import os
import pickle
from multiprocessing import Pool
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from ast import literal_eval


# CGS SU is used
# [K1,K2,K3]=dyn
# [l]=cm
# [H]=Oe
# [delta chi] = ?
# [U]=cgsV

# par = 0
# perp = 1

# MAP:
#
# Director-(inheritance)->Cell-(list)>Dependence-(list)->Field
#                                                        |
# material_load                                          |
#     |                                                  |
# Material-----------> Minimiser <----------------------
#                           ^
#                           |
# Experement----------------/

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Experiment:
    def __init__(self, filename=None):
        self.eps_par = np.empty(shape=(0, 0), dtype=float)
        self.eps_perp = np.empty(shape=(0, 0), dtype=float)
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        try:
            if isinstance(filename, Experiment):
                self.eps_perp = filename.eps_perp
                self.eps_par = filename.eps_par
            elif isinstance(filename, str):
                f = open(filename, 'r')
                lines = f.readlines()
                eps_par = []
                eps_perp = []
                for x in lines:
                    try:
                        x = x.split('\t')
                        try:
                            eps_par.append([float(x[0]), float(x[1])])
                        except:
                            eps_par.append([np.nan, np.nan])
                        try:
                            eps_perp.append([float(x[0]), float(x[2])])
                        except:
                            eps_perp.append([np.nan, np.nan])
                    except:
                        f.close()
                        raise ImportError(f'File {filename} is broken')
                f.close()
                eps_perp = np.array(eps_perp)
                eps_par = np.array(eps_par)
                self.eps_perp = eps_perp[np.invert(np.isnan(eps_perp[:, 0]))]
                self.eps_par = eps_par[np.invert(np.isnan(eps_par[:, 0]))]
            else:
                raise ImportError(f'load format is unknown {filename = }')
        except ImportError as error:
            # print(error)
            raise

    def __repr__(self):
        return f'\n\teps_par: np.array {self.eps_par.shape}\t eps_perp: np.array {self.eps_perp.shape}'


class Material:
    def __init__(self, system=None):
        self.eps_par = 1.
        self.eps_perp = 1.
        self.state_name = 'LC'
        self.size = np.array([1., 1., 1.])
        self.chi = 0.
        self.U = 0.
        self.anc_theta = 0.
        self.anc_phi = 0.
        self.theta0_par = 0.
        self.theta0_perp = 0.
        self.phi0_par = 0.
        self.phi0_perp = 0.
        if system is not None:
            self.load(system)

    def load(self, system):
        try:
            if isinstance(system, Material):
                self.eps_par = system.eps_par
                self.eps_perp = system.eps_perp
                self.state_name = system.state_name
                self.size = system.size
                self.chi = system.chi
                self.U = system.U
                self.anc_theta = system.anc_theta
                self.anc_phi = system.anc_phi
                self.theta0_par = system.theta0_par
                self.theta0_perp = system.theta0_perp
                self.phi0_par = system.phi0_par
                self.phi0_perp = system.phi0_perp
                return

            elif isinstance(system, list) or isinstance(system, str):
                if isinstance(system, list):
                    _system = copy.copy(system)
                else:
                    if system[-5:] == '.json':
                        if os.path.isfile(system):
                            _system = material_load(system)
                        else:
                            raise ImportError(f'No such file: {system = }')
                    else:
                        raise ImportError(f'Not a .json file: {system = }')

                if 'eps_par' in _system:
                    self.eps_par = _system['eps_par']
                else:
                    self.eps_par = 1.
                    warnings.warn(f'No eps_par info! Set to default {self.eps_par = }')

                if 'eps_perp' in _system:
                    self.eps_perp = _system['eps_perp']
                else:
                    self.eps_perp = 1.
                    warnings.warn(f'No eps_perp info! Set to default {self.eps_perp = }')

                if 'state_name' in _system:
                    self.state_name = _system['state_name']
                else:
                    self.state_name = 'LC'
                    warnings.warn(f'No state name info! Set to default {self.state_name = }')
                if 'size' in _system:
                    self.size = np.array(_system['size'])
                else:
                    self.size = np.array([1., 1., 1.])
                    warnings.warn(f'No size info! Set to default {self.size = }')
                if 'chi' in _system:
                    self.chi = float(_system['chi'])
                else:
                    self.chi = 0.
                    warnings.warn(f'No chi info! Set to default {self.chi = }')
                if 'U' in _system:
                    self.U = float(_system['U'])
                else:
                    self.U = 0.
                    warnings.warn(f'No U info! Set to default {self.U = }')
                if 'anc_theta' in _system:
                    self.anc_theta = float(_system['anc_theta'])
                else:
                    self.anc_theta = 0.
                    warnings.warn(f'No anc_theta info! Set to default {self.anc_theta = }')

                if 'anc_phi' in _system:
                    self.anc_phi = float(_system['anc_phi'])
                else:
                    self.anc_phi = 0.
                    warnings.warn(f'No anc_phi info! Set to default {self.anc_phi = }')

                if 'theta0_perp' in _system:
                    self.theta0_perp = np.array(_system['theta0_perp'])
                else:
                    self.theta0_perp = 0.
                    warnings.warn(f'No theta0_perp info! Set to default {self.theta0_perp = }')

                if 'theta0_par' in _system:
                    self.theta0_par = np.array(_system['theta0_par'])
                else:
                    self.theta0_par = 0.
                    warnings.warn(f'No theta0_par info! Set to default {self.theta0_par = }')

                if 'phi0_perp' in _system:
                    self.phi0_perp = np.array(_system['phi0_perp'])
                else:
                    self.phi0_perp = 0.
                    warnings.warn(f'No phi0_perp info! Set to default {self.phi0_perp = }')

                if 'phi0_par' in _system:
                    self.phi0_par = np.array(_system['phi0_par'])
                else:
                    self.phi0_par = 0.
                    warnings.warn(f'No phi0_par info! Set to default {self.phi0_par = }')

            else:
                raise ImportError(f'load format is unknown {system = }')
        except ImportError as error:
            # print(error)
            raise

    def __repr__(self):
        return f'\n\tstate_name:{self.state_name}\n' \
               f'\tsize:{self.size}\n' \
               f'\tchi: {self.chi}\tU: {self.U}\n' \
               f'\teps_perp: {self.eps_perp}\teps_par: {self.eps_par}\n' \
               f'\tanc_theta: {self.anc_theta}\tanc_phi: {self.anc_phi}\n' \
               f'\ttheta0_perp: {self.theta0_perp}\ttheta0_par: {self.theta0_par}' \
               f'\tphi0_perp: {self.phi0_perp}\tphi0_par: {self.phi0_par}'


class Director:
    def __init__(self, theta=None, phi=None, N=40, theta0=0., phi0=0., mode='perp'):
        if theta is None:
            match mode:
                case 'perp':
                    self.theta = np.array([np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - theta0) + theta0])
                case 'par':
                    self.theta = np.array([np.sin(np.linspace(0., np.pi, N)) * (0. * np.pi - theta0) + theta0])
        else:
            self.theta = np.array(theta)

        self.theta = self.theta.reshape(-1)

        if phi is None:
            match mode:
                case 'perp':
                    self.phi = np.array([np.sin(np.linspace(0., np.pi, N)) * (0.5 * np.pi - phi0) + phi0])
                case 'par':
                    self.phi = np.array([np.sin(np.linspace(0., np.pi, N)) * (0. * np.pi - phi0) + phi0])
        else:
            self.phi = np.array(phi)

        self.phi = self.phi.reshape(-1)

        self.dtheta = np.zeros(N)
        self.dphi = np.zeros(N)
        self.rediff()

    @property
    def N(self):
        return len(self.theta)

    def rediff(self): # dtheta = (d theta/ d z) * l_z; [dtheta] = 1
        self.dtheta[0] = 2 * (self.theta[1] - self.theta[0])
        self.dtheta[1:-1] = self.theta[2:] - self.theta[:-2]
        self.dtheta[-1] = 2 * (self.theta[-1] - self.theta[-2])

        self.dphi[0] = 2 * (self.phi[1] - self.phi[0])
        self.dphi[1:-1] = self.phi[2:] - self.phi[:-2]
        self.dphi[-1] = 2 * (self.phi[-1] - self.phi[-2])

        self.dtheta *= 0.5 * self.N
        self.dphi *= 0.5 * self.N

    def plot(self, title=None, show=False, save=None):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        self.plot_ax(ax1, ax2)

        if title:
            plt.title(title)
        ax1.set_xlabel('z')
        ax1.set_ylabel('ang')
        ax2.set_ylabel('grad')
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save)
        if show:
            plt.show()
        plt.close('all')

    def plot_ax(self, ax1, ax2, color_theta='r', color_phi='b'):
        ax1.plot(np.linspace(0., 1., self.N), self.theta * 180 / np.pi, c=color_theta, label='theta')
        ax1.plot(np.linspace(0., 1., self.N), self.phi * 180 / np.pi, c=color_phi, label='phi')
        if ax2 is not None:
            ax2.plot(np.linspace(0., 1., self.N), self.dtheta, '--', c=color_theta, label='dtheta')
            ax2.plot(np.linspace(0., 1., self.N), self.dphi, '--', c=color_phi, label='dphi')


class Cell(Director):
    def __init__(self):
        self.mode = 'perp'
        self._size = np.array([1, 1, 1])
        self._K1 = 0.
        self._K2 = 0.
        self._K3 = 0.

        self.eps_perp = 0.
        self.eps_par = 0.
        self.chi = 0.

        self.H = 0.
        self.E = 0.

        self.anc_theta = 0.
        self.anc_phi = 0.

        self._K1_n = 0.
        self._K2_n = 0.
        self._K3_n = 0.
        self._E_n = 0.
        self._H_n = 0.
        self._anc_theta_n = 0.
        self._anc_phi_n = 0.

        self.theta0 = 0.
        self.phi0 = 0.

        self.norm = 1.

        super().__init__()

    def init(self, material: Material, K1=0., K2=0., K3=0., mode='perp', H=0., theta=None, phi=None, N=40):
        self.mode = mode
        self._size = np.array(material.size)
        self._K1 = float(K1)
        self._K2 = float(K2)
        self._K3 = float(K3)

        self.eps_perp = material.eps_perp
        self.eps_par = material.eps_par
        self.chi = material.chi

        self.H = float(H)
        self.E = float(material.U / material.size[2])

        self.anc_theta = material.anc_theta
        self.anc_phi = material.anc_phi

        self.norm = self.K3 / (self.size[2] ** 2)
        # print(f'{self.norm = }')

        match mode:
            case 'perp':
                self.theta0 = material.theta0_perp
                self.phi0 = material.phi0_perp
            case 'par':
                self.theta0 = material.theta0_par
                self.phi0 = material.phi0_par

        super().__init__(theta=theta, phi=phi, N=N, theta0=self.theta0, phi0=self.phi0, mode=self.mode)

        # print('k energy mode:')
        if self.anc_phi == -1:
            # print('theta')
            self.k_energy_density = self._k_energy_density_theta
        else:
            # print('theta_phi')
            self.k_energy_density = self._k_energy_density_theta_phi

        # print('H energy mode:')
        if self.mode == 'par':
            # print('par')
            self.H_energy_density = self._H_energy_density_par
        else:
            if self.anc_phi == -1:
                # print('perp theta')
                self.H_energy_density = self._H_energy_density_perp_theta
            else:
                # print('perp theta_phi')
                self.H_energy_density = self._H_energy_density_perp_theta_phi

        # print('Boundary energy mode:')
        if self.anc_theta == float('inf'):
            # print('rigid')
            self.boundary_energy = self._boundary_energy_rigid
        else:
            if self.anc_phi == -1:
                # print('theta')
                self.boundary_energy = self._boundary_energy_theta
            else:
                # print('theta_phi')
                self.boundary_energy = self._boundary_energy_theta_phi

        # print('tp mode:')
        if self.anc_theta == float('inf'):
            if self.anc_phi == -1:
                # print('theta hard')
                self.get_tp = self._get_tp_theta_hard
                self.set_tp = self._set_tp_theta_hard
            else:
                # print('theta phi hard')
                self.get_tp = self._get_tp_theta_phi_hard
                self.set_tp = self._set_tp_theta_phi_hard
        else:
            if self.anc_phi == -1:
                # print('theta soft')
                self.get_tp = self._get_tp_theta_hard
                self.set_tp = self._set_tp_theta_hard
            else:
                # print('theta phi soft')
                self.get_tp = self._get_tp_theta_phi_hard
                self.set_tp = self._set_tp_theta_phi_hard
        return self

    @property
    def norm(self):  # Dyn/cm^2 ?
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm
        self._K1_n = self.K1 / self.norm # cm^2
        self._K2_n = self.K2 / self.norm
        self._K3_n = self.K3 / self.norm
        # [epsilon E ^ 2] = dyn / cm ^ 2
        self._E_n = (self.eps_par - self.eps_perp) * (self.E * self.E) / self.norm # 1
        # [chi H ^ 2] = dyn / cm ^ 2
        self._H_n = self.chi * (self.H * self.H) / self.norm # 1
        self._anc_theta_n = self.anc_theta / self.norm
        self._anc_phi_n = self.anc_phi / self.norm

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        self.norm = self.K3 / (self.size[2] * self.size[2])

    @property
    def K1(self):
        return self._K1

    @K1.setter
    def K1(self, K1):
        self._K1 = K1
        self._K1_n = self.K1 / self.norm

    @property
    def K2(self):
        return self._K2

    @K2.setter
    def K2(self, K2):
        self._K2 = K2
        self._K2_n = self.K2 / self.norm

    @property
    def K3(self):
        return self._K3

    @K3.setter
    def K3(self, K3):
        self._K3 = K3
        self.norm = self.K3 / (self.size[2] * self.size[2])
        self._K3_n = self.K3 / self.norm

    def set_tp(self, tp):
        raise NotImplementedError

    def get_tp(self):
        raise NotImplementedError

    def _set_tp_theta_phi_soft(self, tp):
        self.theta = tp[0:self.N]
        self.phi = tp[self.N:]

    def _get_tp_theta_phi_soft(self):
        return np.concatenate(self.theta, self.phi)

    def _set_tp_theta_phi_hard(self, tp):
        self.theta[1:-1] = tp[:self.N - 1]
        self.phi[1:-1] = tp[self.N - 1:-1]

    def _get_tp_theta_phi_hard(self):
        return np.concatenate(self.theta[1:-1], self.phi[1:-1])

    def _set_tp_theta_soft(self, tp):
        self.theta = tp

    def _get_tp_theta_soft(self):
        return self.theta

    def _set_tp_theta_hard(self, tp):
        self.theta[1:-1] = tp

    def _get_tp_theta_hard(self):
        return self.theta[1:-1]

    def k_energy_density(self):
        raise NotImplementedError

    def _k_energy_density_theta_phi(self): # = 1
        return 0.5 / self.size[2] ** 2 * (
                self._K1_n * np.sin(self.theta) ** 2 * self.dtheta ** 2 +
                self._K2_n * np.sin(self.phi) ** 4 * self.dphi ** 2 +
                self._K3_n * (np.sin(self.theta) ** 2 * np.cos(self.theta) ** 2 * self.dphi ** 2 +
                              np.cos(self.theta) ** 2 * self.dtheta ** 2)
        )

    def _k_energy_density_theta(self):
        return 0.5 / self.size[2] ** 2 * (
                self._K1_n * np.sin(self.theta) ** 2 * self.dtheta ** 2 +
                self._K3_n * np.cos(self.theta) ** 2 * self.dtheta ** 2)

    def H_energy_density(self):
        raise NotImplementedError

    def _H_energy_density_par(self): #1
        return -self._H_n * (np.cos(self.theta) ** 2)

    def _H_energy_density_perp_theta(self):
        return -self._H_n * (np.sin(self.theta) ** 2)

    def _H_energy_density_perp_theta_phi(self):
        return -self._H_n * (np.sin(self.theta) * np.cos(self.phi)) ** 2

    def E_energy_density(self):
        return -self._E_n * (np.cos(self.theta) ** 2)

    def boundary_energy(self):
        raise NotImplementedError

    def _boundary_energy_theta_phi(self):
        return self._anc_theta_n * (
                (self.theta[0] - self.theta0) ** 2 + (self.theta[-1] - self.theta0) ** 2) + self._anc_phi_n * (
                       (self.phi[0] - self.phi0) ** 2 + (self.phi[-1] - self.phi0) ** 2)

    def _boundary_energy_theta(self):
        return self._anc_theta_n * ((self.theta[0] - self.theta0) ** 2 + (self.theta[-1] - self.theta0) ** 2)

    def _boundary_energy_rigid(self):
        return 0.

    def energy_n(self):
        return self.k_energy_density().sum() + self.H_energy_density().sum() + self.E_energy_density().sum() + self.boundary_energy()

    def energy(self):  # = Erg = Dyn * cm
        return self.size[0] * self.size[1] * self.size[2] * self.norm * (self.k_energy_density().sum() +
                                                                         self.H_energy_density().sum() +
                                                                         self.E_energy_density().sum() +
                                                                         self.boundary_energy())/self.N

    def energy_density_plot(self, title='default', show=False, save=None):
        if title == 'default':
            _title = 'H = {:.10f}, Energy = {:.10f}'.format(self.H, self.energy())
        else:
            _title = title

        fig, ax = plt.subplots()
        self.energy_density_plot_ax(ax)
        ax.set_xlabel('z')
        ax.set_ylabel('Energy')
        plt.legend()
        if title: plt.title(_title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def energy_density_plot_ax(self, ax):
        ax.plot(np.linspace(0., 1., self.N), self.k_energy_density(), 'r', label='k')
        ax.plot(np.linspace(0., 1., self.N), self.H_energy_density(), 'b', label='H')

    def restate(self, tp):
        self.set_tp(tp)
        self.rediff()
        return self.energy_n()

    # def strong_restate(self, state):
    #     self.newtp(state)
    #     return self.energy()

    def minimize_state(self):
        minimum = scipy.optimize.minimize(self.restate, self.get_tp(), method='CG', options={'gtol': 1e0, 'disp': True})
        self.restate(minimum.x)
        return self

    def get_epsilon(self):
        return self.N / ((1. / (self.eps_perp + (self.eps_par - self.eps_perp) * (np.cos(self.theta) ** 2))).sum())

    def __repr__(self):
        # self.plot(title='H = {:.10f} Energy = {:.10f}'.format(self.H, self.energy()), show=True)
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
               'E_b_n={}\n'.format(self.K1, self.K2, self.K3, self.mode, self._K1_n, self._K2_n, self.eps_par,
                                   self.eps_perp,
                                   self.chi, self.E, self.H,
                                   self.energy(), self.k_energy_density().sum(), self.H_energy_density().sum(),
                                   self.boundary_energy(), self.energy(), self.k_energy_density().sum(),
                                   self.H_energy_density().sum(),
                                   self.boundary_energy())


class Dependence():
    def __init__(self):
        self.mode = 'perp'
        self.Hlist = []
        self._K1 = 0.
        self._K2 = 0.
        self._K3 = 0.
        self._N = 0.
        self.states = []

    def init(self, material: Material, experiment: Experiment, K1=0., K2=0., K3=0., mode='perp', N=40):
        self.mode = mode
        match self.mode:
            case 'perp':
                self.Hlist = experiment.eps_perp[:, 0]
            case 'par':
                self.Hlist = experiment.eps_par[:, 0]

        self._K1 = K1
        self._K2 = K2
        self._K3 = K3
        self._N = N
        self.states = [Cell().init(material=material, K1=self.K1, K2=self.K2, K3=self.K3, mode=self.mode, H=H, N=N) for
                       H in self.Hlist]

        return self

    @property
    def eps(self):
        return np.array([lc.get_epsilon() for lc in self.states])

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        for state in self.states: state.N = N

    @property
    def K1(self):
        return self._K1

    @K1.setter
    def K1(self, K1):
        self._K1 = K1
        for state in self.states: state.K1 = K1

    @property
    def K2(self):
        return self._K2

    @K2.setter
    def K2(self, K2):
        self._K2 = K2
        for state in self.states: state.K2 = K2

    @property
    def K3(self):
        return self._K3

    @K3.setter
    def K3(self, K3):
        self._K3 = K3
        for state in self.states: state.K3 = K3

    def minimize(self, nodes=1):
        if nodes == 1:
            self.states = [lc.minimize_state() for lc in self.states]
        else:  # multiprocessing. Linux only
            assert isinstance(nodes, int)
            with Pool(nodes) as p:
                self.states = p.map(self.cm, self.states)
        return self

    def cm(self, x):
        return x.minimize_state()

    def lsr(self, experiment: Experiment):
        return np.linalg.norm(self.eps -
                              (experiment.eps_perp[:, 1] if self.mode == 'perp' else experiment.eps_par[:, 1])) / len(
            self.Hlist)

    def plot(self, show=False, save=None):
        fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        ax2 = None
        self.plot_ax(fig, ax1, ax2)
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$\theta,\,\varphi$')
        if save: plt.savefig(save)
        if show: plt.show()

    def plot_ax(self, fig, ax1, ax2=None):
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        cmap_theta = mpl.cm.autumn
        cmap_phi = mpl.cm.winter
        norm_theta = mpl.colors.Normalize(vmin=0, vmax=max(self.Hlist))
        norm_phi = mpl.colors.Normalize(vmin=0, vmax=max(self.Hlist))
        divider_theta = make_axes_locatable(ax1)
        cax_theta = divider_theta.append_axes("right", size="10%", pad=0.2)
        divider_phi = make_axes_locatable(ax1)
        cax_phi = divider_phi.append_axes("right", size="5%", pad=0.2)

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm_theta, cmap=cmap_theta),
                     cax=cax_theta, orientation='vertical', label='H, Oe')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm_phi, cmap=cmap_phi),
                     cax=cax_phi, orientation='vertical')
        colors_theta = cmap_theta(np.array(self.Hlist) / max(self.Hlist))
        colors_phi = cmap_phi(np.array(self.Hlist) / max(self.Hlist))
        # print(f'{colors_theta = }')
        for lc, color_theta, color_phi in zip(self.states, colors_theta, colors_phi):
            # print(lc)
            # lc.energy_density_plot(show=True)
            lc.plot_ax(ax1=ax1, ax2=ax2, color_theta=color_theta, color_phi=color_phi)

    def plot_one_by_one(self):
        for lc in self.states:
            print(lc)
            lc.energy_density_plot(show=True)
            lc.plot(show=True)

    '''        for idx,lc in enumerate(self.states):
            if title == 'H = ':
                title='H = {:.10f}'.format(self.Hlist[idx])
            lc.plot(title=title,show=show,save=save)
    '''

    def plot_eps(self, title=None, show=False, save=None):
        fig, ax = plt.subplots()
        self.plot_eps_ax(ax)
        ax.set_xlabel('H')
        ax.set_ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def plot_eps_ax(self, ax):
        ax.plot(self.Hlist, self.eps)

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

    def to_dict(self):
        return {'K1': self.K1, 'K2': self.K2, 'K3': self.K3, 'mode': self.mode, 'Hlist': self.Hlist,
                'theta': np.array([lc.theta for lc in self.states], dtype=float),
                'phi': np.array([lc.phi for lc in self.states], dtype=float)}

    def save(self, directory, material: Material):
        print(f'saving ' + directory + material.state_name + '_' + self.mode + '_{:.10f}_{:.10f}_{:.10f}.json'.format(
            self.K1, self.K2, self.K3))
        with open(directory + material.state_name + '_' + self.mode +
                  '_{:.10f}_{:.10f}_{:.10f}.json'.format(self.K1, self.K2, self.K3), 'w') as fp:
            json.dump(self.to_dict(), fp, cls=NumpyEncoder)

    def load_from_file(self, material: Material, experiment: Experiment, filename):
        try:
            with open(filename, 'r') as fp:
                print(f'{filename = }')
                file = json.load(fp)
                # print(file.keys())
                # print(f'{len(file["Hlist"]) = } {len(file["tp"]) = }')
                self.mode = file['mode']
                self.Hlist = file['Hlist']
                self._K1 = file['K1']
                self._K2 = file['K2']
                self._K3 = file['K3']
                self.states = [
                    Cell().init(material=material, K1=self.K1, K2=self.K2, K3=self.K3, mode=self.mode, H=H, theta=theta,
                                phi=phi) for H, theta, phi in zip(file["Hlist"], file["theta"], file["phi"])]
            return self
        except:
            warnings.warn(f'file {filename} loading failed')
            return None

    def nearest(self, H) -> Cell:
        idx = np.argmin(np.abs(np.array(self.Hlist) - H))
        return self.states[idx]

    def __repr__(self):
        return f'\n\tK1: {self.K1}\n' \
               f'\tK2: {self.K2}\n' \
               f'\tK3: {self.K3}\n'


def material_load(filename, load_messages=False):
    f = open(filename)
    system = json.load(f)
    _system = dict()
    for key in system.keys():
        try:
            value = literal_eval(system[key])
            if type(value) is float or type(value) is int:
                _system[key] = value
                if load_messages:
                    print(f'system[{key}] saved as number :\n{_system[key]}')
            else:
                _system[key] = np.array(value)
                if load_messages:
                    print(f'system[{key}] saved as numpy array :\n{_system[key]}')
        except:
            _system[key] = system[key]
            if load_messages:
                print(f'system[{key}] saved as it is:\n{_system[key]}')
    return _system


class Field:
    def __init__(self):
        self.perp_points = {}
        self.par_points = {}

    @property
    def N(self):
        N_perp = np.array([self.perp_points[p].N for p in self.perp_points], dtype=int)
        N_par = np.array([self.par_points[p].N for p in self.par_points], dtype=int)
        assert np.all(N_perp == N_perp[0])
        N_perp = N_perp[0]
        assert np.all(N_par == N_par[0])
        N_par = N_par[0]
        assert N_perp == N_par
        return N_perp

    @N.setter
    def N(self, N: int):
        for p in self.perp_points:
            self.perp_points[p].N = N
            self.par_points[p].N = N

    @property
    def K1_list(self):
        return sorted(list(set([key[0] for key in self.perp_points.keys()])))

    @property
    def K2_list(self):
        return sorted(list(set([key[1] for key in self.perp_points.keys()])))

    @property
    def K3_list(self):
        return sorted(list(set([key[2] for key in self.perp_points.keys()])))

    def init(self, material, experiment, K1_list=None, K2_list=None, K3_list=None, N=40):
        for k1 in K1_list:
            for k2 in K2_list:
                for k3 in K3_list:
                    kv = (k1, k2, k3)
                    self.perp_points[kv] = Dependence().init(material=material, experiment=experiment,
                                                             K1=k1, K2=k2, K3=k3, mode='perp', N=N)

                    self.par_points[kv] = Dependence().init(material=material, experiment=experiment,
                                                            K1=k1, K2=k2, K3=k3, mode='par', N=N)

    def from_directory(self, directory, material, experiment):
        for filename in os.listdir(directory):
            if filename[-5:] == '.json':
                d = Dependence()
                try:
                    d.load_from_file(filename=directory + filename, material=material, experiment=experiment)
                    match d.mode:
                        case 'perp':
                            print('perp')
                            assert np.all(d.Hlist == experiment.eps_perp[:, 0])
                            self.perp_points[(d.K1, d.K2, d.K3)] = d
                        case 'par':
                            print('par')
                            assert np.all(d.Hlist == experiment.eps_par[:, 0])
                            self.par_points[(d.K1, d.K2, d.K3)] = d
                except AssertionError as e:
                    print(e)
                    warnings.warn(f'{filename} load error, exception')
                except Exception as e:
                    print(e)
                    warnings.warn(f'{filename} load error')

    def save(self, directory, material):
        # [(print(d), self.perp_points[d].save(directory=directory, material=material)) for d in self.perp_points]
        [self.perp_points[d].save(directory=directory, material=material) for d in self.perp_points]
        [self.par_points[d].save(directory=directory, material=material) for d in self.par_points]

    def minimize(self, nodes=1):
        [self.perp_points[d].minimize(nodes=nodes) for d in self.perp_points]
        [self.par_points[d].minimize(nodes=nodes) for d in self.par_points]

    def plot_ax(self, ax, points=None):
        if points is None:
            points = [..., ..., ...]
        # nearest or all
        print(f'{points = }')
        k1 = [sorted(self.K1_list, key=lambda x: abs(x - points[0]))[0]] if type(points[0]) is float else self.K1_list
        k2 = [sorted(self.K2_list, key=lambda x: abs(x - points[1]))[0]] if type(points[1]) is float else self.K2_list
        k3 = [sorted(self.K3_list, key=lambda x: abs(x - points[2]))[0]] if type(points[2]) is float else self.K3_list

        print(k1, k2, k3)
        print(len(self.perp_points.keys()))
        print(self.perp_points.keys())
        for i in k1:
            for j in k2:
                for k in k3:
                    self.perp_points[(i, j, k)].plot_eps_ax(ax)
                    self.par_points[(i, j, k)].plot_eps_ax(ax)

    def nearest_perp(self, point):
        # returns Dependence (perp) on a grid to given point
        assert len(point) == 3
        k1 = sorted(self.K1_list, key=lambda x: abs(x - point[0]))[0]
        k2 = sorted(self.K2_list, key=lambda x: abs(x - point[1]))[0]
        k3 = sorted(self.K3_list, key=lambda x: abs(x - point[2]))[0]
        return self.perp_points[(k1, k2, k3)]

    def nearest_par(self, point) -> Dependence:
        # returns Dependence (par) on a grid to given point
        assert len(point) == 3
        k1 = sorted(self.K1_list, key=lambda x: abs(x - point[0]))[0]
        k2 = sorted(self.K2_list, key=lambda x: abs(x - point[1]))[0]
        k3 = sorted(self.K3_list, key=lambda x: abs(x - point[2]))[0]
        return self.par_points[(k1, k2, k3)]

    def lsr_perp(self, experiment: Experiment):
        l = {}
        for idx in self.perp_points:
            l[idx] = self.perp_points[idx].lsr(experiment=experiment)
        return l

    def lsr_par(self, experiment: Experiment):
        l = {}
        for idx in self.par_points:
            l[idx] = self.par_points[idx].lsr(experiment=experiment)
        return l

    def best(self, experiment: Experiment):
        l_perp = self.lsr_perp(experiment)
        l_par = self.lsr_par(experiment)
        l_total = {}
        for idx in self.perp_points:
            l_total[idx] = l_perp[idx] + l_par[idx]
        m = min(l_total, key=l_total.get)
        print(f'Minimum at {m}')
        return m

    def best_perp(self, experiment: Experiment) -> Dependence:
        l_perp = self.lsr_perp(experiment)
        l_par = self.lsr_par(experiment)
        l_total = {}
        for idx in self.perp_points:
            l_total[idx] = l_perp[idx] + l_par[idx]
        m = min(l_total, key=l_total.get)
        print(f'Minimum at {m}')
        return self.perp_points[m]

    def best_par(self, experiment: Experiment) -> Dependence:
        l_perp = self.lsr_perp(experiment)
        l_par = self.lsr_par(experiment)
        l_total = {}
        for idx in self.perp_points:
            l_total[idx] = l_perp[idx] + l_par[idx]
        m = min(l_total, key=l_total.get)
        print(f'Minimum at {m}')
        return self.par_points[m]

    def __repr__(self):
        return f'\n\tK1_list: {self.K1_list}\n' \
               f'\tK2_list: {self.K2_list}\n' \
               f'\tK3_list: {self.K3_list}\n' \
               f'\tpoints: {len(self.perp_points)}|{len(self.perp_points)}'


class Minimiser:
    def __init__(self):
        self.material = Material()
        self.experiment = Experiment()
        self.field = Field()

    def init(self, material, experiment, save_directory, K1_list=None, K2_list=None, K3_list=None, N=40):
        self.material = Material(material)
        self.experiment = Experiment(experiment)

        theta0_perp = math.acos(math.sqrt(
            (self.experiment.eps_perp[0, 1] - self.material.eps_perp) / (
                    self.material.eps_par - self.material.eps_perp)))
        phi0_perp = 0 * np.pi / 2
        theta0_par = math.acos(math.sqrt(
            (self.experiment.eps_par[0, 1] - self.material.eps_perp) / (
                    self.material.eps_par - self.material.eps_perp)))
        phi0_par = 0 * np.pi / 2
        print(f'{theta0_perp*180/np.pi = }\t{phi0_perp*180/np.pi = }')
        print(f'{theta0_par*180/np.pi = }\t{phi0_par*180/np.pi = }')

        self.material.theta0_perp = theta0_perp
        self.material.theta0_par = theta0_par
        self.material.phi0_perp = phi0_perp
        self.material.phi0_par = phi0_par

        self.save_directory = save_directory
        self.field = Field()
        self.field.init(material=self.material, experiment=self.experiment, K1_list=K1_list, K2_list=K2_list,
                        K3_list=K3_list, N=N)

        return self

    def load_from_directory(self, material, experiment, save_directory):
        self.material = Material(material)
        self.experiment = Experiment(experiment)

        theta0_perp = math.acos(math.sqrt(
            (self.experiment.eps_perp[0, 1] - self.material.eps_perp) / (
                    self.material.eps_par - self.material.eps_perp)))
        phi0_perp = 0 * np.pi / 2
        theta0_par = math.acos(math.sqrt(
            (self.experiment.eps_par[0, 1] - self.material.eps_perp) / (
                    self.material.eps_par - self.material.eps_perp)))
        phi0_par = 0 * np.pi / 2
        print(f'{theta0_perp*180/np.pi = }\t{phi0_perp*180/np.pi = }')
        print(f'{theta0_par*180/np.pi = }\t{phi0_par*180/np.pi = }')

        self.material.tp0_perp, self.material.tp0_par = np.array([theta0_perp, phi0_perp]), np.array(
            [theta0_par, phi0_par])
        self.save_directory = save_directory
        self.field = Field()
        self.field.from_directory(material=self.material, experiment=self.experiment, directory=save_directory)

        return self

    def minimize(self, nodes=1):
        assert isinstance(nodes, int) and nodes > 0
        self.field.minimize(experiment=self.experiment)

    def save(self):
        self.field.save(directory=self.save_directory, material=self.material)
        # self.perp_points = self.perp_points.reshape(-1)
        # for p in self.perp_points:
        #     p.save(directory=self.directory, state_name=self.state_name)
        # self.perp_points = self.perp_points.reshape(self.shape)
        # self.par_points = self.par_points.reshape(-1)
        # for p in self.par_points:
        #     p.save(directory=self.directory, state_name=self.state_name)
        # self.par_points = self.par_points.reshape(self.shape)
        # np.savez(self.directory + self.state_name + '.npz', Kv=self.Kv,
        #          perp_eps_diff=self.perp_eps_diff, par_eps_diff=self.par_eps_diff,
        #          exp_eps_perp=self.exp_eps_perp, exp_eps_par=self.exp_eps_par)

    # def diff(self, lc='all'):
    #     if isinstance(lc, Dependence):
    #         if lc.get_mode() == 'perp':
    #             return np.linalg.norm(self.exp_eps_perp[:, 1] - lc.get_eps_dependence())
    #         else:
    #             return np.linalg.norm(self.exp_eps_par[:, 1] - lc.get_eps_dependence())
    #
    #     elif isinstance(lc, np.ndarray) or isinstance(lc, list):
    #         lc_par_arr = self.par_points[lc].reshape(-1)
    #         for ct, lc0 in enumerate(lc_par_arr):
    #             lc_par_arr[ct] = self.diff(lc0)
    #         lc_perp_arr = self.perp_points[lc].reshape(-1)
    #         for ct, lc0 in enumerate(lc_perp_arr):
    #             lc_perp_arr[ct] = self.diff(lc0)
    #         return np.array([lc_par_arr, lc_perp_arr], dtype='float64')
    #
    #     elif lc == 'all':
    #         d = self.diff(np.full(self.perp_points.shape, True))
    #         return np.array((d[0].reshape(self.shape), d[1].reshape(self.shape)))

    def plot_only_practics(self, title=None, show=False, save=None):
        plt.plot(self.experiment.eps_par[:, 0], self.experiment.eps_par[:, 1], 'bx', label=r'$\varepsilon_{\parallel}$')
        plt.plot(self.experiment.eps_perp[:, 0], self.experiment.eps_perp[:, 1], 'rx', label=r'$\varepsilon_{\perp}$')
        plt.legend()
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    # def plot_smooth(self, lc='all', title=None, show=False, save=None):
    #     from scipy.signal import savgol_filter
    #     plt.plot(self.exp_eps_par[:, 0], self.exp_eps_par[:, 1], 'bx', label=r'$\varepsilon_{\parallel}$')
    #     plt.plot(self.exp_eps_perp[:, 0], self.exp_eps_perp[:, 1], 'rx', label=r'$\varepsilon_{\perp}$')
    #
    #     if lc == 'all':
    #         par_points = self.par_points.reshape(-1)
    #     else:
    #         par_points = [self.par_points[tuple(l.tolist())] for l in lc]
    #     for p in par_points:
    #         yhat = savgol_filter((self.exp_eps_par[:, 0], p.get_eps_dependence()), 21, 3)
    #         plt.plot(yhat[0], yhat[1])
    #
    #     if lc == 'all':
    #         perp_points = self.perp_points.reshape(-1)
    #     else:
    #         perp_points = [self.perp_points[tuple(l.tolist())] for l in lc]
    #     for p in perp_points:
    #         yhat = savgol_filter((self.exp_eps_perp[:, 0], p.get_eps_dependence()), 21, 3)
    #         plt.plot(yhat[0], yhat[1])
    #
    #     plt.legend()
    #     plt.xlabel('H')
    #     plt.ylabel(r'$\varepsilon$')
    #     if title: plt.title(title)
    #     if save:  plt.savefig(save)
    #     if show:  plt.show()
    #     plt.close('all')

    def plot(self, points=None, title=None, show=False, save=None):
        fig, ax = plt.subplots()

        ax.plot(self.experiment.eps_par[:, 0], self.experiment.eps_par[:, 1], 'bx', label=r'$\varepsilon_{\parallel}$')
        ax.plot(self.experiment.eps_perp[:, 0], self.experiment.eps_perp[:, 1], 'rx', label=r'$\varepsilon_{\perp}$')

        self.field.plot_ax(ax, points=points)

        plt.legend()
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def plot_best(self, title=None, show=False, save=None):
        fig, ax = plt.subplots()

        ax.plot(self.experiment.eps_par[:, 0], self.experiment.eps_par[:, 1], 'bx', label=r'$\varepsilon_{\parallel}$')
        ax.plot(self.experiment.eps_perp[:, 0], self.experiment.eps_perp[:, 1], 'rx', label=r'$\varepsilon_{\perp}$')

        kv = self.field.best(self.experiment)

        self.field.plot_ax(ax, points=kv)

        # Static and Dynamic Continuum Theory of Liquid Crystals pp.78
        Hc = np.pi / self.material.size[2] * np.sqrt(kv[2] / self.material.chi)
        print(f'{Hc = }')

        ax.axvline(x=Hc, c='grey')

        plt.legend()
        plt.xlabel('H')
        plt.ylabel(r'$\varepsilon$')
        if title: plt.title(title)
        if save:  plt.savefig(save)
        if show:  plt.show()
        plt.close('all')

    def rediff(self):
        self.par_eps_diff, self.perp_eps_diff = self.diff('all')

    def get_point(self, idx, mode='perp') -> Dependence:
        if mode == 'perp':
            return self.perp_points[tuple(idx)]
        else:
            return self.par_points[tuple(idx)]

    def best_K(self):
        best_par_idx = np.unravel_index(np.nanargmin(self.par_eps_diff), self.shape)
        best_perp_idx = np.unravel_index(np.nanargmin(self.perp_eps_diff), self.shape)
        best_Kv_par = self.Kv[best_par_idx]
        best_Kv_perp = self.Kv[best_perp_idx]
        best_par_diff = np.nanmin(self.par_eps_diff)
        best_perp_diff = np.nanmin(self.perp_eps_diff)
        return np.array([best_par_idx, best_perp_idx]), \
               np.array([best_Kv_par, best_Kv_perp]), \
               np.array([best_par_diff, best_perp_diff])

    def diff_plot(self):
        x = self.Kv.reshape(-1, 3)[:, 0]
        y = self.Kv.reshape(-1, 3)[:, 1]
        z = self.Kv.reshape(-1, 3)[:, 2]
        s = 100 * (self.perp_eps_diff.min() / self.perp_eps_diff.reshape(-1)) ** 10
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=s)
        ax.set_xlabel('$K_{11}$')
        ax.set_ylabel('$K_{22}$')
        ax.set_zlabel('$K_{33}$')
        plt.show()

    def diff_plot_K13(self):
        best_K = self.best_K()
        x = self.Kv[:, best_K[0][1, 1], :, 0]
        y = self.Kv[:, best_K[0][1, 1], :, 2]
        z = self.perp_eps_diff[:, best_K[0][1, 1], :]
        plt.contourf(x, y, z)
        plt.xlabel('$K_{11}$')
        plt.ylabel('$K_{33}$')
        plt.title('$K_{22} = ' + '{:.2e}$'.format(best_K[1][1, 1]))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f'\nexperiment:{self.experiment}\n' \
               f'material:{self.material}\n' \
               f'field:{self.field}'

    # def plot_maxangle(self,title=None,show=False,save=None):
    #     shape = self.perp_points.shape
    #     self.perp_points = self.perp_points.reshape(-1)
    #     for p in self.perp_points:
    #         p.plot_maxangle()
    #     self.perp_points = self.perp_points.reshape(shape)
