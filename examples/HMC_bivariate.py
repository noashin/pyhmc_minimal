import numpy as np
from scipy.stats import multivariate_normal
from .hmcparameter import HMCParameter
from .hmc import HMC


class StateMultivarNormal(HMCParameter):

    def __init__(self, init_val, mu, sigma_inv):
        self.super().__init__(np.array(init_val))
        self.mu = mu
        self.sigma_inv = sigma_inv

        #def gen_init_value(self):
        #   self.value = multivariate_normal.rvs(self.mu, self.sigma)

    def get_energy_grad(self):
        return np.dot(self.sigma_inv, self.value)

    def get_energy(self):
        return np.dot(self.value.transpose(), np.dot(self.sigma_inv, self.value)) / 2


class VelParam(HMCParameter):

    def __init__(self, init_val):
        super().__init__(np.array(init_val))
        dim = np.array(init_val).shape
        self.mu = np.zeros(dim)
        self.sigma = 1

    def gen_init_value(self):
        self.value = multivariate_normal.rvs(self.mu, self.sigma)

    def get_energy_grad(self):
        return self.value

    def get_energy(self):
        return np.dot(self.value, self.value) / 2


def run_HMC():
    state = StateMultivarNormal(np.array([0, 0]), [1, 1], 1)
    vel = VelParam(np.array[0.1, 0.1])

    delta = 0.3
    n = 10
    m = 1000
    hmc = HMC(state, vel, delta, n, m)

    hmc.HMC()

    return
