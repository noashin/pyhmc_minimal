import numpy as np


class HMC():
    def __init__(self, state_param, velocity_param, delta=0.1, n=10, m=500):
        
        # Attributes necessary for the HMC algorithm
        self.state_param = state_param
        self.velocity_param = velocity_param
        self.delta = delta
        self.n = n
        self.m = m

        # Attributes used to monitor the HMC
        self.state_samps = [] # the actual samples
        self.vel_samps = []
        self.alphas = []
        self.all_state_samps = [] # also the rejected proposals
        self.all_vel_samps = []
        self.leap_frog_steps = []
        self.accepted = 0


    def get_state(self):
        return self.state_param

    def get_samples(self):
        return self.state_samps

    def calc_acceptence_rate(self):
        return self.accepted / len(self.state_samps)

    def leapfrog(self, save_steps, *args):

        # Half a step
        tmp_vel = self.velocity_param.get_value() - self.delta / 2 * self.state_param.get_energy_grad(*args)
        self.velocity_param.set_value(tmp_vel)

        for i in range(self.n):
            state_val = self.state_param.get_value() + self.delta * self.velocity_param.get_energy_grad()
            self.state_param.set_value(state_val)

            vel_val = self.velocity_param.get_value() - self.delta * self.state_param.get_energy_grad(*args)
            self.velocity_param.set_value(vel_val)

            if save_steps:
                self.leap_frog_steps.append([state_val, vel_val])

        # another half a step
        state_val = self.state_param.get_value() + self.delta * self.velocity_param.get_energy_grad()
        self.state_param.set_value(state_val)

        vel_val = self.velocity_param.get_value() - self.delta / 2 * self.state_param.get_energy_grad(*args)
        # negate momentum to make the proposal symmetric
        self.velocity_param.set_value(- vel_val)

    def HMC(self, *args):
        for i in range(self.m):
            self.velocity_param.gen_init_value()
            vel_val_old = self.velocity_param.get_value()
            state_val_old = self.state_param.get_value()

            self.leapfrog(False, *args)

            prob = np.exp(- self.state_param.get_energy(*args) + self.state_param.get_energy_for_value(
                state_val_old, *args) - self.velocity_param.get_energy() + self.velocity_param.get_energy_for_value(
                vel_val_old))

            alpha = np.min((1, prob))
            self.alphas.append(alpha)
            self.all_state_samps.append(self.state_param.get_value())
            self.all_vel_samps.append(self.velocity_param.get_value())

            p = np.random.random()

            if p > alpha:
                # reject - set the state and velocity values to the previous
                self.state_param.set_value(state_val_old)
                self.velocity_param.set_value(vel_val_old)
            else:
                self.accepted += 1

            self.state_samps.append(self.state_param.get_value())
            self.vel_samps.append(self.velocity_param.get_value())