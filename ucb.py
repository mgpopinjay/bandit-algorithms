import numpy as np
import math


### ELIMINATION ALGORITHM  ###
# Given knowledge of the horizon, but not the gaps between arms

class UCB(object):
    def __init__(self, rew_avg):  ## Initialization

        self.means = rew_avg                     # vector of true means of the arms
        self.num_iter = num_iter                 # current time index t
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(self.means)   # best arm given the true mean rewards
        self.chosen_arm = int                    # arm chosen for exploitation
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.ucb_arr = np.zeros(rew_avg.size)    # vector containing the upper confidence bounds
        self.cum_reg = [0]                       # cumulative regret
        self.time = 0.0
        self.restart()

        return None


    def restart(self):  ## Restart the algorithm: Reset the time index to zero and the upper confidence values to high values (done),
        ## Set the values of the empirical means, number of pulls, and cumulative regret vector to zero.

        self.time = 0.0
        self.ucb_arr = 1e5 * np.ones(self.num_arms)

        ## Your code here

        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to UCB

        ## Your code here

    def update_stats(self, arm, rew):  ## Update the empirical means, the number of pulls, and increment the time index

        ## Your code here

        return None

    def update_ucb(self):  ## Update the vector of upper confidence bounds

        ## Your code here

        return None

    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector

        ## Your code here

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        ## Your code here

        return None




### BANDIT ARM REWARD NOISE FUNCTION ###

def get_reward(rew_avg):
    """
    Reward incoporates a noise, epsilon, modeled as a length k random variable
    drawn from a multivariate normal distribution, where the mean is zero vector
    covariance is a (k x k) idendity matrix--making the arms uncorrelated as
    meant to in an unstructure environment.
    """

    # Add epsilon (sub-gaussian noise) to reward
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
    epsilon = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + epsilon

    return reward


### DRIVER ALGO  ###

def run_algo(rew_avg, num_iter, num_trial):
    regret = np.zeros((num_trial, num_iter))

    algo = UCB(rew_avg)

    for k in range(num_trial):
        algo.restart()

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        for t in range(num_iter - 1):
            rew_vec = get_reward(rew_avg)

            algo.iterate(rew_vec)

        regret[k, :] = np.asarray(algo.cum_reg)

    return regret



if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    rew_avg = np.asarray([0.8, 0.96, 0.7, 0.5, 0.4, 0.3])
    num_iter, num_trial = int(5e4), 30

    regret = run_algo(rew_avg, num_iter, num_trial)