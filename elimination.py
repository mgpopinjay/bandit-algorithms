import numpy as np
import math


### ELIMINATION ALGORITHM  ###
# Given knowledge of the horizon, but not the gaps between arms

class Elimination():
    def __init__(self, rew_avg, num_iter):  ## Initialization

        self.means = rew_avg                     # vector of true means of the arms
        self.num_iter = num_iter                 # current time index t
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(rew_avg)      # best arm given the true mean rewards
        self.chosen_arm = int                    # arm chosen for exploitation
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.global_time = 0.0
        self.epoch_time = 0.0                    # time index in epoch l i.e: it goes from 1,..., m_l * size(self.A)
        self.A = np.arange(self.num_arms)        # active set in epoch l
        self.cum_reg = [0]                       # cumulative regret
        self.m = np.ceil(
            2 ** (2 * self.global_time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.global_time))))
                                                 # sampling size m for each arm in the active set in epoch l
        self.delta = 1                           # separation criteria for measuring gaps

        self.restart()
        return None


    def restart(self):  ## Restart the algorithm:

        ## Reset self.epoch_time to zero, and the values of the empirical means and number of pulls to zero.
        self.epoch_time = 0.0
        self.num_pulls = np.zeros(rew_avg.size)
        self.emp_means = np.zeros(rew_avg.size)

        return None

    def get_best_arm(self):  ## For each time index in epoch l, find the best arm according to e-greedy

    ## Hint: this is similar to ETC: use step 4 in algorithm 2 in the textbook
        self.chosen_arm = np.argmax(self.emp_means)  ### what means to use???

    def update_stats(self, rew, arm):  ## Update the empirical means, the number of pulls, and increment self.epoch_time

        self.num_pulls[arm] += 1
        self.emp_means[arm] += (rew[arm] - self.emp_means[arm]) / self.num_pulls[arm]
        self.epoch_time += 1

        return None

    def update_elim(self):  ## Update the active set
        self.m = np.ceil(
            2 ** (2 * self.global_time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.global_time))))

        # Compute confidence bonus
        bonus = np.sqrt( np.log(self.num_iter * self.delta ** 2) / self.delta ** 2 )
        print("bonus:", bonus)

        # Compare high-low estimates and elimiate if gaps are big
        rated_best = max(self.emp_means) - bonus

        for arm, mean in enumerate(self.emp_means):
            rated_low = mean + bonus
            if rated_low < rated_best:


        # Subtract eliminated arms from original set
        self.A

        return None

    def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

        ## Your code here

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        for arm in self.A:

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

    algo = Elimination(rew_avg, num_iter)

    for k in range(num_trial):

        # Refresh algo's global variables for each trial
        algo.cum_reg = [0]
        algo.m = 10
        algo.global_time = 0.0

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        while len(algo.cum_reg) <= num_iter:

            if len(algo.cum_reg) >= num_iter:
                break

            for t in range(int(algo.m) * algo.A.size):   # epoch size = sampling size * number of active arms

                if len(algo.cum_reg) >= num_iter:
                    break

                else:
                    rew_vec = get_reward(rew_avg)
                    algo.iterate(rew_vec)

            algo.update_elim()

            algo.restart()

            algo.global_time += 1

        regret[k, :] = np.asarray(algo.cum_reg)

    return regret



if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    rew_avg = np.asarray([0.8, 0.88, 0.5, 0.7, 0.65])
    num_iter, num_trial = int(1500), 250

    regret = run_algo(rew_avg, num_iter, num_trial)