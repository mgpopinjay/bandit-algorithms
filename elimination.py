import numpy as np
import matplotlib.pyplot as plt
import math


### ELIMINATION ALGORITHM  ###
# Given knowledge of a fixed horizon, but not the gaps between arms

class Elimination():
    def __init__(self, rew_avg, num_iter):  ## Initialization

        self.means = rew_avg                     # vector of true means of the arms
        self.num_iter = num_iter                 # current time index t
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(rew_avg)      # best arm given the true mean rewards
        self.chosen_arm = int                    # arm chosen for exploitation
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.global_time = 0
        self.epoch_time = 0                      # time index in epoch l i.e: it goes from 1,..., m_l * size(self.A)
        self.A = np.arange(self.num_arms)        # active set in epoch l
        self.cum_reg = [0]                       # cumulative regret
        self.m = 0

        # self.m = np.ceil(
        #     2 ** (2 * self.global_time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.global_time))))
        #                                          # sampling size m for each arm in the active set in epoch l

        self.delta = 1.0                         # separation criteria for measuring gaps
        self.mask = np.ones(len(self.A), dtype=bool)

        self.restart()
        return None


    def restart(self):  ## Restart the algorithm:

        ## Reset self.epoch_time to zero, and the values of the empirical means and number of pulls to zero.

        self.epoch_time = 0.0

        self.num_pulls = np.zeros(self.A.size)
        self.emp_means = np.zeros(self.A.size)
        self.genie_arm = np.argmax(self.means)

        self.mask = np.ones(len(self.A), dtype=bool)
        self.mask_id = []

        self.m = 10
        self.delta = 1.0

        print("\nRestart...")
        print("")


        return None

    def get_best_arm(self):  ## For each time index in epoch l, find the best arm according to e-greedy

    ## Hint: this is similar to ETC: use step 4 in algorithm 2 in the textbook
        self.chosen_arm = np.argmax(self.emp_means)  ### what means to use???

    def update_stats(self, rew, arm):  ## Update the empirical means, the number of pulls, and increment self.epoch_time

        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1) + rew[arm]) / self.num_pulls[arm]
        # self.emp_means[arm] += (rew[arm] - self.emp_means[arm]) / self.num_pulls[arm]
        self.epoch_time += 1

        return None

    def update_elim(self):  ## Update the active set
        self.m = np.ceil(
            2 ** (2 * self.global_time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.global_time))))
        # print("\nm:", self.m)

        # Compute confidence bonus : Add 1 to Global time???
        print("Global time + 1 :", self.global_time+1, type(self.global_time))
        print("delta / m:", self.delta, " / ", self.m, type(self.delta), type(self.m))

        bonus = np.sqrt( np.log((self.global_time + 1) * self.delta ** 2) / (2 * self.m) )

        print("bonus:", bonus)

        # Compare high-low estimates and eliminate lower arms if gaps are big
        rated_best = max(self.emp_means) - bonus

        for idx, arm in enumerate(self.A):
            rated_low = self.emp_means[arm] + bonus
            if rated_low < rated_best:
                self.mask_id.append(idx)

        print("Pre-mask:", self.emp_means)

        # Mask active arm list, emperical means and number of pulls
        self.mask[self.mask_id] = False
        self.emp_means = self.emp_means[self.mask]
        self.num_pulls = self.num_pulls[self.mask]
        self.A = self.A[self.mask]

        print("Reduced Arms:", self.A)
        print("self.means:", self.means)
        print("self.mask:", self.mask)

        self.means = self.means[self.mask]

        # Re-index active arms
        self.A = np.arange(self.A.size)

        print("After-mask:", self.emp_means)

        return None

    def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

        # print("rew_vec", rew_vec.size)
        # print("genie_arm:", self.genie_arm)
        # print("arm:", arm)

        reg = rew_vec[self.genie_arm] - rew_vec[arm]  # regret as the "loss" in reward
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        print("\nActive Arms:", self.A)

        arm = int(self.epoch_time) % self.A.size
        print("Pulling Arm:", arm)
        print("Rew_vec:", rew_vec)

        self.update_stats(rew_vec, arm)               # update means, iteration count & pull count
        self.update_reg(rew_vec, arm)     # update regret


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
        print("\nTrial:", k)
        print("")

        # Refresh algo's global variables for each trial
        algo.cum_reg = [0]
        algo.m = 10
        algo.global_time = 0.0
        algo.means = rew_avg
        algo.mask_id = []
        algo.mask = np.ones(len(algo.A), dtype=bool)

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        while len(algo.cum_reg) <= num_iter:

            if len(algo.cum_reg) >= num_iter:
                break

            # At each epoch, play each arm m times
            # for t in range(int(algo.m)):   # epoch size = sampling size * number of active arms
            for t in range(int(algo.m) * algo.A.size):  # epoch size = sampling size * number of active arms

                # if len(algo.cum_reg) >= num_iter:
                if len(algo.cum_reg) == num_iter:
                    break

                else:
                    print("algo.means:", algo.means)
                    rew_vec = get_reward(algo.means)
                    algo.iterate(rew_vec)

                print("\nCum_reg Size:", len(algo.cum_reg))

            # if len(algo.cum_reg) < num_iter:
            print("\nBegin Elimination...\n")

            algo.update_elim()

            algo.delta = algo.delta / 2

            algo.restart()
            algo.global_time += 1

        # print("cum_reg.size:", len(algo.cum_reg))

        regret[k, :] = np.asarray(algo.cum_reg)

    return regret



if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    rew_avg = np.asarray([0.8, 0.88, 0.5, 0.7, 0.65])
    num_iter, num_trial = int(1500), 2

    reg = run_algo(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    avg_reg.shape


    ### PLOT RESULT ###

    # Normal scale
    plt.plot(avg_reg, label="UCB Avg. Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of UCB Bandit')
    plt.legend()
    plt.show()
