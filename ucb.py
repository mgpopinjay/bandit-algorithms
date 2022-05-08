import numpy as np
import matplotlib.pyplot as plt


### UPPER-CONFIDENCE BOUND ALGORITHM  ###
# Given unknown horizon and unknown gaps between arms

class UCB(object):
    def __init__(self, rew_avg):  ## Initialization

        self.means = rew_avg                     # vector of true means of the arms
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


    def restart(self):  ## Restart the algorithm

        # Reset the time index to zero and the upper confidence values to high values.
        self.time = 0.0
        self.ucb_arr = 1e5 * np.ones(self.num_arms)

        # Reset values of the empirical means, number of pulls, and cumulative regret vector to zero.
        self.num_pulls = np.zeros(rew_avg.size)
        self.emp_means = np.zeros(rew_avg.size)
        self.cum_reg = [0]

        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to UCB

        self.chosen_arm = np.argmax(self.ucb_arr)

    def update_stats(self, arm, rew):  ## Update the empirical means, the number of pulls, and increment the time index

        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1) + rew[arm]) / self.num_pulls[arm]
        self.time += 1

        return None

    def update_ucb(self):  ## Update the vector of upper confidence bounds

        # Use the Infinite Horizon version, Algo 6 in 8.1 of book "Bandit Algorithms"
        func = 2 * np.log(1 + self.time*((np.log(self.time))**2))
        for i in range(self.num_arms):
            if self.num_pulls[i] == 0:
                continue
            else:
                self.ucb_arr[i] = self.emp_means[i] + np.sqrt(func/self.num_pulls[i])

        return None

    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector

        reg = rew_vec[self.genie_arm] - rew_vec[arm]  # regret as the "loss" in reward
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        # At any time, play the arm with the largest empirical estimate plus the UCB
        self.chosen_arm = np.argmax(self.ucb_arr)

        # Update regret and key variables
        self.update_reg(self.chosen_arm, rew_vec)
        self.update_stats(self.chosen_arm, rew_vec)
        self.update_ucb()

        return None



### BANDIT ARM REWARD NOISE FUNCTION ###

def get_reward(rew_avg):

    # Add sub-gaussian noise to reward
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
    noise = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + noise

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

    # Log scale x-axis
    plt.semilogx(avg_reg, label="UCB Avg. Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of UCB Bandit (Semilogx)')
    plt.legend()
    plt.show()