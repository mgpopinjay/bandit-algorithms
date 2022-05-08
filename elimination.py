import numpy as np
import matplotlib.pyplot as plt


### ELIMINATION ALGORITHM  ###
# Given knowledge of a fixed horizon, but not the gaps between arms

class Elimination():
    def __init__(self, rew_avg, num_iter):  ## Initialization

        self.means = rew_avg  # vector of true means of the arms
        self.num_iter = num_iter  # current time index t
        self.num_arms = rew_avg.size  # number of arms (k)
        self.genie_arm = np.argmax(rew_avg)  # best arm given the true mean rewards
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.round_time = 0  # round index
        self.epoch_time = 0  # time index in epoch l i.e: it goes from 1,..., m_l * size(self.A)
        self.A = np.arange(self.num_arms)  # active set in epoch l
        self.cum_reg = [0]  # cumulative regret
        self.m = np.ceil(
            2 ** (2 * self.round_time) * np.log(
                max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.round_time))))
        # sampling size m for each arm in the active set in epoch l
        self.restart()

        return None

    def restart(self):  ## Restart the algorithm:

        ## Reset self.epoch_time to zero, and the values of the empirical means and number of pulls to zero.
        self.epoch_time = 0.0
        self.num_pulls = np.zeros(self.num_arms)
        self.emp_means = np.zeros(self.num_arms)

        return None

    def get_best_arm(self):  ## For each time index in epoch l, find the best arm according to e-greedy

        # Using step 4 in Algo 2 of Exercise 6.8 in text Bandit Algorithm
        if self.epoch_time <= self.m * self.A.size:
            return int(self.A[int(self.epoch_time % self.A.size)])

    def update_stats(self, rew, arm):  ## Update the empirical means, the number of pulls, and increment epoch time.

        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1) + rew[arm]) / self.num_pulls[arm]
        self.epoch_time += 1

        return None

    def update_elim(self):  ## Update the active set
        self.m = np.ceil(
            2 ** (2 * self.round_time) * np.log(
                max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.round_time))))
        temp = []

        # Using step 6 in Algo 2 of Exercise 6.8 in text Bandit Algorithm
        for i in self.A:
            if self.emp_means[i] + 2 ** (-self.round_time) >= np.max(self.emp_means):
                temp.append(i)
        self.A = np.asarray(temp)

        return None

    def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

        reg = rew_vec[self.genie_arm] - rew_vec[arm]
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        arm = self.get_best_arm()
        self.update_stats(rew_vec, arm)  # update empirical means, pull count and epoch time.
        self.update_reg(rew_vec, arm)  # update regret.

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
    cov = 0.01 * np.eye(rew_avg.size)
    noise = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + noise

    return reward


### DRIVER ALGO  ###

def run_algo(rew_avg, num_iter, num_trial):
    for k in range(num_trial):
        regret = np.zeros((num_trial, num_iter))
        algo = Elimination(rew_avg, num_iter)
        algo.cum_reg = [0]
        algo.m = 10
        algo.round_time = 0.0

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        while len(algo.cum_reg) <= num_iter:

            if len(algo.cum_reg) >= num_iter:
                break

            # At each round, play each arm m times
            for t in range(int(algo.m) * algo.A.size):  # epoch size = sampling size * number of active arms

                if len(algo.cum_reg) >= num_iter:
                    break
                else:
                    rew_vec = get_reward(rew_avg)
                    algo.iterate(rew_vec)

            algo.update_elim()
            algo.restart()
            algo.round_time += 1

        regret[k, :] = np.asarray(algo.cum_reg)

    return regret


if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    rew_avg = np.asarray([0.8, 0.88, 0.5, 0.7, 0.65])
    num_iter, num_trial = int(1500), 250

    reg = run_algo(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    avg_reg.shape


    ### PLOT RESULT ###

    # Normal scale
    plt.plot(avg_reg, label="Elimination")
    plt.xlabel('time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with Elimination - Average')
    plt.legend()
    plt.show()

    # Log scale x-axis
    plt.semilogx(avg_reg, label="Elimination")
    plt.xlabel('time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with Elimination - Average (semilogx)')
    plt.legend()
    plt.show()
