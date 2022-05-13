import numpy as np
import pickle
import matplotlib.pyplot as plt

### EPSLION GREEDY ALGORITHM ###

# Given an arbitrary horizon with known gaps between arms

class egreedy():
    def __init__(self, rew_avg):

        self.means = rew_avg  # vector of true means of the arms
        self.num_arms = rew_avg.size  # number of arms (k)
        self.genie_arm = np.argmax(rew_avg)  # best arm given the true mean rewards
        self.chosen_arm = int  # arm chosen for exploitation
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.cum_reg = [0]  # cumulative regret
        self.iter = 0  # current iteration index
        self.eps = 1  # probability of choosing an arm uniformly at random at time t
        self.C = 1  # sufficiently large universal constant

        sorted_rew = np.sort(rew_avg)[::-1]  # sort true mean reward
        self.delta = sorted_rew[0] - sorted_rew[1]  # minimum suboptimality gap

        return None

    def restart(
            self):  ## Restart the algorithm: Reset the time index to zero and epsilon to 1 (done), the values of the empirical means,
        ## number of pulls, and cumulative regret to zero.

        self.num_pulls = np.zeros(rew_avg.size)
        self.emp_means = np.zeros(rew_avg.size)
        self.cum_reg = [0]
        self.iter = 0
        self.eps = 1

        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to e-greedy

        self.chosen_arm = np.argmax(self.emp_means)

    def update_stats(self, rew,
                     arm):  ## Update the empirical means, the number of pulls, epsilon, and increment the time index
        """
        In Epsilon-Greedy algorithm, instead of having a batch of log(n) samples per arm
        as in ETC algorithm, one simply ensures that at any time n, there's roughly log(n)
        samples, which, by Harmonic sequence, sum up to log(n) samples in aggregate.

        """

        self.iter += 1
        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1.0) + rew[arm]) / self.num_pulls[arm]


        # Update exploration probaility in porportion to num of arms, time and delta
        C = self.C
        k = self.num_arms
        t = self.iter
        delta = self.delta

        self.eps = min(1, C * k / (t * delta ** 2))

        return None

    def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

        reg = rew_vec[self.genie_arm] - rew_vec[arm]  # regret as the "loss" in reward
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        # Generate Bernoulli random variable with P(1) = eps at time t
        explore = np.random.binomial(1, self.eps, 1).item()

        if explore:
            # Case: Exploration
            self.chosen_arm = np.random.randint(self.num_arms, size=1)  # draw an arm
            self.update_stats(rew_vec, self.chosen_arm)  # update means, epsilon, iteration count & pull count
            self.update_reg(rew_vec, self.chosen_arm)  # update regret

        else:
            # Case: Exploitation
            self.get_best_arm()  # pick an arm based on max empirical mean
            self.update_stats(rew_vec, self.chosen_arm)  # update means, epsilon, iteration count & pull count
            self.update_reg(rew_vec, self.chosen_arm)  # update regret

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

    algo = egreedy(rew_avg)

    for j in range(num_trial):
        algo.restart()

        if (j + 1) % 10 == 0:
            print('Instance number = ', j + 1)

        for t in range(num_iter - 1):
            rew_vec = get_reward(rew_avg)
            algo.iterate(rew_vec)

        regret[j, :] = np.asarray(algo.cum_reg)

    return regret



if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    # reward_avg:  manually defined reward avg for each arm
    rew_avg = np.asarray([0.96,0.7,0.5,0.6,0.1])
    num_iter, num_trial = int(1e4), 10

    reg = run_algo(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    avg_reg.shape

    ### PLOT RESULT ###

    # Normal scale
    plt.plot(avg_reg, label="Epsilon-Greedy Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of Epsilon-Greedy Bandit')
    plt.legend()
    plt.show()

    # Log scale x-axis
    plt.semilogx(avg_reg, label="Epsilon-Greedy Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of Epsilon-Greedy Bandit (Semilogx)')
    plt.legend()
    plt.show()