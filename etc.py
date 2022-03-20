import numpy as np
import matplotlib.pyplot as plt
import math


### EXPLORE-THEN-COMMIT (ETC) ALGORITHM ###

class ETC():
  def __init__(self, rew_avg, m):

    self.means = rew_avg  # vector of true means of the arms
    self.m = m  # explore-exploit trade-off threshold
    self.num_arms = rew_avg.size  # number of arms (k)
    self.genie_arm = np.argmax(rew_avg)  # best arm given the true mean rewards
    self.chosen_arm = int  # arm chosen for exploitation
    self.num_pulls = np.zeros(rew_avg.size)  # vector of empirical means of arms
    self.emp_means = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
    self.cum_reg = [0]  # cumulative regret
    self.iter = 0  # current iteration index

    return None

  def restart(self):  ## Restart the algorithm: reset means, pull counts & regret

    self.num_pulls = np.zeros(rew_avg.size)
    self.emp_means = np.zeros(rew_avg.size)
    self.cum_reg = [0]
    self.iter = 0

    return None

  def get_best_arm(self):  ## For each time index, find the best arm according to ETC.

    self.chosen_arm = np.argmax(self.emp_means)

  def update_stats(self, rew,
                   arm):  ## Update the empirical means, the number of pulls, and increment the iteration index

    self.iter += 1
    self.num_pulls[arm] += 1
    self.emp_means[arm] += (rew[arm] - self.emp_means[arm]) / self.num_pulls[arm]

    return None

  def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

    reg = rew_vec[self.genie_arm] - rew_vec[arm]  # regret as the "loss" in reward
    # reg = self.means[self.genie_arm] - self.means[arm]  # regret as the "loss" in reward
    reg += self.cum_reg[-1]
    self.cum_reg.append(reg)

    return None

  def iterate(self, rew_vec):  ## Iterate the algorithm

    # Case: Exploration
    if self.iter < self.m * self.num_arms:  # check if t is within first m*k rounds
      self.chosen_arm = self.iter % self.num_arms  # selec next arm with (t mod k)
      self.update_stats(rew_vec, self.chosen_arm)  # update means, iteration count & pull count
      self.update_reg(rew_vec, self.chosen_arm)  # update regret

    # Case: Exploitation
    else:  # if t > m*k rounds
      self.get_best_arm()  # pick an arm based on max empirical mean
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

def run_algo(rew_avg, m, num_iter, num_trial):
  regret = np.zeros((num_trial, num_iter))
  algo = ETC(rew_avg, m)

  for j in range(num_trial):
    algo.restart()

    if (j + 1) % 10 == 0:
      print('Trial number = ', j + 1)

    for t in range(num_iter - 1):
      rew_vec = get_reward(rew_avg)
      algo.iterate(rew_vec)

    regret[j, :] = np.asarray(algo.cum_reg)

  return regret



if __name__ == '__main__':

  ### INITIALIZE EXPERIMENT PARAMETERS ###

  # reward_avg:  manually defined reward avg for each arm
  rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
  num_iter, num_trial = int(1e4), 30
  m = [1, 10, 100, 1000]

  # Compute optimal m with Equation 6.5 of text "Bandit Algorithm"
  sorted_rew = np.sort(rew_avg)               # sort from smallest to largest reward
  delta = sorted_rew[-1] - sorted_rew[0]      # (best-worst) helps avoid picking the worst arm

  m_optim = max(1, math.ceil( (4 / delta**2) * ( math.log(num_iter * delta**2 / 4)) ) )
  m.append(m_optim)
  m.sort()

  
  ### RUN EXPERIMENT ###

  regrets = []
  for m_val in m:
    reg = run_algo(rew_avg, m_val, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    regrets.append(avg_reg)


  ### PLOT RESULT ###

  vertical = len(rew_avg) * m_optim      # theoretical inflection point for cumulative regret

  for m_val, r in zip(m, regrets):
    plt.plot(r, label="m="+str(m_val))

  plt.axvline(x=vertical, label="x = len(rew_avg) * m")
  plt.xlabel('iterations')
  plt.ylabel('cumulative regret')
  plt.title('Cumulative Regret with ETC Bandit')
  plt.legend()
  plt.show()