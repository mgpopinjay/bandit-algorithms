import numpy as np
import matplotlib.pyplot as plt


### EXP3 ALGORITHM  ###

class EXP3(object):
    def __init__(self, rew_avg, eta):  ## Initialization

        self.means = rew_avg                                    # vector of true means of the arms
        self.num_iter = num_iter                                # current time index t
        self.num_arms = rew_avg.size                            # number of arms (k)
        self.genie_arm = np.argmax(self.means)                  # best arm given the true mean rewards
        self.chosen_arm = int
        self.num_plays = np.zeros(rew_avg.size)                 # vector of number of times that arm k has been pulled
        self.S = np.zeros(rew_avg.size)                         # vector of estimated reward by the end of time t
        self.hindsight_rew = np.zeros(rew_avg.size)
        self.probs_arr = np.ones(rew_avg.size) / rew_avg.size   # sampling distribution vector P_t
        self.cum_reg = [0]                                      # cumulative regret
        self.time = 0.0                                         # current time index
        self.eta = eta                                          # learning rate
        self.restart()

        return None


    def restart(self):  ## Reset self.time, num_plays, S, and cum_reg to zero; and set probs_arr to be uniform

        self.time = 0.0
        self.num_plays = np.zeros(self.num_arms)
        self.S = np.zeros(self.num_arms)
        self.hindsight_rew = np.zeros(rew_avg.size)
        self.probs_arr = np.ones(self.num_arms) / self.num_arms
        self.cum_reg = [0]

        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to EXP3

        self.chosen_arm = np.random.choice(len(self.means), 1, p=self.probs_arr)


    def update_exp3(self, arm, rew_vec):  ## Compute probs_arr and update the total estimated reward for each arm

        # Update prob distribution
        self.probs_arr = np.exp(self.eta * self.S) / np.sum(np.exp(self.eta * self.S))

        # Update total estimated reward
        prob = self.probs_arr[arm]
        reward = self.means[arm]

        for i in range(self.num_arms):
            if arm == i:
                self.S[i] += 1 - ((1 - reward) / prob)
            else:
                self.S[i] += 1

        # Track hindsight rewards
        self.hindsight_rew += rew_vec[self.genie_arm]

        return None

    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector

        # max reward in hindsight
        hindsight_arm = np.argmax(self.hindsight_rew)
        hindsight_max = rew_vec[hindsight_arm]
        reg = hindsight_max - rew_vec[arm]

        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm
        self.time += 1.0
        self.get_best_arm()  # sample an arm based on latest distribution
        self.update_exp3(self.chosen_arm, rew_vec)
        self.update_reg(self.chosen_arm, rew_vec)

        self.num_plays[self.chosen_arm] += 1

        return None


### BANDIT ARM REWARD NOISE FUNCTION ###

def get_reward(rew_avg, var):
    mean = np.zeros(rew_avg.size)
    cov = var * np.eye(rew_avg.size)
    epsilon = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + epsilon

    return reward


### DRIVER ALGO  ###

def run_algo(rew_avg, eta, num_iter, num_trial, var):
    regret = np.zeros((num_trial, num_iter))

    algo = EXP3(rew_avg, eta)

    for k in range(num_trial):
        algo.restart()

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        for t in range(num_iter - 1):
            rew_vec = get_reward(rew_avg, var)
            algo.iterate(rew_vec)
        regret[k, :] = np.asarray(algo.cum_reg)

    return regret


if __name__ == '__main__':

    ### INITIALIZE EXPERIMENT PARAMETERS ###

    rew_avg = np.asarray([0.8, 0.7, 0.5])
    num_iter, num_trial = int(2e3), 1

    eta = np.sqrt(np.log(rew_avg.size) / (num_iter * rew_avg.size))
    var = 0.01

    reg = run_algo(rew_avg, eta, num_iter, num_trial, var)

    avg_reg = np.mean(reg, axis=0)


    ### PLOT RESULT ###

    # Normal scale
    plt.plot(avg_reg, label="EXP3")
    plt.xlabel('time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with EXP3 - Average')
    plt.legend()
    plt.show()

    # Log scale x-axis
    plt.semilogx(avg_reg, label="EXP3")
    plt.xlabel('time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with EXP3 - Average (semilogx)')
    plt.legend()
    plt.show()