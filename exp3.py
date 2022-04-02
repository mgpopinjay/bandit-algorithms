import numpy as np
import matplotlib.pyplot as plt


### EXP3 ALGORITHM  ###



class EXP3(object):
    def __init__(self, rew_avg, eta):  ## Initialization

        self.means = rew_avg                     # vector of true means of the arms
        self.num_iter = num_iter                 # current time index t
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(self.means)   # best arm given the true mean rewards
        self.num_plays = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.probs_arr = []                      # sampling distribution vector P_t
        self.S = np.zeros(rew_avg.size)          # vector of estimated reward by the end of time t
        self.cum_reg = [0]                       # cumulative regret
        self.time = 0.0                          # current time index
        self.eta = eta                           # learning rate
        self.restart()

        return None


    def restart(self):  ## Restart the algorithm:
        ## Reset the values of self.time, num_plays, S, and cum_reg to zero, and set probs_arr to be uniform

        self.time = 0.0
        self.num_plays = np.zeros(rew_avg.size)
        self.S = np.zeros(rew_avg.size)
        self.cum_reg = [0]
        self.probs_arr = []


        ## Your Code here

        return None

    def get_best_arm(self):  ## For each time index, find the best arm according to EXP3

    ## use np.random.choice

    def update_exp3(self, arm, rew_vec):  ## Compute probs_arr and update the total estimated reward for each arm

        ## Fill in

        return None

    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector

        ## Fill in
        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm
        self.time += 1.0

        ## Fill in

        return None


### BANDIT ARM REWARD NOISE FUNCTION ###

def get_reward(rew_avg):
    """

    """

    # Add epsilon (sub-gaussian noise) to reward
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
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
    num_iter, num_trial = int(2e3), 20

    eta = np.sqrt(np.log(rew_avg.size) / (num_iter * rew_avg.size))
    var = 0.01

    reg = run_algo(rew_avg, eta, num_iter, num_trial, var)

    avg_reg = np.mean(reg, axis=0)
    # avg_reg.shape


    ### PLOT RESULT ###

    # Normal scale
    plt.plot(avg_reg, label="Exp3 Avg. Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of Exp3 Adversarial Bandit')
    plt.legend()
    plt.show()

    # # Log scale x-axis
    # plt.semilogx(avg_reg, label="Exp3 Avg. Regret")
    # plt.xlabel('iterations')
    # plt.ylabel('cumulative regret')
    # plt.title('Cumulative Regret of Exp3 Adversarial Bandit (Semilogx)')
    # plt.legend()
    # plt.show()