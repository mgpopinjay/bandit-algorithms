import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm


# Context-action feature map
available_arms = np.array([
(1, 1, 0, 0),
(1, 0, 1, 0),
(1, 0, 0, 1),
(0, 1, 1, 0),
(0, 1, 0, 1),
(0, 0, 1, 1)])


class context_arm(object):
  def __init__(self, available_arms=available_arms):
    self.theta = np.array((0.1, 0.4, 0.2, 0.3))   
    self.available_arms = np.array(available_arms)
    self.num_arms = len(available_arms)

  def pull_arm(self, arm_idx): # Return X_t given the index of the arm played    
    action_t = available_arms[arm_idx]
    noise_t = np.random.normal(0, .25)
    reward = self.theta.T @ action_t + noise_t
    
    return reward

  def genie_reward(self): # Return the genie reward

    # Genie knows both theta and full feature map of best actions according to context.
    reward = max(self.theta.T @ available_arms.T)

    return reward



class LinUCB():

  def __init__(self, available_arms): # Initialization
    self.arms = available_arms
    self.num_arms = len(self.arms)               # number of arms in the decision set
    self.d = len(self.arms[0])                   # dimension of the space \mathbb{R}^d
    self.reward_history = []                     # vector containing past rewards
    self.reward_est = np.zeros(self.num_arms)    # total reward obtained using self.arms[arm_idx
    self.pull_cnter = np.zeros(self.num_arms)    # number of times self.arms[arm_idx] has been pulled
    self.alpha = 2
    self.V = np.identity(self.d)                 # d*d matrix defined in eq 19.6  
    self.b = np.atleast_2d(np.zeros(self.d)).T   # summation defined in eq 19.5 



  def choose_arm(self): # Compute UCB scores and return the selected arm and its index

    # Equation 19.5    
    V_inverse = np.linalg.inv(self.V) 
    theta_hat = V_inverse @ self.b
    ucb_scores = []

    # Compute UCB scores as defined in equation 19.2
    for arm in self.arms:
      arm = np.atleast_2d(arm).T
      ucb = theta_hat.T @ arm
      ucb += self.alpha * np.sqrt(arm.T@(V_inverse@arm))
      ucb_scores.append(ucb[0][0])

    # Per equation 19.3, pick an arm at time t based on above USB scores 
    arm_idx = np.argmax(ucb_scores)
    arm = self.arms[arm_idx]

    return arm, arm_idx

  
  def update(self, reward, arm_idx): # update the parameters

    self.pull_cnter[arm_idx] += 1
    
    # Update reward vector by chosen arm and append to history
    self.reward_est[arm_idx] += reward
    self.reward_history.append(reward)
    
    # Update self.V and self.b using equations (19.5) and (19.6)
    arm = np.atleast_2d(self.arms[arm_idx]).T
    self.V += arm * arm.T
    self.b += reward * arm


### Experiment Runner Function ###
def regret_vs_horizon(REPEAT, HORIZON):

    LinUCB_history = np.zeros(HORIZON)
    my_context_arm = context_arm()

    for rep in tqdm(range(REPEAT)):

        LinUCB_instance = LinUCB(available_arms)
        
        for i in range(HORIZON):
            arm, arm_idx = LinUCB_instance.choose_arm()
            reward = my_context_arm.pull_arm(arm_idx)
            LinUCB_instance.update(reward, arm_idx)
        LinUCB_history += np.array(LinUCB_instance.reward_history)

    LinUCB_expected_reward = LinUCB_history / REPEAT
    LinUCB_expected_reward = np.cumsum(LinUCB_expected_reward)
    best_rewards = my_context_arm.genie_reward()
    best_rewards = best_rewards * np.linspace(1, HORIZON, num=HORIZON)
    LinUCB_regret = best_rewards - LinUCB_expected_reward
    return LinUCB_regret


if __name__ == '__main__':

    ### Experiments ###
    REPEAT = 500
    HORIZON = 10000
    LinUCB_regret = regret_vs_horizon(REPEAT, HORIZON)


    ### Plot Results ###
    plt.plot(LinUCB_regret)
    plt.xlabel("Horizon")
    plt.ylabel("Cumulative Regret")
    plt.title("LinUCB: Regret vs Horizeon")
    plt.show()

    horizon = np.linspace(1, HORIZON, num=HORIZON)

    plt.semilogx(horizon[1000:], LinUCB_regret[1000:])
    plt.xlabel('Horizon')
    plt.ylabel('Cumulative Regret')
    plt.title('LinUCB: Regret vs Horizeon (Semilogx)')
    plt.show()

    