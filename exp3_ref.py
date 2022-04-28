import random, math
import matplotlib.pyplot as plt
import numpy as np


def sample_from_distribution(mu):
    val = random.random()
    # print val,
    if val <= mu:
        return 1
    else:
        return 0


# print "mu1:",
mu0 = .8
mu1 = .7
mu2 = .5

# total_time = int(2e3)
# N = 5
total_time = 1000
N = 1
eta = np.sqrt(np.log(3) / (total_time * 3))

cum_loss = np.zeros(3)
# cum_loss = np.zeros(3)
exp3_regret = np.zeros((total_time, N))
exp3_num_op_arms = np.zeros((total_time, N))
iterations = [0] * total_time
best_mu = max(mu0, mu1, mu2)

# rew_avg = np.asarray([0.8, 0.7, 0.5])
# num_iter, num_trial = int(2e3), 20
# eta = np.sqrt(np.log(rew_avg.size) / (num_iter * rew_avg.size))
# var = 0.01


for it in range(N):
    total_reward = 0
    total_best_reward = 0
    cum_loss.fill(0.0)
    p = [1 / 3, 1 / 3, 1 / 3]
    p = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype="float64")
    # p.astype(float64)
    print("iter:", it)
    print("p:", p)
    print("cum_loss:", cum_loss)

    for t in range(total_time):
        # eta = math.sqrt(2*math.log(2) / (total_time*2))
        # print np.exp(eta * cum_loss)

        # Calculate sampling distribution
        p = np.exp(eta * cum_loss) / np.sum(np.exp(eta * cum_loss))
        # print("p:", p)

        # Sample an arm from p
        # arm_num = sample_from_distribution(p[0])
        arm_num = np.random.choice(3, 1, p=p)
        # print("arm_num:", arm_num)

        if arm_num == 0:
            new_sample = sample_from_distribution(mu0)
            cum_loss[1] += (new_sample) / p[1]
            cum_loss[2] += (new_sample) / p[2]

            # print arm_num

            if t == 0:
                exp3_num_op_arms[t][it] = 1
            else:
                exp3_num_op_arms[t][it] = exp3_num_op_arms[t - 1][it] + 1

            total_best_reward += new_sample

        if arm_num == 1:
            new_sample = sample_from_distribution(mu1)
            cum_loss[0] += (new_sample) / p[0]
            cum_loss[2] += (new_sample) / p[2]

            # print arm_num

            if t == 0:
                exp3_num_op_arms[t][it] = 1
            else:
                exp3_num_op_arms[t][it] = exp3_num_op_arms[t - 1][it] + 1

            total_best_reward += new_sample

        else:
            new_sample = sample_from_distribution(mu2)
            cum_loss[0] += new_sample / p[0]
            cum_loss[1] += new_sample / p[1]

            if t == 0:
                exp3_num_op_arms[t][it] = 0
            else:
                exp3_num_op_arms[t][it] = exp3_num_op_arms[t - 1][it]

            total_best_reward += new_sample
            # total_best_reward += sample_from_distribution(mu1)

        total_reward += new_sample
        exp3_regret[t][it] = total_best_reward - total_reward
        iterations[t] = t
        if t % 100 == 0:
            print(t, exp3_num_op_arms[t][it], exp3_regret[t][it], p)

avg_exp3_regret = [0] * total_time
std_regret = [0] * total_time

for i in range(total_time):
    # print i, exp3_regret[i]
    avg_exp3_regret[i] = np.average(exp3_regret[i])
    std_regret[i] = np.std(exp3_regret[i] / 10)
    if i % 100 != 0:
        std_regret[i] = 0
    # if i % 1000 == 0:
    #     print(avg_exp3_regret[i], i, type(avg_exp3_regret), len(avg_exp3_regret))

exp3_avg_op_arms = [0] * total_time
std_op_arms = [0] * total_time
for i in range(total_time):
    exp3_avg_op_arms[i] = np.average(exp3_num_op_arms[i]) / (i + 1)
    std_op_arms[i] = np.std(exp3_num_op_arms[i]) / (10 * (i + 1))
    if i % 100 != 0:
        std_op_arms[i] = 0
    # if i % 1000 == 0:
    #     print(exp3_avg_op_arms[i], i)

plt.errorbar(iterations, avg_exp3_regret, yerr=std_regret, label='EXP3')
plt.xlabel('Iterations')
plt.ylim([0, 350])
plt.ylabel('Regret')
plt.legend()
plt.show()

plt.errorbar(iterations, exp3_avg_op_arms, yerr=std_op_arms, label='EXP3')
plt.xlabel('Iterations')
plt.ylim([0.5, 1])
plt.ylabel('Rate of Optimal Arms')
plt.legend()
plt.show()