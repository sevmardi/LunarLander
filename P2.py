from gym import wrappers
import time
import csv
from collections import deque
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
import numpy as np

average_rewards = deque([], 100)
api_key = "sk_lxRBtl95SM2DXnZfBBigg"
tdir = "./Project2-LunarLander-GA2"

def single_iteration(env, weights, done = False):

    global average_rewards
    state = env.reset()
    reward_sum = 0

    while done == False:
        action_probabilities = np.matmul(weights.T, state)
        best_action = np.argmax(action_probabilities)
        state, reward, done, _ = env.step(best_action)
        reward_sum += reward

    average_rewards.append(reward_sum)

    return reward_sum


def run_algo(max_gen = 250, UPLOAD = False, alpha = 0.0003, pop_size = 50, \
             sigma_start=0.3, test_weights = False):

    start_time = time.time()
    random_seed = 9
    sigma_end = 0.1
    sigma_dec = (sigma_start -sigma_end) / max_gen
    reward_goal = 200
    total_rewards = []
    np.random.seed(random_seed)

    env = gym.make('LunarLander-v2')
    env.reset()
    if UPLOAD:
        env = wrappers.Monitor(env, tdir, force=True, video_callable=False)

    env.seed(random_seed)
    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    weights = np.ones((state_size, action_size)) #TODO ZERO TO ONE
    sigma = sigma_start
    fixed_weights = False


    for gen in range(max_gen):

        sigma -= sigma_dec
        reward_matrix = []
        noise_matrix = np.random.randn(pop_size, state_size, action_size)

        for j in range(pop_size):
            temp_weights = weights + sigma * noise_matrix[j]
            reward_matrix.append(single_iteration(env, temp_weights))


        if np.mean(average_rewards) >= reward_goal:
            fixed_weights = True
            print("SUCCESSFUL WEIGHTS FOUND: AVERAGE REWARD {}".format(np.mean(average_rewards)))
            if not test_weights:
                break

        #Update Weights
        if not fixed_weights:
            weighted_weights = np.matmul(noise_matrix.T, reward_matrix).T
            new_weights = weights + alpha / (pop_size * sigma) * weighted_weights
            weights = new_weights

        gen_mean = np.mean(reward_matrix)
        total_rewards.append(gen_mean)

        print("Generation {}, Population Mean: {}, sigma: {}".format(gen, gen_mean, sigma))

    env.close()

    total_time = time.time() - start_time
    print("Runtime is :{}".format(total_time))

    if UPLOAD:
        gym.upload(tdir, api_key=api_key)

    np.save('weights.npy', weights)

    return total_rewards, weights

def test_alphas(alpha_iter):

    with open('reward_vs_alpha.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for alpha in alpha_iter:
            total_rewards, _ = run_algo(max_gen=500, UPLOAD=False, alpha=alpha, test_weights=False)
            filewriter.writerow(total_rewards)

def test_pop_size(pop_size_iter):

    with open('reward_vs_pop_size.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for pop_size in pop_size_iter:
            total_rewards, _ = run_algo(max_gen=400, UPLOAD=False, pop_size=pop_size, test_weights=False)
            filewriter.writerow(total_rewards)

def test_sigma_start(sigma_start_iter):

    with open('reward_vs_sigma_start.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for sigma_start in sigma_start_iter:
            total_rewards, _ = run_algo(max_gen=400, UPLOAD=False, sigma_start=sigma_start, test_weights=False)
            filewriter.writerow(total_rewards)



def test_weights(weights, episodes = 1000, UPLOAD= False, graph = False):

    start_time = time.time()
    random_seed = 9
    total_rewards = []

    UPLOAD_GENERATION_INTERVAL = 10  # Generate a video at this interval

    env = gym.make('LunarLander-v2')
    env.reset()

    if UPLOAD:
        env = wrappers.Monitor(env, tdir, force=True, video_callable=False)

    for i in range(episodes):
        total_rewards.append(single_iteration(env, weights))

    env.close()

    if UPLOAD:
        gym.upload(tdir, api_key=api_key)

    return total_rewards


def graph_rewards_vs_alphas(file, iter):

    df = pd.read_csv(file, sep=',', header=None)
    df = df.T
    df.columns = iter
    df = df[[0.0001, 0.0002, 0.0003, 0.0004]]
    graph_plot = df.plot(linewidth=0.75)
    graph_plot.set_xlabel('Generations')
    graph_plot.set_ylabel('Reward')
    # plt.show()
    plt.save()



if __name__ == "__main__":

    # #Find the rewards for the best alpha value
    alpha_iter = [0.0001, 0.0002, 0.0003, 0.0004, 0.001, 0.002, 0.003, 0.004]
    test_alphas(alpha_iter)

    #Find the rewards for the best population size
    pop_size_iter = [25,50,75,100]
    test_pop_size(pop_size_iter)

    # Find the rewards for the best sigma start size
    sigma_start_iter = [0.1, 0.2, 0.3, 0.4]
    test_sigma_start(sigma_start_iter)

    # Find the best weights
    _, best_weights = run_algo(max_gen=250, UPLOAD=True, alpha=0.0003, test_weights=False)
    np.save('test_weights123.npy', best_weights)

    # Test the best weights
    returns = test_weights(episodes = 100, weights = np.load('test_weights.npy'), UPLOAD = True)
    np.save('optimal_weights_returns.npy', returns)


    # Graph Rewards vs HyperParameter

    # VS ALPHA
    csv_file = 'reward_vs_alpha.csv'
    iter = alpha_iter
    
    #VS POP SIZE
    csv_file = 'reward_vs_pop_size.csv'
    iter = pop_size_iter
    
    # VS SIGMA
    
    csv_file = 'reward_vs_sigma_start.csv'
    iter = sigma_start_iter

    graph_rewards_vs_alphas(file='reward_vs_alpha.csv')