""" Trains an Actor with CACLA learning through a Critic for the OpenAI Gym Lunar Lander """
# ----------------------------------------------
# 2 neural networks are used, 200 neurons for the single hidden layer 
# Actor  is Psi parameters, one NN for each action. 4 actions = 0,1, 2, 3. 
# Sigmoid activation function with RMS
# Critic is Theta parameters, one NN. 
# ReLu and Sigmoid Functions with RMS as Adaptive Learning Rate 
# ----------------------------------------------

# https://github.com/AdriannaGmz/LunarLander/blob/master/lunar_cacla.py

import numpy as np
import picke as pickle 
import gym 

# hyperparameters
H = 200         # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4    # learning_rate of actor
beta = 1e-2     # learning_rate of critic
gamma = 0.99    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization, Actor and Critic
D = 8 # observation space
A = 4 # action space

modelA['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA['Psi2'] = np.random.randn(A,H) / np.sqrt(H)

modelC = {}
modelC['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
modelC['Theta2'] = np.random.randn(H) / np.sqrt(H)

gradA_buffer = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 
gradC_buffer = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

rmspropA_cache = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 
rmspropC_cache = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
err_probs = np.zeros(A);

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def gaussian_sample(mean, std):
	gauss_sample = np.random.normal(mean, std, None)

	return gauss_sample


def epislon_greedy_exploration(best_action, episode_number):
	epsilon = epsilon_scale/(1+ 0.001 * episode_number)
	prob_vector = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]
	prob_vector[best_action] = epsilon/4 + 1 - epsilon
	action_to_explore = prob_vector[best_action] = epsilon/4 + 1 - epsilon

	return action_to_explore

def sample_from_action_probs(action_prob_values):
  cumsum_action = np.cumsum(action_prob_values)
  sum_action = np.sum(action_prob_values)
  #sample_action = np.random.choice(4, 1, True, action_prob_values)
  sample_action = int(np.searchsorted(cumsum_action,np.random.rand(1)*sum_action))
  return sample_action

def take_random_action():
  sampled_action = np.random.randint(4)
  return sampled_action


def actor_forward(x):
  hA = np.dot(modelA['Psi1'], x)
  hA[hA<0] = 0    # ReLU nonlinearity
  logp = np.dot(modelA['Psi2'], hA)  
  p = sigmoid(logp)
  return p, hA  

def critic_forward(x):
  hC = np.dot(modelC['Theta1'], x)
  hC[hC<0] = 0      
  logv = np.dot(modelC['Theta2'], hC)
  v = sigmoid(logv)
  return v, hC 	



