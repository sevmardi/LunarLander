import numpy as np


class BaseAgent(object):
    """ Basic agent with some basic functions implemented, such as
    and epsilon-greedy action selection.

    self._wrapper   A subclass of the Wrapper class, which translates the
                    environment to an interface for generic Reinforcement
                    Learning Agents
    self._total_reward  Total reward for one training episode

    Also has some basic algorithm parameters:
    self._epsilon   Value in [0, 1] for creating randomness in the greedy method
    self._alpha     Step size parameter

    """

    def __init__(self, wrapper, epsilon=0.1, alpha=0.5, seed=42):
        self._wrapper = wrapper  # environment wrapper that provides extra info
        self._epsilon = epsilon  # randomness
        self._alpha = alpha      # step size parameter
        self._total_reward = 0
        np.random.seed(seed)

    def initialise_episode(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    def select_action(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    def epsilon_greedy(self, q):
        """Select action in an epsilon-greedy fashion, based on action values q.
        Return corresponding action id

        :param q:   Array of length |actions|, containing the action values
        """
        # Select an action greedily
        if np.random.random_sample() > self._epsilon:
            # which actions have the highest expectation?
            max_exp = max(q)
            max_exp_action_idx = [i for i in range(len(q))
                                  if q[i] == max_exp]
            if not max_exp_action_idx:
                print(q)
            return int(np.random.choice(max_exp_action_idx, 1)[0])
        # Or select an action randomly
        return np.random.choice(len(q))

    def learn(self):
        # raise NotImplementedError("Subclass must implement abstract method")
        return 0.0


class DNQAgent(object):
    """This agent uses DQN for making action decisions with 1-epsilon probability"""

    def __init__(self, name, state_dim, action_dim, epsdecay=0.995,
                 buffersize=500000, samplesize=32, minsamples=10000,
                 gamma=0.99, state_norm_file='../params/state-stats.pkl', update_target_freq=600,
                 nnparams={  # Basic DQN setting
                     'hidden_layers': [(40, 'relu'), (40, 'relu')],
                     'loss': 'mse',
                     'optimizer': Adam(lr=0.00025),
                     'target_network': False}):
        """Accentps a unique agent name, number of variables in the state, number of actions and parameters of DQN then initialize the agent"""
        # Unique name for the agent
        self.name = name
        # no:of state and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Create buffer for experience replay
        self.memory = Memory(maxsize=buffersize)
        # Set initial epsilon to 1.0
        self.eps = 1.0
        # Minimum number of samples in the buffer to start learning
        self.minsamples = minsamples
        # Number of random samples to be drawn from the buffer for experience
        # replay
        self.samplesize = samplesize
        # Decay factor for epsilon for each episode
        self.epsdecay = epsdecay
        # Discount factor for Q learning
        self.gamma = gamma
        # Dictionary of DQN parameters
        self.nnparams = nnparams
        # Create the base predictor neural network
        # and if required the target neural network too.
        self._create_nns_()
        # Load the state variable normalizers from pickle file if exists
        self._load_state_normalizer_(state_norm_file)
        # Update frequency of the target network in number of steps
        self.update_target_freq = update_target_freq
        # Boolean flag indicating whether the agent started learning or not
        self.started_learning = False
        # Keeps a count of number of steps.
        self.steps = 0

        def _load_state_normalizer_(self, state_norm_file):
            self.mean = np.zeros(self.state_dim)
            self.std = np.ones(self.state_dim)

            return None


class MyAgent(BaseAgent):
    """ TODO: add description for this class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: implement the rest of your initialisation

    def initialise_episode(self):
        # TODO: implement your own method
        raise NotImplementedError("function not yet implemented")
        # pass

    def select_action(self, *args):
        # TODO: implement your own function
        raise NotImplementedError("function not yet implemented")
        # pass

    def train(self):
        # TODO: implement your own function
        raise NotImplementedError("function not yet implemented")
        # return reward

    # TODO: implement all other functions and methods needed for your agent
