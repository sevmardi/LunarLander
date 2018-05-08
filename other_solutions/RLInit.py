from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

class RLInit:
	def __init__(self, memory, eps, window_size=1):
		super(RLInit, self).__init__()
		self.memory = memory
		self.eps = eps
		self.window_size = window_size
		
	def get_eps_policy_and_memory(self):
		return (
			SequentialMemory(limit = self.memory, window_length = self.window_size),
			EpsGreedyQPolicy(eps = self.eps)

		)

	