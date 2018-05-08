from rl.agents import SARSAAgent, DQNAgent

class DQNAgentInitializer:
	def __init__(self, model,memory, policy, action):
		
		self.model = model
		self.memory = memory
		self.policy = policy
		self.action = action
		
	def get_agent(self):
		agent = DQNAgent(
			model = self.model, 
			policy = self.policy, 
			nb_steps_warmup= 10,
			target_model_update= 1e-2,
			nb_actions = self.action, 
			memory = self.memory,
			enable_double_dnq = False
		)


		return agent


