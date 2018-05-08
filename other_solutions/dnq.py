import numpy as np
import pandas as pd 
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)


class DeepDNQNetwork:
	def __init__(self, arg):
		super(DeepDNQNetwork, self).__init__()
		self.arg = arg

	