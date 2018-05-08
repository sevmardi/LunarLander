# -*- coding: utf-8 -*-
""" Environment wrapper class for Reinforcement Learning PA - Spring 2018

Details:
    File name:          wrapper.py
    Author:             Anna Latour
    Date created:       19 March 2018
    Date last modified: 19 March 2018
    Python Version:     3.4

Description:
    Implementation of a superclass for environment wrappers. A wrapper allows
    you to model the environment you need to learn by e.g. providing functions
    that aid discretisation.

Related files:
    main.py
    cartpole_wrapper.py
"""

import gym


class Wrapper(object):
    """ Wrapper: A Supper class for an environment helps you to specify how you model the
    environment such that it can interface with a general Reinforcement Learning
    agent."""

    def __init__(self, env_name, actions):
        self._env = gym.make(env_name)
        self._actions = actions
        self._number_of_steps = 0

    def reset(self):
        self._number_of_steps = 0
        return self._env.reset()

    def action_space(self):
        return self._env.action_space

    def observation_space(self):
        return self._env.observation_space

    def step(self, action):
        self._number_of_steps += 1
        return self._env.step(action)

    def close(self):
        self._env.close()

    def actions(self):
        if self._actions is None:
            raise NotImplementedError("Subclass must define actions")
        return self._actions

    def solved(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")


class LunarLanderWrapper(Wrapper):
    """ TODO: Add a description for your wrapper
    """

    # TODO: define list of actions (HINT: check LunarLander-v2 source code to
    # figure out what those actions are)
    _actions = []

    def __init__(self):
        # Don't change environment name
        super().__init__(env_name='LunarLander-v2', actions=self._actions)
        # TODO: write the rest of your initialisation function

    def solved(self, reward):
        return True if reward >= 200 else False

    # TODO: implement all other functions and methods needed for your wrapper
    
    