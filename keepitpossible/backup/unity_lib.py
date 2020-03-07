# coding=utf-8
# Copyright 2018 The Dopamine Authors.
# Modifications copyright 2019 Unity Technologies.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Obstacle Tower-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal preprocessing, which
is in charge of:
  . Converting observations to greyscale.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from obstacle_tower_env import ObstacleTowerEnv

import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

import gin.tf
import cv2

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.


@gin.configurable
def create_otc_environment(environment_path=None):
    """Wraps an Obstacle Tower Gym environment with some basic preprocessing.

    Returns:
      An Obstacle Tower environment with some standard preprocessing.
    """
    assert environment_path is not None
    env = ObstacleTowerEnv(environment_path, 0, retro=False)
    env = OTCPreprocessing(env)
    return env


def nature_dqn_network(num_actions, network_type, state):
    """The convolutional network used to compute the agent's Q-values.

    Args:
      num_actions: int, number of actions.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(net, 32, [8, 8], stride=4)
    net = slim.conv2d(net, 64, [4, 4], stride=2)
    net = slim.conv2d(net, 64, [3, 3], stride=1)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512)
    q_values = slim.fully_connected(net, num_actions, activation_fn=None)
    return network_type(q_values)


def rainbow_network(num_actions, num_atoms, support, network_type, state):
    """The convolutional network used to compute agent's Q-value distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(
        net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    net = slim.flatten(net)
    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    net = slim.fully_connected(
        net,
        num_actions * num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    logits = tf.reshape(net, [-1, num_actions, num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(support * probabilities, axis=2)
    return network_type(q_values, logits, probabilities)


def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
    """The Implicit Quantile ConvNet.

    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    state_net = tf.cast(state, tf.float32)
    state_net = tf.div(state_net, 255.)
    state_net = slim.conv2d(
        state_net, 32, [8, 8], stride=4,
        weights_initializer=weights_initializer)
    state_net = slim.conv2d(
        state_net, 64, [4, 4], stride=2,
        weights_initializer=weights_initializer)
    state_net = slim.conv2d(
        state_net, 64, [3, 3], stride=1,
        weights_initializer=weights_initializer)
    state_net = slim.flatten(state_net)
    state_net_size = state_net.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

    batch_size = state_net.get_shape().as_list()[0]
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random_uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

    quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    quantile_net = slim.fully_connected(
        quantile_net,
        state_net_size,
        weights_initializer=weights_initializer)
    # Hadamard product.
    net = tf.multiply(state_net_tiled, quantile_net)

    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    quantile_values = slim.fully_connected(
        net,
        num_actions,
        activation_fn=None,
        weights_initializer=weights_initializer)

    return network_type(quantile_values=quantile_values, quantiles=quantiles)


@gin.configurable
class OTCPreprocessing(object):
    """A class implementing image preprocessing for OTC agents.

    Specifically, this converts observations to greyscale. It doesn't
    do anything else to the environment.
    """

    def __init__(self, environment):
        """Constructor for an Obstacle Tower preprocessor.

        Args:
          environment: Gym environment whose observations are preprocessed.

        """
        self.environment = environment
        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

        self.stage_reward = 0.0
        self.previous_stage_time_remaining = 3000
        self.previous_reward = 0
        self.previous_keys = 0
        self.previous_time_remaining = 3000
        self.tableAction = self.createActionTable()

    def createActionTable(self):
        tableAction = []
        for a in range(0, 3):
            for b in range(0, 3):
                for c in range(0, 2):
                    tableAction.append([a, b, c, 0])
        # print("Action option: ", tableAction[6:12])
        return tableAction

    @property
    def observation_space(self):
        return self.environment.observation_space

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def reset(self):
        """Resets the environment. Converts the observation to greyscale,
        if it is not.

        Returns:
          observation: numpy array, the initial observation emitted by the
            environment.
        """
        observation = self.environment.reset()
        observation = observation[0]
        self.stage_reward = 0.0
        self.previous_stage_time_remaining = 3000
        self.previous_reward = 0
        self.previous_keys = 0
        self.previous_time_remaining = 3000

        self.previous_stage_time_remaining = 3000
        if(len(observation.shape) > 2):
            observation = cv2.cvtColor(cv2.convertScaleAbs(observation, alpha=(255.0 / 1.0)), cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84))

        return observation

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
          mode: Mode argument for the environment's render() method.
            Valid values (str) are:
              'rgb_array': returns the raw ALE image.
              'human': renders to display via the Gym renderer.

        Returns:
          if mode='rgb_array': numpy array, the most recent screen.
          if mode='human': bool, whether the rendering was successful.
        """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment. Converts the observation to
        greyscale, if it is not.

        Remarks:

          * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
          * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
          action: The action to be executed.

        Returns:
          observation: numpy array, the observation following the action.
          reward: float, the reward following the action.
          is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
          info: Gym API's info data structure.
        """
        observation, reward, game_over, info = self.environment.step(np.array(self.tableAction[int(action)-1]))
        observation, keys, time_remaining = observation
        self.stage_reward, previous_stage_time_remaining = self.reward_compute(done=game_over,
                                                                               reward_total=self.stage_reward,
                                                                               keys=keys,
                                                                               previous_keys=self.previous_keys,
                                                                               reward=reward,
                                                                               previous_reward=self.previous_reward,
                                                                               time_remaining=time_remaining,
                                                                               previous_time_remaining=self.previous_time_remaining,
                                                                               previous_stage_time_remaining=self.previous_stage_time_remaining)
        self.previous_reward = reward
        self.previous_keys = keys
        self.previous_time_remaining = time_remaining
        self.game_over = game_over
        if(len(observation.shape) > 2):
            observation = cv2.cvtColor(cv2.convertScaleAbs(observation, alpha=(255.0 / 1.0)), cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84))
        return observation, self.stage_reward, game_over, info

    def reward_compute(
            self,
            done,
            reward_total,
            keys,
            previous_keys,
            reward,
            previous_reward,
            time_remaining,
            previous_time_remaining,
            previous_stage_time_remaining):
        # 定義獎勵公式
        # reward 是從環境傳來的破關數
        # keys 是撿到鑰匙的數量
        # time_remaining 是剩餘時間
        # 過關最大獎勵為10
        # 一把鑰匙為5
        # 時間果實暫時只給0.5，因為結束會結算剩餘時間，會有獎勵累加的問題。
        # 如果過關，給予十倍過關獎勵 - (場景開始的時間-剩餘時間)/1000
        # print("time_remaining ", time_remaining,
        #       " previous_time_remaining ", previous_time_remaining,
        #         " reward ", reward)
        if reward < 0.2:
            reward = 0
        if (reward - previous_reward) > 0.8:
            # ***如果剩餘時間比場景時間多會變成加分獎勵，可能會極大增加Agent吃時間果實的機率。
            # ***另一種方式是剩餘的時間直接/1000加上去，這樣就沒有累加效果。
            print("Pass ", reward, " Stage!")
            reward_total += (reward - previous_reward) * 100 - \
                            (previous_stage_time_remaining - time_remaining)
            # 過關之後把時間留到下一關，儲存這回合時間供下次計算過關使用
            previous_time_remaining = time_remaining
            previous_stage_time_remaining = time_remaining

        # 假設過關的時候有順便吃到果實或鑰匙，所以預設為同時可以加成
        if previous_keys > keys:
            print("Get Key")
            reward_total += 5

        if previous_time_remaining < time_remaining and previous_time_remaining != 0:
            print("Get time power up")
            reward_total += 0.5
        else:
            reward_total -= 0.1
        if done and previous_time_remaining > 100:
            print("Agent died")
            # 如果剩餘時間越多就掛點，扣更多
            reward_total -= (10 + time_remaining / 100)
        return reward_total, previous_stage_time_remaining
