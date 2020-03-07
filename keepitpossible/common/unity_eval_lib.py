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

from keepitpossible.common import action_table
from keepitpossible.common import reward_function
from keepitpossible.common import image_processed


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
        self.reward = 0.
        self.stage_reward = 0.
        self.previous_stage_time_remaining = 3000
        self.time_remaining = 3000
        self.previous_reward = 0
        self.previous_keys = 0
        self.stage_clear = 0
        self.previous_time_remaining = 3000
        self.table_action = action_table.create_rainbow_action_table()

    @property
    def is_grading(self):
        return self.environment.is_grading

    @property
    def done_grading(self):
        return self.environment.done_grading

    @property
    def observation_space(self):
        return self.environment.observation_space

    @property
    def action_space(self):
        self.environment.action_space.n = 10
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
        self.stage_reward = 0.0
        self.previous_stage_time_remaining = 3000
        self.previous_reward = 0
        self.previous_keys = 0
        self.previous_time_remaining = 3000
        self.keys = 0
        self.reward = 0.
        self.stage_reward = 0.
        self.time_remaining = 3000
        self.stage_clear = 0
        self.previous_stage_time_remaining = 3000

        processed_observation = image_processed.gray_progress_bar(
            image=observation[0],
            stage_clear=self.stage_clear,
            time_remaining=self.time_remaining,
            keys=self.keys)
        return processed_observation

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
        # 儲存上一個動作狀態，供計算獎勵用
        self.previous_keys = self.keys
        self.previous_reward = self.reward
        self.previous_time_remaining = self.time_remaining
        # 做出動作，獲得場景資訊,已過關數,代理資訊
        observation, self.reward, self.game_over, info = self.environment.step(
            self.table_action[int(action)])
        # 預處理模型需要的資料
        observation, self.keys, self.time_remaining = observation
        self.stage_reward, self.previous_stage_time_remaining, self.game_over, self.stage_clear = reward_function.compute(done=self.game_over,
                                                                                                                          stage_clear=self.stage_clear,
                                                                                                                          reward_total=self.stage_reward,
                                                                                                                          keys=self.keys,
                                                                                                                          previous_keys=self.previous_keys,
                                                                                                                          reward=self.reward,
                                                                                                                          previous_reward=self.previous_reward,
                                                                                                                          time_remaining=self.time_remaining,
                                                                                                                          previous_time_remaining=self.previous_time_remaining,
                                                                                                                          previous_stage_time_remaining=self.previous_stage_time_remaining)
        self.stage_reward -= 1
        self.stage_reward /= 300
        processed_observation = image_processed.gray_progress_bar(
            image=observation,
            stage_clear=self.stage_clear,
            time_remaining=self.time_remaining,
            keys=self.keys)
        # Eval just return self.reward
        return processed_observation, self.stage_reward, self.game_over, info, self.reward
