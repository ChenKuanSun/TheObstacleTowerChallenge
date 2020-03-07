from obstacle_tower_env import ObstacleTowerEnv
from keepitpossible.common import reward_function
from keepitpossible.common import action_table
from keepitpossible.common import image_processed
import tensorflow as tf
import numpy as np
import cv2


class Worker(object):
    def __init__(
            self,
            GLOBAL_SHARE,
            PARAMETERS,
            SCHEDULE,
            wid,
            retro,
            realtime_mode):
        # 導入模組
        self.GLOBAL_SHARE = GLOBAL_SHARE
        self.PARAMETERS = PARAMETERS
        self.SCHEDULE = SCHEDULE
        self.kprun = self.GLOBAL_SHARE.MODEL
        self.table_action = action_table.create_rainbow_action_table()
        self.wid = wid
        self.env = ObstacleTowerEnv(environment_filename=self.SCHEDULE.ENV_PATH,
                                    worker_id=wid,
                                    retro=retro,
                                    realtime_mode=realtime_mode)
        # 設定關卡
        self.env.floor(self.SCHEDULE.ENV_FLOOR)
        self.step = 0
        self.frame = 8

    def work(self):
        # 只要還沒達到目標回合就LOOP
        while not self.GLOBAL_SHARE.COORD.should_stop():
            # 紀錄步數
            self.step += 1
            # 重設關卡
            self.env.seed(np.random.randint(self.SCHEDULE.ENV_SEED))
            obs = self.env.reset()
            # 初始化
            done = False
            stage_reward = 0.
            stage_clear = 0
            reward = 0
            keys = 0
            buffer_temp = []
            # 檢查是否有吃到加時間的，如果是第一回合出來沒有time_remaining，事先定義
            time_remaining = 3000
            previous_stage_time_remaining = time_remaining
            # 預處理圖像
            observation = obs[0]
            previous_preprocessed_observation_image = image_processed.rgb_progress_bar(image=observation,
                                                                                      stage_clear=stage_clear,
                                                                                      time_remaining=time_remaining,
                                                                                      keys=keys)
            buffer_s, buffer_a, buffer_r = [], [], []
            # 只要沒死
            while not done:
                # 如果模型正在更新就等待更新完成
                if not self.GLOBAL_SHARE.ROLLING_EVENT.is_set():
                    # 等待更新完成
                    self.GLOBAL_SHARE.ROLLING_EVENT.wait()
                    # 清除記憶體，使用新的代理收集資料
                    buffer_s, buffer_a, buffer_r = [], [], []
                # 根據上一次的狀態決定動作
                if buffer_temp :
                    action = self.kprun.choose_action(np.vstack(buffer_temp))
                else:
                    action = self.kprun.choose_action(previous_preprocessed_observation_image)
                action = np.clip(np.random.normal(action, 2.), *[0,6])
                # 儲存上一個動作狀態，供計算獎勵用
                previous_keys = keys
                previous_reward = reward
                previous_time_remaining = time_remaining
                # 做出動作，獲得場景資訊,已過關數,代理資訊
                observation, reward, done, info = self.env.step(self.table_action[int(action)])
                # 預處理模型需要的資料
                observation, keys, time_remaining = observation
                stage_reward, previous_stage_time_remaining, done, stage_clear = reward_function.compute(done=done,
                                                                                                         stage_clear=stage_clear,
                                                                                                         reward_total=stage_reward,
                                                                                                         keys=keys,
                                                                                                         previous_keys=previous_keys,
                                                                                                         reward=reward,
                                                                                                         previous_reward=previous_reward,
                                                                                                         time_remaining=time_remaining,
                                                                                                         previous_time_remaining=previous_time_remaining,
                                                                                                         previous_stage_time_remaining=previous_stage_time_remaining)
                # 做一次動作就扣0.01 減少
                stage_reward -= 0.1
                # Normalize reward
                stage_reward /= 300
                # 儲存這次影像供下次使用
                preprocessed_observation_image = image_processed.rgb_progress_bar(image=observation,
                                                                                   stage_clear=stage_clear,
                                                                                   time_remaining=time_remaining,
                                                                                   keys=keys)

                # 把這次狀態存入 記憶體
                buffer_s.append(
                    np.array([preprocessed_observation_image]).reshape(-1))
                buffer_temp.append(np.array([preprocessed_observation_image]).reshape(-1))
                if len(buffer_temp) > self.frame:
                    buffer_temp.pop(0)
                buffer_a.append(action)
                buffer_r.append(stage_reward)
                # 儲存下一步要參考的圖像
                previous_preprocessed_observation_image = preprocessed_observation_image
                # Buffer滿出來的時候先丟到共享記憶體。
                self.GLOBAL_SHARE.UPDATE_COUNTER += 1
                if len(buffer_s) >= self.SCHEDULE.EP_LEN - \
                        1 or self.GLOBAL_SHARE.UPDATE_COUNTER >= self.PARAMETERS.MIN_BATCH_SIZE:
                    v_s_ = self.kprun.get_v(preprocessed_observation_image)
                    # 計算折扣獎勵
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.PARAMETERS.GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    # 整理維度，堆疊放入共享資源區。
                    bs, ba, br = np.vstack(buffer_s), np.vstack(
                        buffer_a), np.array(discounted_r)[:, np.newaxis]
                    self.GLOBAL_SHARE.QUEUE.put(np.hstack((bs, ba, br)))
                    # 清空
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 如果整個模型步數到達最小BATCH 就整個更新
                    if self.GLOBAL_SHARE.UPDATE_COUNTER >= self.PARAMETERS.MIN_BATCH_SIZE:
                        # 停止收集資料
                        self.GLOBAL_SHARE.ROLLING_EVENT.clear()
                        # 更新PPO
                        self.GLOBAL_SHARE.UPDATE_EVENT.set()
                    # 達到最多EP停止訓練
                    if self.GLOBAL_SHARE.EP >= self.SCHEDULE.EP_MAX:
                        self.GLOBAL_SHARE.COORD.request_stop()
                        break
                # 限制關卡重新訓練，加強收斂
                if stage_clear > self.SCHEDULE.LESSON_END:
                    done = True
            # 可視化數據
            # 平均獎勵
            self.GLOBAL_SHARE.REWARD_AVERAGE.append(stage_reward)
            reward_average = sum(self.GLOBAL_SHARE.REWARD_AVERAGE) / \
                float(len(self.GLOBAL_SHARE.REWARD_AVERAGE))
            self.summary = tf.Summary(value=[tf.Summary.Value(tag="Reward Avarage", simple_value=reward_average)])
            self.kprun.train_writer.add_summary(
                self.summary, self.GLOBAL_SHARE.EP)
            # 個別獎勵
            self.summary = tf.Summary(value=[tf.Summary.Value(tag="Stage Reward/Worker %d" %self.wid, simple_value=stage_reward)])
            self.kprun.train_writer.add_summary(self.summary, self.step)
            # 印出這回合的獎勵
            print(
                '{0:.1f}%'.format(
                    self.GLOBAL_SHARE.EP /
                    self.SCHEDULE.EP_MAX *
                    100),
                '|W%i' %
                self.wid,
                '|Ep_r: %.2f' %
                stage_reward, "Pass %d / %d Stage!" %(stage_clear, self.SCHEDULE.LESSON_END+1)
            )
            self.GLOBAL_SHARE.EP += 1
            # 儲存檢查點
            if self.GLOBAL_SHARE.EP % self.SCHEDULE.CHECKPOINT_STEP == 0:
                self.kprun.save()
        self.env.close()
