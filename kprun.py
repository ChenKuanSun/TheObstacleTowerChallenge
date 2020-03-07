from obstacle_tower_env import ObstacleTowerEnv
# 參數設定檔
# from keepitpossible.common import gcp_parameters
from keepitpossible.common import pc_parameters
# RL模型
from keepitpossible.model import dppo_model
from keepitpossible.model import ex_dppo_model
# 多工
from keepitpossible import dppo_worker
import numpy as np
import tensorflow as tf
import os
import threading
import queue

# 使用第一張GPU 卡 1080
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TF_CPP_MIN_LOG_LEVEL, 1隱藏通知, 2隱藏通知和警告, 3隱藏通知、警告和錯誤
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class KPRUN(object):
    def __init__(self):
        # SCHEDUL參數
        #     N_WORKER：設置幾個代理 預設：4
        #     EP_LEN：代理人自己更新的步數 預設：500
        #     EP_MAX：最大訓練回合數(每個代理人加起來的回合) 預設：4000
        #     ENV_SEED：隨機種子 預設：np.random.randint(5)
        #     ENV_FLOOR：起始樓層 預設：0
        #     LESSON_END：課程完成終止樓層 預設：5
        self.SCHEDULE = pc_parameters.SCHEDULE(N_WORKER=4,
                                               EP_MAX=40000,
                                               LESSON_END=0)
        self.PARAMETERS = pc_parameters.PARAMETERS()
        self.MODEL = ex_dppo_model.DPPOMODEL(GLOBAL_SHARE=self,
                                          PARAMETERS=self.PARAMETERS,
                                          SCHEDULE=self.SCHEDULE)
        # 多工共享資源
        self.UPDATE_COUNTER = 0
        self.REWARD_AVERAGE = []
        self.EP = 0
        self.QUEUE = queue.Queue()
        self.COORD = tf.train.Coordinator()
        # 建立多工事件
        self.UPDATE_EVENT = threading.Event()
        self.ROLLING_EVENT = threading.Event()

    def train(self):
        self.UPDATE_EVENT.clear()
        self.ROLLING_EVENT.set()
        self.MODEL.load()
        # 建立多工物件
        workers = [dppo_worker.Worker(GLOBAL_SHARE=self,
                                      PARAMETERS=self.PARAMETERS,
                                      SCHEDULE=self.SCHEDULE,
                                      wid=i,
                                      retro=False,
                                      realtime_mode=False,
                                      ) for i in range(self.SCHEDULE.N_WORKER)]
        # 執行多工任務
        threads = []
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()  # training
            threads.append(t)
        # 建立模型更新的執行緒
        threads.append(threading.Thread(target=self.MODEL.update, ))
        threads[-1].start()
        self.COORD.join(threads)
        # 儲存模型
        self.MODEL.save()
        # 測試
        self.testing()

    def grading(self, env):
        done = False
        reward = 0.0
        env.seed(np.random.randint(100))
        obs = env.reset()
        previous_preprocessed_observation_image = obs[0]
        while not done:
            action = env.action_space.sample(
                previous_preprocessed_observation_image)
            observation, reward, done, info = env.step(action)
            # 預處理模型需要的資料
            observation_image, keys, time_remaining = observation
            previous_preprocessed_observation_image = observation_image
        env.reset()
        return reward

    def testing(self):
        from keepitpossible.common import action_table
        self.table_action = action_table.create_action_table()
        self.MODEL.load()
        done = False
        reward = 0.0
        env = ObstacleTowerEnv(environment_filename=self.SCHEDULE.ENV_PATH,
                               worker_id=self.SCHEDULE.N_WORKER + 1,
                               retro=False,
                               realtime_mode=True)
        obs = env.reset()
        previous_preprocessed_observation_image = obs[0]
        while not done:
            action = self.MODEL.choose_action(
                previous_preprocessed_observation_image)
            # 做出動作，獲得場景資訊,已過關數,代理資訊
            for _ in self.table_action[int(action)]:
                observation, reward, done, info = env.step(_)
                print(
                    "Action_Chose: ",
                    action,
                    "Action: ",
                    _,
                    " Reward: ",
                    reward)
                if done:
                    break
            # 預處理模型需要的資料
            observation_image, keys, time_remaining = observation
            preprocessed_observation_image = observation_image
            previous_preprocessed_observation_image = preprocessed_observation_image
        env.close()


if __name__ == '__main__':
    kp_run = KPRUN()
    kp_run.testing()
