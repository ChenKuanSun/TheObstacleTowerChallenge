from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import tensorflow as tf
import os
import cv2
# 使用第一張GPU 卡 1080
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TF_CPP_MIN_LOG_LEVEL, 1隱藏通知, 2隱藏通知和警告, 3隱藏通知、警告和錯誤
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

GAMMA = 0.6


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(
            tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=100,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(
            self.normal_dist.sample(1),
            action_bound[0],
            action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(
                self.a)  # loss without advantage
            # advantage (TD_error) guided loss
            self.exp_v = log_prob * self.td_error
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp_v, global_step)  # min(v) = max(-v)

    def learn(self, s, a, td):
        #         s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        #         s = s[np.newaxis, :]
        # get probabilities for all actions
        return self.sess.run(self.action, {self.s: s})


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=100,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            # TD_error = (r+gamma*V_next) - V_eval
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s, s_

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


class MODEL(object):
    def __init__(
            self,
            LR_A=0.01,
            LR_C=0.1,
            ACTION_BOUND=[6, 12],
            VAR_MIN=0.1,
            N_S=84672,
            global_step=0):
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.ACTION_BOUND = ACTION_BOUND
        self.VAR_MIN = VAR_MIN
        self.N_S = N_S
        self.tableAction = self.createActionTable()
        self.global_step = global_step
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag="Stage_reward",
                    simple_value=0)])
        self.train_writer.add_summary(self.summary, self.global_step)
        self.actor = Actor(
            self.sess,
            n_features=N_S,
            lr=LR_A,
            action_bound=ACTION_BOUND)
        self.critic = Critic(self.sess, n_features=N_S, lr=LR_C)
        self.sess.run(tf.global_variables_initializer())
        self.total_step = 0

    def run(self, env, episode=10, env_seed=0, env_floor=0, videoWriter=None):
        self.load()
        episode_reward = []
        # 設定關卡
        env.seed(env_seed)
        env.floor(env_floor)
        # 隨機Action 逼迫Agent往前探索
        randactionprob = 2.
        # 探索率最小值
        randactionprob_MIN = 0.1
        for ep in range(episode):
            self.global_step += 1
            # 重設關卡
            obs = env.reset()
            # 如果要錄影就開錄影功能
            if videoWriter:
                videoWriter.write(cv2.cvtColor(
                    cv2.convertScaleAbs(obs[0], alpha=(255.0 / 1.0)),
                    cv2.COLOR_BGR2RGB))

            # 預處理圖像
            previous_preprocessed_observation_image = np.reshape(obs[0], [
                                                                 1, -1])
            # 初始化
            done = False
            stage_reward = 0.0
            reward = 0
            keys = 0
            # 檢查是否有吃到加時間的，如果是第一回合出來沒有time_remaining，事先定義
            time_remaining = 3000
            previous_stage_time_remaining = time_remaining
            buffer_aa = []
            # 設定走完一個關卡就重新訓練(課程設計)
            while not done:
                # 儲存上一個動作狀態，供計算獎勵用
                previous_keys = keys
                previous_reward = reward
                previous_time_remaining = time_remaining
                # 根據上一次的狀態決定動作
                action = self.actor.choose_action(
                    previous_preprocessed_observation_image)

                action = np.clip(
                    np.random.normal(
                        action,
                        randactionprob),
                    *self.ACTION_BOUND)

                # 做出動作，獲得場景資訊,已過關數,代理資訊
                observation, reward, done, info = env.step(
                    np.array(self.tableAction[int(action)]))
                tf.summary.histogram("Actions",
                                     values=self.tableAction[int(action)])
                buffer_aa.append(int(action))
                # 分開資訊
                observation_image, keys, time_remaining = observation
                if videoWriter:
                    videoWriter.write(cv2.cvtColor(
                        cv2.convertScaleAbs(observation_image,
                                            alpha=(255.0 / 1.0)),
                        cv2.COLOR_BGR2RGB))
                # 預處理圖像
                preprocessed_observation_image = np.reshape(
                    observation_image, [1, -1])
                # 計算獎勵
                stage_reward, previous_stage_time_remaining, done = self.reward_compute(done=done,
                                                                                  reward_total=stage_reward,
                                                                                  keys=keys,
                                                                                  previous_keys=previous_keys,
                                                                                  reward=reward,
                                                                                  previous_reward=previous_reward,
                                                                                  time_remaining=time_remaining,
                                                                                  previous_time_remaining=previous_time_remaining,
                                                                                  previous_stage_time_remaining=previous_stage_time_remaining)
                # 計算TD，這邊設定每步學習
                td_error = self.critic.learn(
                    previous_preprocessed_observation_image,
                    stage_reward,
                    preprocessed_observation_image)
                self.actor.learn(previous_preprocessed_observation_image,
                                 action,
                                 td_error)
                # 由這次的圖像決定下一步的走法
                previous_preprocessed_observation_image = preprocessed_observation_image
            self.summary = tf.Summary(
                value=[
                    tf.Summary.Value(
                        tag="Stage_reward",
                        simple_value=stage_reward)])
            self.train_writer.add_summary(self.summary, self.global_step)
            print("Pass Stage: ", reward)
            print("Actions: ", len(buffer_aa))
            print('Ep: ', ep, " | stage_reward: ", stage_reward)
            episode_reward.append(stage_reward)
            randactionprob = max([randactionprob * .9995, randactionprob_MIN])
        self.save()
        videoWriter.release()
        return episode_reward

    def createActionTable(self):
        tableAction = []
        for a in range(0, 3):
            for b in range(0, 3):
                for c in range(0, 2):
                    tableAction.append([a, b, c, 0])
        print("Action option: ", tableAction[6:12])
        return tableAction

    def actionIndex(self, action):
        return 0 if np.isnan(action) else int((action + 2) * 18 / 4)

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
        # 通過一個會開門的綠門會加0.1
        if (reward - previous_reward) < 0.2:
            reward_total += 3
        if (reward - previous_reward) > 0.9:
            # ***如果剩餘時間比場景時間多會變成加分獎勵，可能會極大增加Agent吃時間果實的機率。
            # ***另一種方式是剩餘的時間直接/1000加上去，這樣就沒有累加效果。
            print("Pass ", reward, " Stage!")
            # reward_total += (reward - previous_reward) * 100 - \
            #                 (previous_stage_time_remaining - time_remaining)
            reward_total += 100
            # 過關之後把時間留到下一關，儲存這回合時間供下次計算過關使用
            previous_time_remaining = time_remaining
            previous_stage_time_remaining = time_remaining
            # Lesson 1 repeat
            if reward > 0.5:
                # self.total_step +=1
                # if self.total_step >=5:
                #     done = True
                #     return reward_total, previous_stage_time_remaining, done
                env.seed(np.random.randint(5))
                # env.reset()
                done = True
            return reward_total, previous_stage_time_remaining, done

        # 假設過關的時候有順便吃到果實或鑰匙，所以預設為同時可以加成
        if previous_keys > keys:
            print("Get Key")
            reward_total += 5

        if previous_time_remaining < time_remaining and previous_time_remaining != 0:
            print("Get time power up")
            reward_total += 2
        else:
            reward_total -= 0.01
        if done and previous_time_remaining > 100:
            print("Agent died")
            # 如果剩餘時間越多就掛點，扣更多
            reward_total -= 100
        return reward_total, previous_stage_time_remaining, done

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model_save/params')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './model_save/params', write_meta_graph=False)


if __name__ == '__main__':
    episode_reward = []
    train_times = 5
    episode = 10000
    worker_id = 1
    retro = False
    realtime_mode = False
    env = ObstacleTowerEnv(
        '../ObstacleTower/obstacletower.exe',
        worker_id=1,
        retro=retro,
        realtime_mode=realtime_mode)
    kprun = MODEL()
    for i in range(0 , train_times + 1):
        fps = 60  # 保存视频的FPS，可以适当调整
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name = "video/obs" + str(i) + ".avi"
        videoWriter = cv2.VideoWriter(name, fourcc, fps, (84, 84))
        episode_reward = kprun.run(env=env,
                                   episode=episode,
                                   env_seed=0,
                                   env_floor=i,
                                   videoWriter=videoWriter)
    env.close()
