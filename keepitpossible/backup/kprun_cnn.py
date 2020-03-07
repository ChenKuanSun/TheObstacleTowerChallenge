from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import tensorflow as tf
import os
import time
import threading
import queue

# 運行環境設定：
# 運行環境設定：
# 設置幾個代理
N_WORKER = 1
# 代理人自己更新的步數
EP_LEN = 500
# 最大訓練回合數(每個代理人加起來的回合)
EP_MAX = N_WORKER * 10000
# 設定更新整個模型：每個代理走了N步就更新
UPDATE_STEP = 20  # 本身是循環更新步
MIN_BATCH_SIZE = N_WORKER * UPDATE_STEP * 1
# 設定輸入的維度
image_features, action_dim = 512, 1
# 限制控制，提高收斂程度
ACTION_BOUND = [6, 12]


# 超參數
# Agent目標替換率
EPSILON = 0.4
# Reward discount factor
GAMMA = 0.7
# Actor 學習率
# A_LR = 0.0001
A_LR = 0.001
# Critic 學習率
# C_LR = 0.0002
C_LR = 0.002



class MODEL(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, 168, 168, 3], 'state')
        c0 = tf.cast(self.tfs, tf.float32) / 255.
        c1 = tf.nn.relu(self.conv(c0,
                                  'c1',
                                  nf=32,
                                  rf=8,
                                  stride=4,
                                  init_scale=np.sqrt(2)))
        c2 = tf.nn.relu(
            self.conv(
                c1,
                'c2',
                nf=64,
                rf=4,
                stride=2,
                init_scale=np.sqrt(2)))
        c3 = tf.nn.relu(
            self.conv(
                c2,
                'c3',
                nf=64,
                rf=3,
                stride=1,
                init_scale=np.sqrt(2)))
        nh = np.prod([v.value for v in c3.get_shape()[1:]])
        h3 = tf.reshape(c3, [-1, nh])
        pre_s = tf.nn.relu(self.fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
        # Critic
        # 定義變數
        # self.tfs = tf.placeholder(tf.float32, [None, image_features], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # 建立網路層
        l1 = tf.layers.dense(
            inputs=pre_s,
            units=100,  # number of hidden units
            activation=tf.nn.relu,
            name='l1'
        )
        self.v = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=None,
            name='V'
        )
        # 計算損益
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # Actor
        # 建立網路
        action_op, action_op_params = self._build_anet(
            'action_op', trainable=True)
        old_action_op, old_action_op_params = self._build_anet(
            'old_action_op', trainable=False)

        # 定義輸出範例
        self.sample_op = tf.squeeze(
            action_op.sample(1),
            axis=0)  # operation of choosing action
        # 更新
        self.update_old_action_op_op = [
            olda.assign(a) for a, olda in zip(
                action_op_params, old_action_op_params)]
        # 定義輸入變數
        self.tfa = tf.placeholder(tf.float32, [None, action_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # 機率比較
        ratio = action_op.prob(self.tfa) / \
            (old_action_op.prob(self.tfa) + 1e-5)
        # 替代損失
        surr = ratio * self.tfadv
        # 減少代理損失
        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        # log
        self.train_writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.tableAction = self.createActionTable()

    def createActionTable(self):
        tableAction = []
        for a in range(0, 3):
            for b in range(0, 3):
                for c in range(0, 2):
                    tableAction.append([a, b, c, 0])
        # print("Action option: ", tableAction[0:17])
        return tableAction

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                # 等待收集資料
                UPDATE_EVENT.wait()
                # 用新的思考模式取代掉舊的模式
                self.sess.run(self.update_old_action_op_op)  #
                # 從各個平台內收集資料
                # print("QUEUE in : ", QUEUE.qsize())
                while QUEUE.qsize():
                    s = QUEUE.get()
                    a = QUEUE.get()
                    r = QUEUE.get()
                
                # print("QUEUE out : ", QUEUE.qsize())
                # s, a, r = data[:, :image_features], data[:,
                #                                          image_features: image_features + action_dim], data[:, -1:]
                adv = self.sess.run(
                    self.advantage, {
                        self.tfs: s, self.tfdc_r: r})
                # 更新AC
                [self.sess.run(self.atrain_op,
                               {self.tfs: s,
                                self.tfa: a,
                                self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(
                    UPDATE_STEP)]
                # 完成更新作業
                UPDATE_EVENT.clear()
                # 重新計數
                GLOBAL_UPDATE_COUNTER = 0
                # 設成可以使用
                ROLLING_EVENT.set()

    # from Open AI baseline
    # def cnn(self, s):

        # return tf.reshape(h, [-1]).eval()

    def conv(self,
             x,
             scope,
             *,
             nf,
             rf,
             stride,
             pad='VALID',
             init_scale=1.0,
             data_format='NHWC',
             one_dim_bias=False):
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
        bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
        nin = x.get_shape()[channel_ax].value
        wshape = [rf, rf, nin, nf]
        with tf.variable_scope(scope):
            w = tf.get_variable(
                "w", wshape, initializer=self.ortho_init(init_scale))
            b = tf.get_variable(
                "b",
                bias_var_shape,
                initializer=tf.constant_initializer(0.0))
            if not one_dim_bias and data_format == 'NHWC':
                b = tf.reshape(b, bshape)
            return tf.nn.conv2d(
                x,
                w,
                strides=strides,
                padding=pad,
                data_format=data_format) + b

    def fc(self, x, scope, nh, *, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(scope):
            nin = x.get_shape()[1].value
            w = tf.get_variable(
                "w", [nin, nh], initializer=self.ortho_init(init_scale))
            b = tf.get_variable(
                "b", [nh], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(x, w) + b

    def ortho_init(self, scale=1.0):
        def _ortho_init(shape, dtype, partition_info=None):
            # lasagne ortho init for tf
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4:  # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
        return _ortho_init
    ####################################################

    def _build_anet(self, name, trainable):
        # 定義Actor 新舊的網路模型
        with tf.variable_scope(name):
            c0 = tf.cast(self.tfs, tf.float32) / 255.
            c1 = tf.nn.relu(self.conv(c0,
                                      'c1',
                                      nf=32,
                                      rf=8,
                                      stride=4,
                                      init_scale=np.sqrt(2)))
            c2 = tf.nn.relu(self.conv(c1,
                                      'c2',
                                      nf=64,
                                      rf=4,
                                      stride=2,
                                      init_scale=np.sqrt(2)))
            c3 = tf.nn.relu(self.conv(c2,
                                      'c3',
                                      nf=64,
                                      rf=3,
                                      stride=1,
                                      init_scale=np.sqrt(2)))
            nh = np.prod([v.value for v in c3.get_shape()[1:]])
            h3 = tf.reshape(c3, [-1, nh])
            pre_s = tf.nn.relu(self.fc(h3,
                                       'fc1',
                                       nh=512,
                                       init_scale=np.sqrt(2)))
            l1 = tf.layers.dense(inputs=pre_s,
                                 units=200,  # number of hidden units
                                 activation=tf.nn.relu,
                                 name='l1',
                                 trainable=trainable
                                 )
            mu = 2 * tf.layers.dense(inputs=l1,
                                     units=action_dim,  # number of hidden units
                                     activation=tf.nn.tanh,
                                     name='mu',
                                     trainable=trainable
                                     )
            sigma = tf.layers.dense(inputs=l1,
                                    units=action_dim,  # output units
                                    activation=tf.nn.softplus,  # get action probabilities
                                    name='sigma',
                                    trainable=trainable
                                    )
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        # 決定下一步該怎麼做
        # s = s[np.newaxis, :]
        s = s.reshape(-1, 168, 168, 3)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, ACTION_BOUND[0], ACTION_BOUND[1])

    def get_v(self, s):
        if s.ndim < 4:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model_save/params')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './model_save/params', write_meta_graph=False)


class Worker(object):
    def __init__(
            self,
            envpath,
            wid,
            retro,
            realtime_mode,
            env_seed=0,
            env_floor=0):
        self.wid = wid
        self.env = ObstacleTowerEnv(environment_filename=envpath,
                                    worker_id=wid,
                                    retro=retro,
                                    realtime_mode=realtime_mode)
        self.kprun = GLOBAL_KPRUN
        self.tableAction = self.createActionTable()
        # 設定關卡
        self.env_seed = env_seed
        self.env_floor = env_floor
        self.step = 0
        self.summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag="Stage_reward " + str(self.wid),
                    simple_value=0)])
        self.kprun.train_writer.add_summary(self.summary, 0)

    def createActionTable(self):
        tableAction = []
        for a in range(0, 3):
            for b in range(0, 3):
                for c in range(0, 2):
                    tableAction.append([a, b, c, 0])
        # print("Action option: ", tableAction[0:17])
        return tableAction

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
        if (reward - previous_reward) > 0 and (reward - previous_reward) < 0.3:
            reward_total += 3
        elif (reward - previous_reward) > 0.9:
            # ***如果剩餘時間比場景時間多會變成加分獎勵，可能會極大增加Agent吃時間果實的機率。
            # ***另一種方式是剩餘的時間直接/1000加上去，這樣就沒有累加效果。
            print("Pass ", reward, " Stage!")
            # reward_total += (reward - previous_reward) * 100 - \
            #                 (previous_stage_time_remaining - time_remaining)

            reward_total += 200
            # 過關之後把時間留到下一關，儲存這回合時間供下次計算過關使用
            previous_time_remaining = time_remaining
            previous_stage_time_remaining = time_remaining
            # Lesson 1 repeat
            if reward > 6.5:
                # self.total_step +=1
                # if self.total_step >=5:
                #     done = True
                #     return reward_total, previous_stage_time_remaining, done
                self.env.seed(np.random.randint(5))
                # env.reset()
                done = True
            return reward_total, previous_stage_time_remaining, done

        # 假設過關的時候有順便吃到果實或鑰匙，所以預設為同時可以加成
        if previous_keys > keys:
            # print("Get Key")
            reward_total += 5

        if previous_time_remaining < time_remaining and previous_time_remaining != 0:
            # print("Get time power up")
            reward_total += 2
        else:
            reward_total -= 0.5
        if done and previous_time_remaining > 100:
            print("Agent died")
            # 如果剩餘時間越多就掛點，扣更多
            # reward_total -= (10 + time_remaining / 100)
            reward_total -= 100
        return reward_total, previous_stage_time_remaining, done

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        # 設定關卡
        self.env.seed(self.env_seed)
        self.env.floor(self.env_floor)
        # 只要還沒達到目標回合就LOOP
        while not COORD.should_stop():
            # 紀錄步數
            self.step += 1
            # 重設關卡
            obs = self.env.reset()
            # 初始化
            done = False
            stage_reward = 0.0
            reward = 0
            keys = 0
            # 檢查是否有吃到加時間的，如果是第一回合出來沒有time_remaining，事先定義
            time_remaining = 3000
            previous_stage_time_remaining = time_remaining
            # 預處理圖像
            # previous_preprocessed_observation_image = np.reshape(obs[0], [-1])
            previous_preprocessed_observation_image = obs[0]
            buffer_s, buffer_a, buffer_r = [], [], []
            # 只要沒死
            while not done:
                # 如果模型正在更新就等待更新完成
                if not ROLLING_EVENT.is_set():
                    # 等待更新完成
                    ROLLING_EVENT.wait()
                    # 清除記憶體，使用新的代理收集資料
                    buffer_s, buffer_a, buffer_r = [], [], []

                # 儲存上一個動作狀態，供計算獎勵用
                previous_keys = keys
                previous_reward = reward
                previous_time_remaining = time_remaining

                # 根據上一次的狀態決定動作
                action = self.kprun.choose_action(
                    previous_preprocessed_observation_image)
                action = np.clip(np.random.normal(
                    action,
                    1.),
                    *[6, 12])

                # 做出動作，獲得場景資訊,已過關數,代理資訊
                observation, reward, done, info = self.env.step(
                    np.array(self.tableAction[int(action)]))

                # 預處理模型需要的資料
                observation_image, keys, time_remaining = observation
                # preprocessed_observation_image = np.reshape(
                #     observation_image, [-1])
                preprocessed_observation_image = observation_image
                stage_reward, previous_stage_time_remaining, done = self.reward_compute(done=done,
                                                                                        reward_total=stage_reward,
                                                                                        keys=keys,
                                                                                        previous_keys=previous_keys,
                                                                                        reward=reward,
                                                                                        previous_reward=previous_reward,
                                                                                        time_remaining=time_remaining,
                                                                                        previous_time_remaining=previous_time_remaining,
                                                                                        previous_stage_time_remaining=previous_stage_time_remaining)
                # Normalize reward~不知道中文怎麼打
                stage_reward = stage_reward+8 / 8

                # 把這次狀態存入 記憶體
                buffer_s.append(np.array([preprocessed_observation_image]))
                buffer_a.append(action)
                buffer_r.append(stage_reward)

                # 儲存下一步要參考的圖像
                previous_preprocessed_observation_image = preprocessed_observation_image

                # 達到更新時，自己先做處理。
                GLOBAL_UPDATE_COUNTER += 1
                # 太多自己就先處理更新
                if len(buffer_s) == EP_LEN - \
                        1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.kprun.get_v(preprocessed_observation_image)
                    # 計算折扣獎勵
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    # 整理維度
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    # 把資料放入共享記憶體
                    QUEUE.put(bs)
                    QUEUE.put(ba)
                    QUEUE.put(br)
                    # print("len(buffer_s)", len(buffer_s))
                    # print("bs.shape", bs.shape)
                    # 清空暫存
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 如果整個模型步數到達最小BATCH 就整個更新
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        # 停止收集資料
                        ROLLING_EVENT.clear()
                        # 更新PPO
                        UPDATE_EVENT.set()
                    # 達到最多EP停止訓練
                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()
                        break
            # 紀錄獎勵
            self.summary = tf.Summary(
                value=[
                    tf.Summary.Value(
                        tag="Stage_reward " + str(self.wid),
                        simple_value=stage_reward)])
            self.kprun.train_writer.add_summary(self.summary, self.step)
            GLOBAL_EP += 1
            print(
                '{0:.1f}%'.format(
                    GLOBAL_EP /
                    EP_MAX *
                    100),
                '|W%i' %
                self.wid,
                '|Ep_r: %.2f' %
                stage_reward,
            )
        self.env.close()


if __name__ == '__main__':
    # 建立物件
    GLOBAL_KPRUN = MODEL()
    # GLOBAL_KPRUN.load()
    # 建立多執行緒
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    # 現在不更新
    UPDATE_EVENT.clear()
    # 設定開始
    ROLLING_EVENT.set()

    workers = [Worker(envpath='./ObstacleTower/obstacletower.exe',
                      wid=i,
                      retro=False,
                      realtime_mode=False,
                      env_seed=0,
                      env_floor=0) for i in range(N_WORKER)]

    # 觀察者
    workers.append(Worker(envpath='./ObstacleTower/obstacletower.exe',
                          wid=N_WORKER + 1,
                          retro=False,
                          realtime_mode=True,
                          env_seed=0,
                          env_floor=0))

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []

    # 宣告共用記憶體
    QUEUE = queue.Queue()
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # 建立模型更新的執行緒
    threads.append(threading.Thread(target=GLOBAL_KPRUN.update, ))
    threads[-1].start()
    COORD.join(threads)
    # 儲存模型
    GLOBAL_KPRUN.save()
    time.sleep(5)

    obs = env.reset()
    print("執行測試環境，如果要離開請按Q")
    previous_preprocessed_observation_image = np.reshape(obs[0])
    while True:
        action = GLOBAL_KPRUN.choose_action(
            previous_preprocessed_observation_image)
        # 多執行緒會有跑不動的問題
        if np.isnan(action):
            action = np.random.randint(6, high=12)
        # 做出動作，獲得場景資訊,已過關數,代理資訊
        observation, reward, done, info = env.step(
            np.array(GLOBAL_KPRUN.tableAction[int(action)]))
        # 預處理模型需要的資料
        observation_image, keys, time_remaining = observation
        preprocessed_observation_image = np.reshape(
            observation_image, [-1])
        previous_preprocessed_observation_image = preprocessed_observation_image
        if done:
            break
    env.close()
