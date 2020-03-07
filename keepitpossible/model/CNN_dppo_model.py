import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

class DPPOMODEL(object):
    def __init__(self,
                 GLOBAL_SHARE=None,
                 PARAMETERS=None,
                 SCHEDULE=None):
        if PARAMETERS is None:
            raise NotImplementedError
        # 導入模組
        self.GLOBAL_SHARE = GLOBAL_SHARE
        self.PARAMETERS = PARAMETERS
        self.SCHEDULE = SCHEDULE

        # 設定輸入的維度
        self.image_features, self.action_dim = 512, 1

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Critic
        # 定義變數
        self.tfs = tf.placeholder(tf.float32, [None, 168, 168, 3], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, self.action_dim], 'discounted_r')
        # 建立神經網路
        net = tf.cast(self.tfs, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4)
        net = slim.conv2d(net, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512)
        self.v = slim.fully_connected(net, self.action_dim, activation_fn=None)

        # 計算損益
        self.advantage = self.tfdc_r - self.v

        self.closs = tf.reduce_mean(tf.square(self.advantage))

        self.ctrain_op = tf.train.AdamOptimizer(
            self.PARAMETERS.C_LR).minimize(
            self.closs)
        # Actor
        # 建立網路
        action_op, action_op_params = self._build_anet(
            'action_op', trainable=True)
        old_action_op, old_action_op_params = self._build_anet(
            'old_action_op', trainable=False)

        # 定義輸出範例
        self.sample_op = tf.argmax(tf.squeeze(action_op.sample(1),axis=0), 1)  # operation of choosing action
        # 更新
        self.update_old_action_op_op = [
            olda.assign(a) for a, olda in zip(
                action_op_params, old_action_op_params)]
        # 定義輸入變數
        self.tfa = tf.placeholder(
            tf.float32, [None, self.action_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # 機率比較
        ratio = action_op.prob(self.tfa) / \
            (old_action_op.prob(self.tfa) + 1e-5)
        # 替代損失
        surr = ratio * self.tfadv
        # 減少代理損失
        self.aloss = -tf.reduce_mean(tf.minimum(surr,
                                                tf.clip_by_value(ratio,
                                                                 1. - self.PARAMETERS.EPSILON,
                                                                 1. + self.PARAMETERS.EPSILON) * self.tfadv))
        self.atrain_op = tf.train.AdamOptimizer(
            self.PARAMETERS.A_LR).minimize(
            self.aloss)

        # log
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(
            self.SCHEDULE.LOG_PATH, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        while not self.GLOBAL_SHARE.COORD.should_stop():
            if self.GLOBAL_SHARE.EP < self.SCHEDULE.EP_MAX:
                # 等待收集資料
                self.GLOBAL_SHARE.UPDATE_EVENT.wait()
                print("Updating AC Model")
                # 用新的思考模式取代掉舊的模式
                self.sess.run(self.update_old_action_op_op)  #
                # 從各個平台內收集資料
                data = [self.GLOBAL_SHARE.QUEUE.get() for _ in range(
                    self.GLOBAL_SHARE.QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :168*168*3], data[:,
                                                168*168*3: 168*168*3 + 1], data[:, -1:]
                s = s.reshape(-1, 168, 168, 3)
                adv = self.sess.run(
                    self.advantage, {
                        self.tfs: s, self.tfdc_r: r})
                # 更新AC
                [self.sess.run(
                    self.atrain_op, {
                        self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(
                    self.PARAMETERS.UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(
                    self.PARAMETERS.UPDATE_STEP)]

                # 完成更新作業
                self.GLOBAL_SHARE.UPDATE_EVENT.clear()
                # 重新計數
                self.GLOBAL_SHARE.UPDATE_COUNTER = 0
                # 設成可以使用
                self.GLOBAL_SHARE.ROLLING_EVENT.set()

    def _build_anet(self, name, trainable):
        # 定義Actor 新舊的網路模型
        with tf.variable_scope(name):
            net = tf.cast(self.tfs, tf.float32)
            net = tf.div(net, 255.)
            net = slim.conv2d(net, 32, [8, 8], stride=4)
            net = slim.conv2d(net, 64, [4, 4], stride=2)
            net = slim.conv2d(net, 64, [3, 3], stride=1)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512)
            mu = 2 * slim.fully_connected(net, self.action_dim, activation_fn=tf.nn.tanh)
            sigma = slim.fully_connected(net, self.action_dim, activation_fn=tf.nn.softplus)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        # 決定下一步該怎麼做
        # s = s[np.newaxis, :]
        s = s.reshape(-1, 168, 168, 3)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a,
                       self.PARAMETERS.ACTION_BOUND[0],
                       self.PARAMETERS.ACTION_BOUND[1])

    def get_v(self, s):
        if s.ndim < 4:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def load(self):
        self.ckpt = tf.train.get_checkpoint_state(
            self.SCHEDULE.CHECKPOINT_PATH)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            pass

    def save(self):
        self.saver.save(
            self.sess,
            self.SCHEDULE.CHECKPOINT_PATH +
            'model.ckpt',
            self.GLOBAL_SHARE.EP,
            write_meta_graph=False)
