import tensorflow as tf
import numpy as np


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

        # 建立神經網路
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.tfs = tf.placeholder(tf.float32, [None, 168, 168, 3], 'state')
        c0 = tf.cast(self.tfs, tf.float32)
        c1 = tf.nn.relu(self.conv(c0,'c1',nf=32,rf=8,stride=4,init_scale=np.sqrt(2)))
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
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        # 建立網路層
        # l1 = tf.layers.dense(
        #     inputs=pre_s,
        #     units=100,  # number of hidden units
        #     activation=tf.nn.relu,
        #     name='l1'
        # )
        self.v = tf.layers.dense(
            inputs=pre_s,
            units=1,  # output units
            activation=None,
            name='V'
        )
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
        self.sample_op = tf.squeeze(
            action_op.sample(1),
            axis=0)  # operation of choosing action
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

    # from Open AI baseline

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
            # l1 = tf.layers.dense(inputs=pre_s,
            #                      units=200,  # number of hidden units
            #                      activation=tf.nn.relu,
            #                      name='l1',
            #                      trainable=trainable
            #                      )
            mu = 2 * tf.layers.dense(inputs=pre_s,
                                     units=self.action_dim,  # number of hidden units
                                     activation=tf.nn.tanh,
                                     name='mu',
                                     trainable=trainable
                                     )
            sigma = tf.layers.dense(inputs=pre_s,
                                    units=self.action_dim,  # output units
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
