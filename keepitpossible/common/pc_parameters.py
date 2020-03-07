import time


class PARAMETERS(object):
    def __init__(self,
                 UPDATE_STEP=5,
                 MIN_BATCH_SIZE=2048,
                 ACTION_BOUND=None,
                 EPSILON=0.2,
                 GAMMA=0.9,
                 A_LR=0.001,
                 C_LR=0.002):
        # 本身是更新時，幾步算一個循環  10
        self.UPDATE_STEP = UPDATE_STEP
        # 設定更新整個模型：每個代理走了N步就更新
        self.MIN_BATCH_SIZE = MIN_BATCH_SIZE
        # 限制控制，提高收斂程度
        self.ACTION_BOUND = ACTION_BOUND if ACTION_BOUND else [0, 7]
        # 超參數
        # Agent目標替換率 0.2
        self.EPSILON = EPSILON
        # Reward discount factor
        self.GAMMA = GAMMA
        # Actor 學習率
        # A_LR = 0.0001
        self.A_LR = A_LR
        # Critic 學習率
        # C_LR = 0.0002
        self.C_LR = C_LR
        print("\n================參數設定================",
              "\n循環步數: %d" % self.UPDATE_STEP,
              "\nBatch size: %s" % self.MIN_BATCH_SIZE,
              "\n動作區間: ", self.ACTION_BOUND,
              "\nEPSILON: ", self.EPSILON,
              "\nGAMMA: ", self.GAMMA,
              "\nActor 學習率: ", self.A_LR,
              "\nCritic 學習率: ", self.C_LR,
              "\n=======================================")


class SCHEDULE(object):
    def __init__(self,
                 LOG_PATH=None,
                 CHECKPOINT_PATH='./model_save_cnn/',
                 CHECKPOINT_STEP=50,
                 ENV_PATH='./ObstacleTower/obstacletower.exe',
                 N_WORKER=2,
                 EP_LEN=128,
                 EP_MAX=4000,
                 ENV_SEED=5,
                 ENV_FLOOR=0,
                 LESSON_END=5):
        timestr = time.strftime("%Y%m%d%H%M")
        # 設定LOG
        self.LOG_PATH = LOG_PATH if LOG_PATH else "logs/%s/" % timestr
        # 設定儲存點
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
        # 設定儲存間隔
        self.CHECKPOINT_STEP = CHECKPOINT_STEP
        # 設定環境檔路徑
        self.ENV_PATH = ENV_PATH
        # 設置幾個代理
        self.N_WORKER = N_WORKER
        # 代理人自己更新的步數
        self.EP_LEN = EP_LEN
        # 最大訓練回合數(每個代理人加起來的回合)
        self.EP_MAX = EP_MAX
        # 隨機種子限制數
        self.ENV_SEED = ENV_SEED
        # 起始樓層
        self.ENV_FLOOR = ENV_FLOOR
        # 課程終止關卡
        self.LESSON_END = LESSON_END
        print("\n================環境設定================",
              "\nLOG 路徑: %s" % self.LOG_PATH,
              "\n儲存點: %s" % self.CHECKPOINT_PATH,
              "\n儲存間隔: %d 步" % self.CHECKPOINT_STEP,
              "\n環境檔路徑: %s " % self.ENV_PATH,
              "\n代理更新的步數: %d " % self.EP_LEN,
              "\n最大訓練回合數: %d " % self.EP_MAX,
              "\n隨機種子限制數: %d" % self.ENV_SEED,
              "\n起始樓層: %d " % self.ENV_FLOOR,
              "\n課程終止關卡: %d" % self.LESSON_END,
              "\n=======================================")
