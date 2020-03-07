# import numpy as np


def compute(done,
            stage_clear,
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
    # 通過一個會開門的綠門會加0.1
    # print("done ", done,
    #       " reward ", reward,
    #       " previous_reward ", previous_reward)
    if (reward - previous_reward) > 0 and (reward - previous_reward) < 0.3:
        reward_total += 100
        # 如果用掉了KEY就在加100 促使AI用鑰匙開門
    if previous_keys > keys:
        reward_total += 100
    elif (reward - previous_reward) > 0.9:
        # ***如果剩餘時間比場景時間多會變成加分獎勵，可能會極大增加Agent吃時間果實的機率。
        # ***另一種方式是剩餘的時間直接/1000加上去，這樣就沒有累加效果。
        # reward_total += (reward - previous_reward) * 100 - \
        #                 (previous_stage_time_remaining - time_remaining)
        print("Pass ", stage_clear, " Stage!\n\n")
        stage_clear += 1
        reward_total += (previous_time_remaining / 100 + 200)
        # 過關之後把時間留到下一關，儲存這回合時間供下次計算過關使用
        previous_time_remaining = time_remaining
        previous_stage_time_remaining = time_remaining

        # 假設過關的時候有順便吃到果實，所以預設為同時可以加成
    if keys > previous_keys:
        print("Get Key\n\n")
        reward_total += 600

    if previous_time_remaining < time_remaining and previous_time_remaining != 0 and (
            reward - previous_reward) < 0.8:
        print("Get time power up\n\n")
        reward_total += 50
        # 時間到就扣100
    if done and time_remaining == 3000 and previous_time_remaining == 0:
        reward_total -= 100
    if done and previous_time_remaining > 100:
        # print("Agent died")
        # 如果剩餘時間越多就掛點，扣更多
        # reward_total -= (10 + time_remaining / 100)
        reward_total -= 100
        # print("time_remaining ", time_remaining,
        #       " previous_time_remaining ", previous_time_remaining,
        #       " previous_reward ", previous_reward,
        #       " reward ", reward,
        #       "reward_total ", reward_total)
    if done:
        print("=====Final in ", stage_clear, " Stage!======\n\n")
    return reward_total, previous_stage_time_remaining, done, stage_clear
