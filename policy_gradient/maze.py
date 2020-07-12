# 方策勾配法で迷路探索
import numpy as np
# import matplotlib.pyplot as plt

# 10^-8 学習終了条件（方策条件）
EPSILON = 10**-8 

"""
方策を決定するパラメータ
行は状態0～7、列は移動方向で↑、→、↓、←を表す
"""
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                    [np.nan, 1, np.nan, 1],  # s1
                    [np.nan, np.nan, 1, 1],  # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1],  # s4
                    [1, np.nan, np.nan, np.nan],  # s5
                    [1, np.nan, np.nan, np.nan],  # s6
                    [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                    ])
"""
softmax関数
@param {theta}
@return {pi}
"""
def softmax(theta):
    beta = 1.0 # 小さくするとランダム性が増す
    [m,n] = theta.shape # theta がm*n 行列
    pi = np.zeros((m,n))
    exp_theta = np.exp(beta*theta)
    for i in range(m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)  # nanを0に変換
    return pi

"""
次の行動と状態を返す
@param {pi} 現在の方策
@return {action,s_next} 
action : 次にとる行動
next_s : 次の状態
"""
def get_nextAS(pi,s):
    direction = ["up", "right", "down", "left"]
    # [s,:] >> s行目のすべての列の要素の確率でdirectionの要素を出力する    
    next_direction = np.random.choice(direction, p=pi[s, :]) 
    if next_direction == "up":
        action = 0
        s_next = s - 3  # 上に移動するときは状態の数字が3小さくなる
    elif next_direction == "right":
        action = 1
        s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
    elif next_direction == "down":
        action = 2
        s_next = s + 3  # 下に移動するときは状態の数字が3大きくなる
    elif next_direction == "left":
        action = 3
        s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる
    return [action, s_next]

"""
迷路を解く(1-episode)
@param {pi} init value : pi_0
@return {SA_history} 状態と行動の履歴
s : 現在の状態 next_s : 次の状態
a : 次にとる行動
"""
def maze_main(pi):
    s=0
    SA_history=[[0,np.nan]]
    while True:
        [a,next_s] = get_nextAS(pi,s)
        SA_history[-1][1] = a
        SA_history.append([next_s,np.nan])
        if next_s==8: # goal point
            break
        s = next_s
    return SA_history

"""
thetaを更新する
@param {theta}
@param {pi}
@param {SA_history}
@return {new_theta}
"""
def update_theta(theta,pi,SA_history):
    alpha = 0.1
    T = len(SA_history) -1 #step
    [m,n] = theta.shape
    delta_theta = theta.copy()
    # delta_thetaを求める
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):  # thetaがnanでない場合
                # SA_historyの要素（SAの0番目（状態））がiの時SAを返す
                SA_i =  [SA for SA in SA_history if SA[0] == i]
                # 状態iで行動jをしたものを取り出す
                SA_ij = [SA for SA in SA_history if SA == [i, j]]
                N_i = len(SA_i)  # 状態iで行動した総回数
                N_ij = len(SA_ij)  # 状態iで行動jをとった回数
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T
    new_theta = theta + alpha * delta_theta
    return new_theta

"""
方策勾配法
@return {[pi,steps]} 
pi 最終的な方策
steps 状態と行動の選択履歴
"""
def policy_gradient():
    steps = []
    theta = theta_0
    pi = softmax(theta)
    count=1
    while True:
        SA_history = maze_main(pi)
        new_theta = update_theta(theta, pi, SA_history)  # パラメータΘを更新
        new_pi = softmax(new_theta)  # 方策πの更新
        # 要素の和
        # print(np.sum(np.abs(new_pi - pi)))  # 方策の変化を出力（すべての要素に対して、new_pi - piを計算　>> 差の合計を求める）
        # print("迷路を解くのにかかったステップ数は" + str(len(SA_history) - 1) + "です")
        steps.append(len(SA_history)-1)
        if np.sum(np.abs(new_pi - pi)) < EPSILON:
            break
        theta = new_theta
        pi = new_pi
        if(count==1 or count==10 or count==30 or count==50 or count ==100 or count==130 or count==150 or count==160):
            print(count)
            print(theta)
        count+=1
    return [theta,pi,steps]

def main():
    [theta,pi,steps] = policy_gradient()
    print(theta)
    # print(pi)
    # print(steps)
    # plt.plot(steps)
    # plt.savefig('foo.png')

if __name__ == "__main__":
    main()