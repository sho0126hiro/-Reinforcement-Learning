import random
import numpy as np
from typing import NamedTuple
from typing import List

import torch
import torch.nn as nn
from torch import optim

class Maze:
    """
    迷路自体に関するクラス
    """
    SIZE = 5 # 迷路はSIZE*SIZE
    def __init__(self):
        """ 迷路全体を構成する2次元配列、迷路の外周を壁とし、それ以外を通路とする。"""
        
        self.PATH = 0
        self.WALL = 1
        
        self.width = Maze.SIZE
        self.height = Maze.SIZE
        self.maze = []
        self.maze_state_number = []
        # 壁を作り始める開始ポイントを保持しておくリスト。
        self.lst_cell_start_make_wall = []
        # 迷路は、幅高さ5以上の奇数で生成する。
        if(self.height < 5 or self.width < 5):
            print("迷路のサイズを5以上にしてください．")
            exit()
        if (self.width % 2) == 0:
            self.width += 1
        if (self.height % 2) == 0:
            self.height += 1
        for x in range(0, self.width):
            row = []
            for y in range(0, self.height):
                if (x == 0 or y == 0 or x == self.width-1 or y == self.height -1):
                    cell = self.WALL
                else:
                    cell = self.PATH
                    # xyとも偶数の場合は、壁を作り始める開始ポイントとして保持。
                    if (x % 2 == 0 and y % 2 == 0):
                        self.lst_cell_start_make_wall.append([x, y])
                row.append(cell)
            self.maze.append(row)
        # スタートとゴールを入れる。
        self.maze[1][1] = 'S'
        self.maze[self.width-2][self.height-2] = 'G'
    def make_maze(self):
        """ 迷路の配列を作り戻す """
        # 壁の拡張を開始できるセルがなくなるまでループする。
        while self.lst_cell_start_make_wall != []:
            # 開始セルをランダムに取得してリストからは削除。
            x_start, y_start = self.lst_cell_start_make_wall.pop(random.randrange(0, len(self.lst_cell_start_make_wall)))
            # 選択候補が通路の場合は壁の拡張を開始する。
            if self.maze[x_start][y_start] == self.PATH:
                # 拡張中の壁情報を保存しておくリスト。
                self.lst_current_wall = []
                self.extend_wall(x_start, y_start)
        return self.maze
    def extend_wall(self, x, y):
        """ 開始位置から壁を2つずつ伸ばす """
        # 壁を伸ばすことのできる方向を決める。通路かつ、その2つ先が現在拡張中の壁ではない。
        lst_direction = []
        if self.maze[x][y-1] == self.PATH and [x, y-2] not in self.lst_current_wall:
            lst_direction.append('up')
        if self.maze[x+1][y] == self.PATH and [x+2, y] not in self.lst_current_wall:
            lst_direction.append('right')
        if self.maze[x][y+1] == self.PATH and [x, y+2] not in self.lst_current_wall:
            lst_direction.append('down')
        if self.maze[x-1][y] == self.PATH and [x-2, y] not in self.lst_current_wall:
            lst_direction.append('left')
        #壁を伸ばせる方向がある場合
        if lst_direction != []:
            #まずはこの地点を壁にして、拡張中の壁のリストに入れる。
            self.maze[x][y] = self.WALL
            self.lst_current_wall.append([x, y])
            # 伸ばす方向をランダムに決める
            direction = random.choice(lst_direction)
            # 伸ばす2つ先の方向が通路の場合は、既存の壁に到達できていないので、拡張を続ける判断のフラグ。
            contineu_make_wall = False
            # 伸ばした方向を壁にする
            if direction == 'up':
                contineu_make_wall = (self.maze[x][y-2] == self.PATH)
                self.maze[x][y-1] = self.WALL
                self.maze[x][y-2] = self.WALL
                self.lst_current_wall.append([x, y-2])
                if contineu_make_wall:
                    self.extend_wall(x, y-2)
            if direction == 'right':
                contineu_make_wall = (self.maze[x+2][y] == self.PATH)
                self.maze[x+1][y] = self.WALL
                self.maze[x+2][y] = self.WALL
                self.lst_current_wall.append([x+2, y])
                if contineu_make_wall:
                    self.extend_wall(x+2, y)
            if direction == 'down':
                contineu_make_wall = (self.maze[x][y+2] == self.PATH)
                self.maze[x][y+1] = self.WALL
                self.maze[x][y+2] = self.WALL
                self.lst_current_wall.append([x, y+2])
                if contineu_make_wall:
                    self.extend_wall(x, y+2)
            if direction == 'left':
                contineu_make_wall = (self.maze[x-2][y] == self.PATH)
                self.maze[x-1][y] = self.WALL
                self.maze[x-2][y] = self.WALL
                self.lst_current_wall.append([x-2, y])
                if contineu_make_wall:
                    self.extend_wall(x-2, y)
        else:
            previous_point_x, previous_point_y = self.lst_current_wall.pop()
            self.extend_wall(previous_point_x, previous_point_y)
    
    def print_maze(self):
        for row in self.maze:
            for cell in row:
                if cell == self.PATH:
                    print(' ', end='')
                elif cell == self.WALL:
                    print("#", end='')
                elif cell == 'S':
                    print('S', end='')
                elif cell == 'G':
                    print('G', end='')
            print()
    
    def make_maze_state(self):
        num = 0
        for y in range(len(self.maze)):
            for x in range(len(self.maze)):
                if 0 < y < Maze.SIZE-1 and 0 < x < Maze.SIZE-1:
                    self.maze_state_number.append(num)
                    num += 1
                else:
                    self.maze_state_number.append('#')
        self.maze_state_number = np.reshape(self.maze_state_number, (self.width, self.height)).tolist()
    
    def print_state(self):
        for row in self.maze_state_number:
            for cell in row:
                print(cell, end='')
            print()

class ActionAndNextState(NamedTuple):
    """
    次の状態行動
    """
    action: int
    next_state: int

class Transition(NamedTuple):
    """
    遷移を格納する
    """
    state: int
    action: int
    reward: float
    next_state: int
    next_moveable: List[int] # stateから遷移できる行動のリスト

class ReplayMemory:
    def __init__(self,capacity: int):
        self.capacity: int = capacity
        self.memory: List[Transition] = []

    def push(self, transition: Transition):
        self.memory.append(transition)
    
    def sample(self, batch_size) -> List[Transition]:
        # random.sample: 重複なし　ランダムでbatch_size分要素選択
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

BATCH_SIZE = 32
CAPACITY = 10000

class DQN:

    GAMMA = 0.9 # time-deray rate
    LEARNING_RATE = 0.01

    def __init__(self):
        """
        モデルの定義  
        fc：　full connected layer  
        input: 状態，行動の組み合わせ，状態: マス番号，行動: 方向index  
        output: 次に行う行動  
        """
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(2,32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32,32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, 1))
        print(self.model)
        # 最適化手法
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.0001)
        # loss function
        self.criterion = nn.MSELoss()
        self.replay_memory = ReplayMemory(CAPACITY)
        
    
    def predict(self, state, action) -> float:
        """
        NNを用いて推論
        @return{float} q_value Q値
        self.model.eval()をされている状態で関数を呼び出してください．
        """
        self.optimiser.zero_grad() # 一度計算された勾配結果を0にリセット
        q_value: float = self.model(torch.Tensor([state, action]))
        return q_value
    
    def replay(self, batch_size):
        """
        Experience Replayでネットワークの重みを学習
        """
        batch_size: int = min(batch_size, len(self.replay_memory))
        minibatch: List[Transition] = random.sample(self.replay_memory, batch_size)
        self.model.eval() # 訓練モードへ切替
        target_data: float = [] # 教師データ
        # 教師データの作成
        for element in minibatch:
            if element.state == Maze.GOAL_NUMBER:
                target_data.append(element.reward) # r = 1
            else:
                q_value_list: List[float] = []
                for a in element.next_moveable:
                    # s_t+1, a_i におけるQ値を取得する
                    q_value: float = self.predict(element.next_state, a)
                    q_value_list.append(q_value)
                # q値の最大値
                q_value_max: float = np.amax(np.array(q_value_list))
                target: float = element.reward + DQN.GAMMA * q_value_max
                target_data.append(target)
        # replay
        self.model.train()
        for i in range(len(minibatch)):
            input_data = torch.Tensor([minibatch[i].state, minibatch[i].action])
            out = self.model(input_data) # 現状のoutを計算
            loss = self.criterion(out, torch.Tensor(target_data[i])) # targetとの比較: lossの計算
            self.optimiser.zero_grad() # 勾配初期化
            loss.backward() # 勾配計算
            self.optimiser.step() # パラメータ更新

    
class Agent:
    """
    エージェントに関するクラス
    """
    ACTION: List[int] = [0,1,2,3] # up, right, down, left
    EPISODE: int = 10000
    WALK_MAX = 1000

    def __init__(self, maze: Maze, dqn: DQN):
        self.maze = maze
        self.dqn = dqn
        # up, right, down, left に動きたいとき，
        # 各変数を現在の状態に加える．
        self.__default_moveable = [-Maze.SIZE, 1, Maze.SIZE, -1] 

    def get_action_and_next_state(self, state: int, epsilon: float) -> ActionAndNextState:
        """
        行動と次の状態を取得する  
        using epsilon-greedy-method  
        """
        if np.random.rand() < epsilon:
            action = np.random.choice(Agent.ACTION)
        else:
            action = self.get_best_action(state)
        for i in range(len(Agent.ACTION)):
            if action == Agent.ACTION[i]:
                s_next = state + self.__default_moveable[i]
                break
        return ActionAndNextState(action,s_next)
    
    def get_best_action(self, state: int) -> int:
        """
        NNで一度学習（指定の状態からOutを出力）し，
        outの値が最大の物を選ぶ
        """
        q_max: float = -10000.00
        # choose action
        moveable = self.get_moveable(state)
        for move_candidate in moveable:
            q_value: float = self.dqn.predict(state, move_candidate)
            if q_max < q_value:
                q_max = q_value
                action = move_candidate
        return action

    def get_moveable(self, state: int) -> List[int]:
        """
        行動可能な選択肢を取得する
        """
        # up, right, down, left 
        moveable: List[int] = self.__default_moveable.copy()
        for move_candidate in self.__default_moveable:
            next_state = state + move_candidate
            # 外枠
            if next_state < 0:
                moveable.pop(0) # up
            if next_state > Maze.SIZE**2-1:
                moveable.pop(2) # down
            if state % Maze.SIZE == 0 or state == 0:
                moveable.pop(3) # left
            if state % Maze.SIZE == Maze.SIZE -1 or state == Maze.SIZE**2-1:
                moveable.pop(1) # right
            # 迷路内障害物関係
            ...
        return moveable

    def run(self):
        for episode in range(Agent.EPISODE):
            state = 0
            for step in range(Agent.WALK_MAX):
                moveable = self.get_moveable(state)
                action = self.get_action_and_next_state(state)
                ...

def main():
    maze = Maze()
    maze.make_maze()
    maze.make_maze_state()
    dqn = DQN()
    agent = Agent(maze,dqn)
    agent.run()

if __name__ == "__main__":
    main()