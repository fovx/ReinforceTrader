import gym
from gym import Env
from gym.spaces import Discrete, Box
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os, sys
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

SEED = 1234
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

data_path = ("C:/Users/USER/Desktop/project1/data/sinsung2_with_updownrate.csv")

df = pd.read_csv(data_path)

class RLEnv(gym.Env):
    def __init__(self,df, window_size = 5): #window_size변경 가능
        ###데이터 입력 및 전처리####
        self.minmaxscaler = MinMaxScaler()
        # 내가 원하는 특성 반환
        columns_to_scale = self.get_feature_to_scale()
        # 선택한 열에 대해 minmax(0-1사이 값으로)변환
        scaled_data = self.minmaxscaler.fit_transform(df[columns_to_scale])
        self.window_size = window_size #윈도우 크기를 인스턴스 변수로 설정
        #시퀀스로 변환한 데이터
        self.windowdata = self.sliding_window(scaled_data, self.window_size)

        ####행동 공간 정의####
        self.action_space = gym.spaces.Discrete(3) #buy/sell/hold

    def sliding_window(self, data, window_size):
        sequences = []
        for i in range(len(data) - window_size):
            sequences.append(data[i:i + window_size])
        return np.array(sequences)
    
    #내가 원하는 특성을 골라서 반환하는 함수 return안에 원하는 피처를 넣으면 된다.
    def get_feature_to_scale(self): 
        return ['close', 'volume', 'updownrate', 'UPDW']


    





#환경이 제대로 돌아가는지 확인하는 코드
# 환경을 초기화합니다.
env = RLEnv(df, window_size=5)
state = env.reset()

# 시퀀스 데이터 확인
print("Sequence data shape:", env.windowdata.shape)

# 한 스텝씩 환경을 진행합니다.
for _ in range(5):  # 5개의 스텝 진행
    action = env.action_space.sample()  # 임의의 액션 선택
    next_state, reward, done, _ = env.step(action)
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)

# 환경을 닫습니다.
env.close()
