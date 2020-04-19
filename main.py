from env.gym_environment import TradingEnv
from env.data_generators import SineSignal, DoubleSineSignal, TripleSineSignal

import gym
import json
import datetime as dt
import pandas as pd
import numpy as np
import optuna

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DDPG


def evaluate(model, num_steps=1000):
    """
    Evaluate RL agent
    Return mean reward
    """
    ep_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset(testing = True)
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays because we are using vectorized env
        obs, rewards, dones, _ = env.step(action)
        env.render(calc_sortino_or_render = True)
       
    
        # Stats
        for i in range(env.num_envs):
            ep_rewards[i][-1] += rewards[i]
            if dones[i]:
                ep_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(ep_rewards[i])     
        n_episodes += len(ep_rewards[i])   

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


#generator = TripleSineSignal(period_1=75, period_2=50, period_3=12.5, amplitude_1 = 3, amplitude_2 = 2, amplitude_3 = 0.5)
generator = DoubleSineSignal(period_1=50, period_2=12.5, amplitude_1 = 2, amplitude_2 = 0.5) #SineSignal(period_1 = 50, amplitude_1 = 2) #DoubleSineSignal(period_1=50, period_2=12.5, amplitude_1 = 2, amplitude_2 = 0.5) #(period_1=100, period_2=10, amplitude_1 = 10, amplitude_2 = 2)
generator_testing = SineSignal(period_1 = 25, amplitude_1 = 1)

chosen_timesteps = 20000
reward_function = 'Price' #'Sortino' for Sortino Ratio, 'Price' for price change based reward. 
commmission = .2
history_lookback = 2
environment = TradingEnv(data_generator=generator,
                            testing_data_generator = generator_testing,
                            reward_function=reward_function,
                            commission_fee=commmission,
                            history_lookback=history_lookback)



env = DummyVecEnv([lambda: environment])

params = json.load( open("model_params.json"))

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam']
}


model = PPO2(MlpPolicy, env, verbose=0, nminibatches=1, **model_params) #tensorboard_log="./ml_trading_tensorboard/"

print("Learning Rate: ", model.learning_rate)

# Agent Training
train_timesteps = chosen_timesteps 
model.learn(total_timesteps=train_timesteps)

#Evaluate mean reward without visualisation
#mean_reward = evaluate(model, num_steps=5000)

#Testing
done = False
obs = env.reset(testing = True)
for i in range(int(chosen_timesteps*1.20)): #set a training period that is a 
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if 'status' in info and info['status'] == 'Plot closed.':
        done = True
    else:
        environment.render()