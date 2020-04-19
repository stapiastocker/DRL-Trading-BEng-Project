from env.gym_environment import TradingEnv
from env.data_generators import SineSignal, DoubleSineSignal

import gym
import json
import datetime as dt
import pandas as pd
import numpy as np
import optuna

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


#Hyper-Parameter Optimisation
def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

def objective_fn(trial):
    agent_params = optimize_ppo2(trial)

    generator = SineSignal(period_1 = 25, amplitude_1 = 1) #DoubleSineSignal(period_1=50, period_2=12.5, amplitude_1 = 2, amplitude_2 = 0.5) #(period_1=100, period_2=10, amplitude_1 = 10, amplitude_2 = 2)
    chosen_timesteps = 20000
    reward_function = 'Price' #'Sortino' for Sortino Ratio, 'Price' for price change based reward. 
    commmission = .2
    history_lookback = 2
    environment = TradingEnv(data_generator=generator,
                            reward_function=reward_function,
                            commission_fee=commmission,
                            history_lookback=history_lookback)


    env = DummyVecEnv([lambda: environment])
    
    model = PPO2(MlpPolicy, env, nminibatches=1, **agent_params)
    
    model.learn(int(chosen_timesteps*0.75))
    
    rewards_ = []
    reward_total = 0.0
    
    state = None
    obs = env.reset()
    for i in range(int(chosen_timesteps*0.25)):
        action, state = model.predict(obs, state=state)
        obs, reward, _, _ = env.step(action)
        print("Reward", reward)
        reward_total += reward[0]
        print("Reward Total", reward_total)

    rewards_.append(reward_total)
    reward_total = 0.0
    obs = env.reset()

    last_reward = np.mean(rewards_)
    trial.report(-1 * last_reward)
    return -1 * last_reward

def optimise(n_trials = 10, n_jobs = 1): 
    study = optuna.create_study(study_name='optimize_profit', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    json.dump(study.best_params, open( "model_params_sortino.json", 'w' ))


#Optimise
optimise()