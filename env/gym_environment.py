import gym
from gym import spaces
from tgym.core import Env
from tgym.utils import calc_spread
import numpy as np
import pandas as pd
import random
import json
from empyrical import sortino_ratio
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics


mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "lines.linewidth": 1,
        "lines.markersize": 8
    }
)


class TradingEnv(Env):
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """
    metadata = {'render.modes': ['human']}
    visualization = None

    actions = {'hold': 0, 'buy': 1, 'sell': 2 }

    positions = {'flat': np.array([1, 0, 0]), 'long': np.array([0, 1, 0]), 'short': np.array([0, 0, 1])}

    def __init__(self, data_generator, testing_data_generator, reward_function, commission_fee=0, history_lookback=2):
        """Initialisation
        """
        self._reward_function = reward_function
        self._data_gen = data_generator
        self._testing_data_gen = testing_data_generator
        self._iteration_num = 0
        self._start_render = True
        self._commission_fee = commission_fee
        self._price_history = []
        self._history_lookback = history_lookback
        self._price_range = 0
        self._net_worth = 0
        self._net_worths = []
        self._net_worth_ma = 0
        self._net_worths_ma = []
        self._long_ma = False
        self.temp_pnl_ma = 0
        self.previous_ma_1 = float('nan')
        self.previous_ma_2 = float('nan')
        self.reset()


        self.action_space = spaces.Discrete(3)  

        observation = self._get_observation()
        self.state_shape = observation.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float32)


        

    def reset(self, testing = False):
        """Reset environment. Reset rewards, data generator, pnl...
        """
        self._iteration_num = 0
        print("Testing True?:", testing)
        if testing:
            self._data_gen = self._testing_data_gen
        self._data_gen.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self.positions['flat']
        self._entry_value = 0
        self._exit_value = 0
        self._entry_value_ma = 0
        self._exit_value_ma = 0
        self._closed_plot = False
        self._balance = 1
        self._plot = False


        for i in range(self._history_lookback): #history_length: historical values to add to the observation vector.
            self._price_history.append(self._data_gen.next()) #next calls the next value in the iteration

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action_value = self.actions['hold'] #sets the action when reset to hold array [1, 0, 0]
        return observation

    def step(self, action):
        """Take an action such as buy,sell,hold and compute the reward.
        Reward is only calculated once a position is closed. 
        """

        self._action_value = action
        print('Action:', self._action_value)
        self._iteration_num += 1
        print("Iteration: ", self._iteration_num)
        
        reward = 0 
        trade_pnl = 0
        trade_sortino_pnl = 0
        info = {}
        done = False
        if action == self.actions['buy'] and all(self._position == self.positions['flat']): 
            reward -= self._commission_fee
           
            self._position = self.positions['long']
            self._entry_value = self._price_history[-1]
            info['entry_value'] = self._entry_value
            self._entry_point = self._iteration_num
            self._plot = True

        elif action == self.actions['sell'] and all(self._position == self.positions['long']): 
            reward -= self._commission_fee
            
            self._exit_value = self._price_history[-1]   
            info['exit_value'] = self._exit_value
            if self._reward_function == 'Sortino':
                trade_sortino_pnl = self._calc_int_sortino(self._entry_point)
            elif self._reward_function == 'Price':
                trade_pnl = self._calc_trade_pnl()
            info['trade_pnl'] = trade_pnl
            self._position = self.positions['flat']
            self._entry_value = 0
            self._entry_point = 0
            self._plot = True
        
        #Sortino based reward
        if self._reward_function == 'Sortino':
            if abs(trade_sortino_pnl) != float("inf") and not np.isnan(reward):
                pass
            else:
                print("Else Triggered")
                trade_sortino_pnl = 0
            reward += trade_sortino_pnl
            print("Instant Sortino:", trade_sortino_pnl)
            self._total_pnl += trade_sortino_pnl
            self._total_reward += reward
        #Price based reward
        elif self._reward_function == 'Price':
            reward += trade_pnl
            self._total_pnl += trade_pnl 
            self._total_reward += reward


        # Game over code
        try:
            self._price_history.append(self._data_gen.next()) 
        except StopIteration:
            info['status'] = 'No more data'
            done = True

        observation = self._get_observation()
        return observation, reward, done, info
    
    def sortino_ratio_calc(self, net_worths):
        #Sortino Ratio
        if net_worths:
            length = len(net_worths)
            if length < 100:
                returns = np.diff(net_worths)[-length:] 
            else:
                returns = np.diff(net_worths)[-100:]
            s_r =  sortino_ratio(returns = returns) 
            return s_r
        else:
            return 0

    def moving_average(self, price_data, period):
        """
        Needs to take the price history and perform avarages based on the values passed. 
        Each iterations creates a new point to be added to the plot. 
        So function should only create one point 
        """
        if len(price_data) < period:
            return float("nan") 
        else:
            average_price = statistics.mean(price_data[-period:])
            return average_price


    def render(self, calc_sortino_or_render = False, savefig=False, filename='myfig'):
        """
        Plotting
        """
        if calc_sortino_or_render:
            if self._start_render:
                self._net_worth = 0
                self._net_worths = []
                self._start_render = False

            if any(self._position == self.positions['long']) and self._start_render == False:
                temp_pnl = self._calc_temp_pnl()
            
            if (temp_pnl != 0) and (temp_pnl is not None) and self._start_render == False: #and (temp_pnl is not None)
                self._net_worth += temp_pnl
                self._net_worths.append(self._net_worth)
                print("Net Worths: ", self._net_worths[-2:])

            #Moving Average Strategy Sortino Calc. 
            ma_1 = self.moving_average(self._price_history, 5)
            ma_2 = self.moving_average(self._price_history, 10)

            if ma_1 > ma_2 and self.previous_ma_1 < self.previous_ma_2:
                #Buy
                self._entry_value_ma = self._price_history[-1]
                self._long_ma = True 

            if self._long_ma:
                #calc temp pnl
                self.temp_pnl_ma =  self._price_history[-1] - self._price_history[-2]

            if (self.temp_pnl_ma != 0) and (self.temp_pnl_ma is not None):
                self._net_worth_ma += self.temp_pnl_ma
                self._net_worths_ma.append(self._net_worth_ma)

            if ma_1 < ma_2 and self.previous_ma_1 > self.previous_ma_2 and self._long_ma == True:
                #Sell
                self._exit_value_ma = self._price_history[-1]
                pnl = self._exit_value_ma - self._entry_value_ma
                self._entry_value_ma = 0 
                self._long_ma = False

            

            self.previous_ma_1 = ma_1
            self.previous_ma_2 = ma_2

            sortino = self.sortino_ratio_calc(self._net_worths)
            sortino_ma = self.sortino_ratio_calc(self._net_worths_ma)
            print("Sortino Ratio:", sortino)
            print("MA Sortino Ratio", sortino_ma)



        else:
            if self._start_render:
                self._f, self._ax = plt.subplots(sharex=True, figsize=(10, 6))
                self._ax.set_ylabel('Price')
                self._ax.set_xlabel('Iteration')
                self._start_render = False
                self._net_worth = 0
                self._net_worths = []



            #calc intermediary pnl
            if any(self._position == self.positions['long']) and self._start_render == False:
                temp_pnl = self._calc_temp_pnl()
            
            if (temp_pnl != 0) and (temp_pnl is not None) and self._start_render == False: #and (temp_pnl is not None)
                self._net_worth += temp_pnl
                self._net_worths.append(self._net_worth)
                print("Net Worths: ", self._net_worths[-2:])

            #Price
            price = self._price_history[-1]#[1]
            self._ax.plot([self._iteration_num, self._iteration_num + 1],
                            [price, price], color='black')

            #moving average 1
            ma_1 = self.moving_average(self._price_history, 5)
            self._ax.plot([self._iteration_num-1, self._iteration_num],
                            [self.previous_ma_1, ma_1], color='blue', alpha = 0.45) #ah they're just doing small lines...
            
            #moving average 2
            ma_2 = self.moving_average(self._price_history, 10)
            self._ax.plot([self._iteration_num-1, self._iteration_num],
                            [self.previous_ma_2, ma_2], color='darkgoldenrod', alpha = 0.8) #ah they're just doing small lines...

            #moving average crossover checks
            if ma_1 > ma_2 and self.previous_ma_1 < self.previous_ma_2:
                self._ax.scatter(self._iteration_num + 0.5, price, color='blue', marker='x', alpha = 1, linewidth=3)
                self._entry_value_ma = self._price_history[-1]
                self._long_ma = True

            if self._long_ma:
                #calc temp pnl
                self.temp_pnl_ma =  self._price_history[-1] - self._price_history[-2]

            if (self.temp_pnl_ma != 0) and (self.temp_pnl_ma is not None):
                self._net_worth_ma += self.temp_pnl_ma
                self._net_worths_ma.append(self._net_worth_ma)
                print("Ma Net Worths:", self._net_worths[-10:])

            if ma_1 < ma_2 and self.previous_ma_1 > self.previous_ma_2 and self._long_ma == True:
                self._ax.scatter(self._iteration_num + 0.5, price, color='darkgoldenrod', marker='x', alpha = 1, linewidth=3)
                self._exit_value_ma = self._price_history[-1]
                pnl = self._exit_value_ma - self._entry_value_ma
                self._entry_value_ma = 0 
                self._long_ma = False

            self.previous_ma_1 = ma_1
            self.previous_ma_2 = ma_2

            ymin, ymax = self._ax.get_ylim()
            yrange = max(self._price_range, ymax - ymin)
            if (self._action_value == self.actions['sell']).all() and self._plot:
                self._ax.scatter(self._iteration_num + 0.5, price + 0.04 *
                                    yrange, color='orangered', marker='v', linewidth = 3)
                self._plot = False
            elif (self._action_value == self.actions['buy']).all() and self._plot:
                self._ax.scatter(self._iteration_num + 0.5, price - 0.04 *
                                    yrange, color='lawngreen', marker='^', linewidth = 3)
                self._plot = False
            
            sortino = self.sortino_ratio_calc(self._net_worths)
            sortino_ma = self.sortino_ratio_calc(self._net_worths_ma)
            print("Sortino Ratio:", sortino)
            print("MA Sortino Ratio", sortino_ma)

            plt.suptitle('Cumul. Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                        'Cumul. PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                        'Sortino Ratio: ' + "%.4f" % sortino + ' ~ ' +
                        'Position: ' + ['flat', 'long', 'short'][list(self._position).index(1)] + ' ~ ' +
                        'Entry Price: ' + "%.2f" % self._entry_value)
            plt.suptitle('Sine Wave Time Series', y=0.90)
            #plt.title('Superimposed Noisy Sine Waves Time Series')
            #plt.suptitle('Three Noisy Sine Waves Superimposed Time Series', y=0.90)
            self._f.tight_layout()
            plt.xticks(range(self._iteration_num)[::5])
            plt.xlim([max(0, self._iteration_num - 80.5), self._iteration_num + 0.5])
            plt.subplots_adjust(top=0.85)
            plt.pause(0.01)
            if savefig:
                plt.savefig(filename)

    def _get_observation(self):
        """Concatenate all required elements for the observation.
        """
        
        print("Entry Price", np.array([self._entry_value]))

        position = self._position.tolist()

        observation = np.array([prices for prices in self._price_history[-self._history_lookback:]] + [self._entry_value] +  position)

        return observation

    def _calc_trade_pnl(self,):
        """Calculate the PnL for a closed position. 

        """
        if all(self._position == self.positions['long']):
            return self._exit_value - self._entry_value
        if all(self._position == self.positions['short']):
            return self._entry_value - self._exit_value


    def _calc_temp_pnl(self):
        """Calculate the PnL at each price change .

        """
        if all(self._position == self.positions['long']):
            return self._price_history[-1] - self._price_history[-2]
        if all(self._position == self.positions['short']):
            return self._entry_value - self._price_history[-1]
    
    def _calc_int_sortino(self, entry_point):
        """
        Calculate the Sortino Ratio after each closed trade. 
        """
        
        sortino = self.sortino_ratio_calc(self._price_history[entry_point:])

        return sortino

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.

        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])
