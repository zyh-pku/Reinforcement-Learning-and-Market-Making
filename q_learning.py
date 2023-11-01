import numpy as np
import matplotlib.pyplot as plt
from market import MarketEnvironment

class QLearningAgent:

    def __init__(self, dim_X, dim_Y, dim_action_a, dim_action_b, Delta, N_RL_iter=10000, N_learning_steps=3000):
        self.dim_X = dim_X
        self.dim_Y = dim_Y
        self.dim_action_a = dim_action_a
        self.dim_action_b = dim_action_b
        self.N_RL_iter = N_RL_iter
        self.N_learning_steps = N_learning_steps
        self.Delta = Delta
        self.GAMMA = 0.95
        self.GAMMA_Delta = np.exp(-self.GAMMA*self.Delta)
        
        # Initialize Q-table and other matrices
        self.Q_table = np.zeros((dim_X, dim_Y, dim_action_a, dim_action_b))
        self.Q_table_track = np.zeros((dim_X, dim_Y, dim_action_a, dim_action_b, N_RL_iter))
        self.state_counter_matrix = np.zeros((dim_X, dim_Y))
        self.state_action_counter_matrix = np.zeros((dim_X, dim_Y, dim_action_a, dim_action_b))

        # Define learning rate
        eps = 0.8
        eps0 = 0.95
        epoch = 2
        learning_rate = [eps*((eps0)**(i//epoch)) for i in range(N_learning_steps)]
        self.learning_rate_matrix = np.zeros((dim_X, dim_Y, dim_action_a, dim_action_b, N_learning_steps))
        for idx_x in range(dim_X):
            for idx_y in range(dim_Y):
                for p_a in range(dim_action_a):
                    for p_b in range(dim_action_b):
                        self.learning_rate_matrix[idx_x, idx_y, p_a, p_b, :] = np.array(learning_rate)

        # Define exploration probability
        eps = 1.0
        eps0 = 0.8
        epoch = 100
        EPSILON_list = [eps*((eps0)**(i//epoch)) for i in range(N_learning_steps)]
        self.explore_prob_matrix = np.zeros((dim_X, dim_Y, N_learning_steps))
        for idx_x in range(dim_X):
            for idx_y in range(dim_Y):
                self.explore_prob_matrix[idx_x, idx_y, :] = np.array(EPSILON_list)

    def plot_learning_parameters(self):
        # Plot learning rate
        plt.figure(1, figsize=(16, 5))
        plt.plot(self.learning_rate_matrix[0, 0, 0, 0, :], label='learning rate')
        plt.xlabel('step')
        plt.ylabel('Learning Rate')
        plt.show()

        # Plot exploration probability
        plt.figure(2, figsize=(16, 5))
        plt.plot(self.explore_prob_matrix[0, 0, :], label='exploration probability')
        plt.xlabel('step')
        plt.ylabel('Probability for Exploration')
        plt.show()

    def update(self, env,):
        for i in range(self.N_RL_iter):
            if i % 100 == 0:
                print(f"Iteration: {i}")
            # the transition from i to i+1
            self.Q_table_track[:,:,:,:,i] = self.Q_table[:,:,:,:]
            # main part of the Q-learning algorithm

            # Part 1: choose the action to do given the state at i-th time point
            p_a = 0 # quoted ask price
            p_b = 0 # quoted bid price
            #########
            # Make a list of the actions available from the current state
            idx_x = int(env.X_data) # all are integers, i.e., 0,1,2,...,dim_X-1
            idx_y = int(env.Y_data) # all are integers, i.e., 0,1,2,...,dim_Y-1


            count_xy = int(self.state_counter_matrix[idx_x,idx_y])
            self.state_counter_matrix[idx_x,idx_y] = count_xy+1

            EPSILON = self.explore_prob_matrix[ idx_x, idx_y, count_xy ]  #exploration probability for this state

            x = idx_x + 1 # midprice=x*tick_size/2 and x is in (1,2,...,dim_X)
            y = idx_y - env.N_Y # y = the true signed integer value of inventory

            if y == -env.N_Y: # then sell order is not allowed
                action_a_list = [dim_action_a-1] # do nothing for ask order
                action_b_list = env.prices[env.prices<x/2] # the action is exactly equal to the index
                # x/2 is because the middle price is on grid: 0, 1/2, 1, 3/2, 2, ...,
                # but the quoted price is on grid: 0,1,2,...
                p_a = dim_action_a-1

                if np.random.binomial(1, EPSILON) == 1:
                    p_b = np.random.choice(action_b_list)
                else:
                    Q_values_xy = self.Q_table[idx_x, idx_y, :, :][np.ix_( list(action_a_list), list(action_b_list) )]
                    idx_max_a, idx_max_b = np.where(Q_values_xy == Q_values_xy.max())
                    p_b = action_b_list[ idx_max_b[0] ]

            elif y == env.N_Y: # then buy order is not allowed
                action_a_list = env.prices[env.prices>x/2]
                action_b_list = [dim_action_b-1] # do nothing for buy order
                p_b = dim_action_b-1

                if np.random.binomial(1, EPSILON) == 1:
                    p_a = np.random.choice(action_a_list)
                else:
                    Q_values_xy = self.Q_table[idx_x, idx_y, :, :][np.ix_( list(action_a_list), list(action_b_list) )]
                    idx_max_a, idx_max_b = np.where(Q_values_xy == Q_values_xy.max())
                    p_a = action_a_list[ idx_max_a[0] ]


            else: # then both sell and buy orders are allowed
                action_a_list = env.prices[env.prices>x/2] # the action is exactly equal to the index
                action_b_list = env.prices[env.prices<x/2] # the action is exactly equal to the index
                if np.random.binomial(1, EPSILON) == 1:
                    p_a = np.random.choice(action_a_list)
                    p_b = np.random.choice(action_b_list)
                else:
                    Q_values_xy = self.Q_table[idx_x, idx_y, :, :][np.ix_( list(action_a_list), list(action_b_list) )]
                    idx_max_a, idx_max_b = np.where(Q_values_xy == Q_values_xy.max())
                    p_a = action_a_list[ idx_max_a[0] ] # Wrong(do not do this!!! the index list is different!!): indeed, we can directly use idx_max_b[0], because the action is exactly equal to the index
                    p_b = action_b_list[ idx_max_b[0] ]

            # the above will output p_a p_b, this action is together with the state at time i

            # then we update our counter for (state, action) pair
            count_s_a = int(self.state_action_counter_matrix[idx_x, idx_y, p_a, p_b])
            self.state_action_counter_matrix[idx_x, idx_y, p_a, p_b] = count_s_a+1


            # observe the reward and the next state

            reward, idx_x_i1, idx_y_i1, action_a_list, action_b_list = env.step(p_a, p_b, x, y, idx_x)

            # Update Q-table
            Q_values_xy = self.Q_table[idx_x_i1, idx_y_i1, :, :][np.ix_( action_a_list, action_b_list )]

            Q_value_max_new_i1 = Q_values_xy.max()

            Q_value_new = reward + self.GAMMA_Delta * Q_value_max_new_i1
            Q_value_old = self.Q_table[idx_x, idx_y, p_a, p_b]

            self.Q_table[idx_x, idx_y, p_a, p_b] = self.learning_rate_matrix[idx_x, idx_y, p_a, p_b, count_s_a] * (Q_value_new-Q_value_old) + Q_value_old

        self.plot_result()

    def plot_result(self,):
        plt.figure(1, figsize=(20, 8))
        M=self.N_RL_iter
        plt.plot(self.Q_table_track[0,2,1,3,:M])
        plt.plot(self.Q_table_track[0,2,2,3,:M])
        plt.plot(self.Q_table_track[0,2,0,3,:M])
        plt.plot(self.Q_table_track[0,2,3,3,:M])
        plt.xlabel('step')
        plt.ylabel('Q(s,a)')
        plt.show()

# Example usage
N_P = 2 # price grid dimension - 1 (because we start from 0)
N_Y = 1 # (inventory grid dimension - 1)/2 (because we allow both - and + and 0)
Delta = 0.1

dim_X = 2*N_P-1
dim_Y = 2*N_Y+1
dim_action_a = N_P+2
dim_action_b = N_P+2

agent = QLearningAgent(dim_X, dim_Y, dim_action_a, dim_action_b, Delta)
# agent.plot_learning_parameters()

env = MarketEnvironment(N_P, N_Y, dim_action_a, dim_action_b, Delta)
env.reset()

agent.update(env)
