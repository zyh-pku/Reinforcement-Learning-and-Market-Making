
import numpy as np
import matplotlib.pyplot as plt
from market import MarketEnvironment

class QLearningAgent:

    def __init__(self, env, dim_midprice_grid, dim_inventory_grid, dim_action_ask_price, dim_action_buy_price, Delta=0.1,  \
        N_Bellman_iter = 100, Bellman_iter_threshold = 0.001, Bellman_iter_V_initial = 2.3, \
            V_RL_iter_threshold = 0.065, V_RL_iter_initial = 2.3, \
        Q_upper_bound=4., UCB=True,\
        bonus_coef_0=0.1, bonus_coef_1=1., ucb_H=50, \
        lr = 0.8, lr_exponent = 2, \
        exp = 1.0, exp0 = 0.8, exp_epoch = 100, \
        N_RL_iter=12*10**4, N_learning_steps=3*10**4):
        # env is the environment class object
        # dim_midprice_grid is the number of midprice levels
        # dim_inventory_grid is the number of inventory levels
        # dim_action_ask_price is the number of ask price levels
        # dim_action_buy_price is the number of buy price levels
        # V_RL_iter_initial is the initial value for the Q-table in the RL iteration

        # V_RL_iter_threshold is the threshold for stopping the RL iteration when the value function error is less than this threshold
        # lr , lr_exponent are the learning rate and the exponent for the learning rate schedule
        # exp, exp0, exp_epoch are the exploration probability, the initial exploration probability, and the epoch for the exploration probability schedule
        # N_RL_iter is the total number of RL iterations
        # N_learning_steps is the total number of learning steps for each (state, action) pair
        # Delta is the time increment
        # GAMMA is the discount factor
        # GAMMA_Delta is the discount factor computed with the time increment Delta
        # Q_upper_bound is the upper bound for the Q-table, and is only used when UCB=True
        
        # N_Bellman_iter is the total number of Bellman equation iterations for finding the true value function and the optimal policy
        # Bellman_iter_threshold is the threshold for stopping the Bellman equation iteration
        # Bellman_iter_V_initial is the initial value for the value function in the Bellman equation iteration

        self.env = env
        self.dim_midprice_grid = dim_midprice_grid
        self.dim_inventory_grid = dim_inventory_grid
        self.dim_action_ask_price = dim_action_ask_price
        self.dim_action_buy_price = dim_action_buy_price
        self.N_RL_iter = N_RL_iter
        self.N_learning_steps = N_learning_steps
        self.Delta = Delta
        self.GAMMA = 0.95
        self.GAMMA_Delta = np.exp(-self.GAMMA*self.Delta)
        self.Q_upper_bound = Q_upper_bound

        
        # true value function and optimal policy obtained by Bellman equation iteration
        self.N_Bellman_iter = N_Bellman_iter
        self.V_star = Bellman_iter_V_initial+np.zeros((self.dim_midprice_grid, self.dim_inventory_grid))
        self.V_star_converge_track = Bellman_iter_V_initial+np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.N_Bellman_iter+1))
        self.ask_price_star = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid))
        self.buy_price_star = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid))
        self.Bellman_iter_threshold = Bellman_iter_threshold
        self.Bellman_iter_steps_converge = 0
        
        # value function error iteration steps
        self.V_RL_iter_steps = N_RL_iter
        self.V_RL_iter_threshold = V_RL_iter_threshold
        self.V_error_track = np.zeros( N_RL_iter )
        self.V_error_full = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid))
        self.V_error = 10        
        # value function error local setup
        
        # ucb
        self.UCB = UCB  # if True , use UCB exploration; if False, use eps-greedy exploration
        self.bonus_coef_0 = bonus_coef_0
        self.bonus_coef_1 = bonus_coef_1
        self.ucb_H = ucb_H

        # eps-greedy
        self.exp = exp  # exploration probability
        self.exp0 = exp0
        self.exp_epoch = exp_epoch
        self.exp_smallest = 0.00001 
        
        # learning-rate
        self.lr = lr
        # self.eps0 = eps0 
        self.lr_exponent = lr_exponent
        

        # Initialize Q-table and other matrices (for vanilla Q-learning with eps-greedy exploration)
        self.Q_table = V_RL_iter_initial + np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price))
        self.Q_table_track = V_RL_iter_initial + np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price, N_RL_iter))
        self.state_counter_matrix = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid))
        self.state_action_counter_matrix = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price))

        # Initialize an additional Q_hat-table for Q-learning with UCB exploration
        self.Q_hat_table = V_RL_iter_initial + np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price) ) + self.Q_upper_bound
        self.Q_hat_table_track = V_RL_iter_initial + np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price, N_RL_iter) ) + self.Q_upper_bound

        self.set_learning_rate()
        
        # comment the two below because we already have the true values and policies
        self._find_V_star_and_optimal_policy() 
        # self.print_true_values_and_plot_Bellman_iteration()

        if self.UCB:
            # Define bonus rate (for Q-learning with UCB exploration)
            bonus_list = [np.sqrt( (self.bonus_coef_1 * np.log(i+2) + self.bonus_coef_0 )/(i+1) ) for i in range(N_learning_steps)]
            self.bonus_matrix = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price, N_learning_steps))
            for idx_midprice in range(self.dim_midprice_grid):
                for idx_inventory in range(self.dim_inventory_grid):
                    for idx_ask_price in range(self.dim_action_ask_price):
                        for idx_buy_price in range(self.dim_action_buy_price):
                            self.bonus_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price, :] = np.array(bonus_list)

        else:
            # Define exploration probability (for vanilla Q-learning with eps-greedy exploration)
            explore_epsilon_list = [self.exp*((self.exp0)**(i//self.exp_epoch)) for i in range(N_learning_steps)]
            self.explore_prob_matrix = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, N_learning_steps))
            for idx_midprice in range(self.dim_midprice_grid):
                for idx_inventory in range(self.dim_inventory_grid):
                    self.explore_prob_matrix[idx_midprice, idx_inventory, :] = np.array(explore_epsilon_list)

    def set_learning_rate(self, ):
        if self.UCB:
            # Define learning rate (for Q-learning with UCB exploration)
            learning_rate_schedule = [(self.ucb_H+1)/(self.ucb_H+i) for i in range(self.N_learning_steps)]
        else:
            # Define learning rate (for vanilla Q-learning with eps-greedy exploration)
            # learning_rate_schedule = [self.lr*((self.eps0)**(i//self.lr_exponent)) for i in range(self.N_learning_steps)]
            learning_rate_schedule = [self.lr*1/((i+1)**self.lr_exponent) for i in range(self.N_learning_steps)]

        self.learning_rate_matrix = np.zeros((self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price, self.N_learning_steps))
        for idx_midprice in range(self.dim_midprice_grid):
            for idx_inventory in range(self.dim_inventory_grid):
                for idx_ask_price in range(self.dim_action_ask_price):
                    for idx_buy_price in range(self.dim_action_buy_price):
                        self.learning_rate_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price, :] = np.array(learning_rate_schedule)


    def update(self, ):
        if self.UCB:
            self.Q_table = np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid, self.dim_action_ask_price, self.dim_action_buy_price) ) + self.Q_upper_bound

        for i in range(self.N_RL_iter):
            ###### compute the steps such that value function error is less than the threshold:
            V_RL = np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid ) )

            for idx_midprice in range(self.dim_midprice_grid):
                for idx_inventory in range(self.dim_inventory_grid):
                    V_RL[idx_midprice,idx_inventory]=np.max(self.Q_table[idx_midprice,idx_inventory,:,:])
            # compute the value function error at the current step:
            self.V_error = np.max( abs( V_RL - self.V_star ) )
            self.V_error_track[i] = self.V_error
            if self.V_error < self.V_RL_iter_threshold:
                self.V_error_full = V_RL - self.V_star
                self.V_RL_iter_steps = i
                break
            ######
            if i % (10**4) == 0:
                print(f"Iteration: {i}")
                print(f"V error: {self.V_error}")
            # the transition from i to i+1
            self.Q_table_track[:,:,:,:,i] = self.Q_table[:,:,:,:]
            self.Q_hat_table_track[:,:,:,:,i] = self.Q_hat_table[:,:,:,:]
            # main part of the Q-learning algorithm

            # Part 1: choose the action to do given the state at i-th time point
            idx_ask_price = 0 # quoted ask price
            idx_buy_price = 0 # quoted bid price
            #########
            # Make a list of the actions available from the current state
            idx_midprice = int(self.env.midprice_data) # all are integers, i.e., 0,1,2,...,self.dim_midprice_grid-1
            idx_inventory = int(self.env.inventory_data) # all are integers, i.e., 0,1,2,...,self.dim_inventory_grid-1
            count_state = int(self.state_counter_matrix[idx_midprice, idx_inventory])

            self.state_counter_matrix[idx_midprice,idx_inventory] = count_state+1
            
            if not self.UCB:
                EPSILON = self.explore_prob_matrix[ idx_midprice, idx_inventory, count_state ]  #exploration probability for this state
                EPSILON = max(EPSILON, self.exp_smallest) # set a smallest epsilon for exploration probability

            midprice_integer = idx_midprice + 1 # midprice=midprice_integer*tick_size/2 and midprice_integer is in (1,2,...,self.dim_midprice_grid)
            inventory = idx_inventory - self.env.bound_inventory # inventory = the true signed integer value of inventory

            if inventory == -self.env.bound_inventory: # then sell order is not allowed
                action_ask_price_list = [self.dim_action_ask_price-1] # do nothing for ask order
                action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                # midprice_integer/2 is because the middle price is on grid: 0, 1/2, 1, 3/2, 2, ...,
                # but the quoted price is on grid: 0,1,2,...
                idx_ask_price = self.dim_action_ask_price-1

                if not self.UCB and np.random.binomial(1, EPSILON) == 1:
                    idx_buy_price = np.random.choice(action_buy_price_list)
                else:
                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( list(action_ask_price_list), list(action_buy_price_list) )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_buy_price = action_buy_price_list[ idx_optimal_buy[0] ]

            elif inventory == self.env.bound_inventory: # then buy order is not allowed
                action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2]
                action_buy_price_list = [self.dim_action_buy_price-1] # do nothing for buy order
                idx_buy_price = self.dim_action_buy_price-1

                if not self.UCB and np.random.binomial(1, EPSILON) == 1:
                    idx_ask_price = np.random.choice(action_ask_price_list)
                else:
                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( list(action_ask_price_list), list(action_buy_price_list) )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_ask_price = action_ask_price_list[ idx_optimal_ask[0] ]


            else: # then both sell and buy orders are allowed
                action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2] # the action is exactly equal to the index
                action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                if not self.UCB and np.random.binomial(1, EPSILON) == 1:
                    idx_ask_price = np.random.choice(action_ask_price_list)
                    idx_buy_price = np.random.choice(action_buy_price_list)
                else:
                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( list(action_ask_price_list), list(action_buy_price_list) )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_ask_price = action_ask_price_list[ idx_optimal_ask[0] ] # Wrong(do not do this!!! the index list is different!!): indeed, we can directly use idx_optimal_buy[0], because the action is exactly equal to the index
                    idx_buy_price = action_buy_price_list[ idx_optimal_buy[0] ]

            # the above will output idx_ask_price idx_buy_price, this action is together with the state at time i

            # then we update our counter for (state, action) pair
            count_state_action = int(self.state_action_counter_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price])
            self.state_action_counter_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price] = count_state_action+1


            # observe the reward and the next state

            reward, idx_midprice_next, idx_inventory_next, action_ask_price_list, action_buy_price_list = self.env.step(idx_ask_price, idx_buy_price, midprice_integer, inventory)

            Q_values_at_state = self.Q_table[idx_midprice_next, idx_inventory_next, :, :][np.ix_( action_ask_price_list, action_buy_price_list )]

            # Update Q-table
            if self.UCB:

                Q_value_new = reward + self.GAMMA_Delta * Q_values_at_state.max() + self.bonus_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price, count_state_action]
                Q_value_old = self.Q_hat_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price]

                self.Q_hat_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price] = self.learning_rate_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price, count_state_action] * (Q_value_new-Q_value_old) + Q_value_old

                self.Q_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price] = min(self.Q_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price], self.Q_hat_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price])

            else:

                Q_value_new = reward + self.GAMMA_Delta * Q_values_at_state.max()
                Q_value_old = self.Q_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price]

                self.Q_table[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price] = self.learning_rate_matrix[idx_midprice, idx_inventory, idx_ask_price, idx_buy_price, count_state_action] * (Q_value_new-Q_value_old) + Q_value_old


        # self.plot_result()
    def plot_V_error_result(self,):
        M=self.V_RL_iter_steps
        plt.plot(self.V_error_track[:M])
        plt.xlabel('step')
        plt.ylabel('value error')
        plt.show()    
        
    def plot_result(self,):
        plt.figure(1, figsize=(20, 8))
        M=self.N_RL_iter
        plt.plot(self.Q_table_track[0,2,1,3,:M], label='a=1')
        plt.plot(self.Q_table_track[0,2,2,3,:M], label='a=2')
        plt.plot(self.Q_table_track[0,2,0,3,:M])
        plt.plot(self.Q_table_track[0,2,3,3,:M])
        plt.xlabel('step')
        plt.ylabel('Q(s,a)')
        plt.show()

        if self.UCB:
            plt.figure(2, figsize=(20, 8))
            M=self.N_RL_iter
            plt.plot(self.Q_hat_table_track[0,2,1,3,:M], label='a=1')
            plt.plot(self.Q_hat_table_track[0,2,2,3,:M], label='a=2')
            plt.plot(self.Q_hat_table_track[0,2,0,3,:M])
            plt.plot(self.Q_hat_table_track[0,2,3,3,:M])
            plt.xlabel('step')
            plt.ylabel('Q_hat(s,a)')
            plt.show()

    def plot_learning_parameters(self):
        if self.UCB:
            label_prefix = 'UCB'
        else:
            label_prefix = 'epsilon-greedy'
        # Plot learning rate
        plt.figure(1, figsize=(16, 5))
        plt.plot(self.learning_rate_matrix[0, 0, 0, 0, :], label=f'{label_prefix}: learning rate')
        plt.xlabel('step')
        plt.ylabel('Learning Rate')
        plt.show()

        if self.UCB:
            # Plot bonus
            plt.figure(2, figsize=(16, 5))
            plt.plot(self.bonus_matrix[0, 0, 0, 0, :], label='bonus')
            plt.xlabel('step')
            plt.ylabel('Bonus')
            plt.show()
        else:
            # Plot exploration probability
            plt.figure(2, figsize=(16, 5))
            plt.plot(self.explore_prob_matrix[0, 0, :], label='exploration probability')
            plt.xlabel('step')
            plt.ylabel('Probability for Exploration')
            plt.show()
       
    def _find_V_star_and_optimal_policy(self, ):
        # this function is to find the true value function (self.V_star) and the optimal policy (self.ask_price_star,self.buy_price_star) by Bellman equation iteration
        # the reward function here must be align with the reward function in the function step() in market.py
        
        V_max_increment = float('inf')
        i = -1
        while V_max_increment > self.Bellman_iter_threshold and i < self.N_Bellman_iter-1:
            i+=1
            self.V_star = self.V_star_converge_track[ : , : , i]
            V_max_increment = 0
            for idx_midprice in range(self.dim_midprice_grid):
                for idx_inventory in range(self.dim_inventory_grid):
                    midprice_integer = int(idx_midprice + 1)
                    inventory = int(idx_inventory - self.env.bound_inventory)
                    
                    V_max_at_this_state = -float('inf') # 
                    
                    if inventory == -self.env.bound_inventory: # then sell order is not allowed
                        action_ask_price_list = [self.dim_action_ask_price-1] # do nothing for ask order
                        action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                        self.ask_price_star[idx_midprice , idx_inventory] = action_ask_price_list[0]
                        for idx_buy_price in action_buy_price_list:
                            prob_buy_order_filled = self.env.prob_executed( midprice_integer*self.env.tick_size/2 - idx_buy_price*self.env.tick_size )  

                            prob_price = self.env.trans_prob_matrix_midprice[idx_midprice, ] # probability distribution vector

                            prob_inventory = np.zeros( self.dim_inventory_grid ) # probability distribution vector
                            prob_inventory[idx_inventory + 1] = prob_buy_order_filled
                            prob_inventory[idx_inventory] = 1 - prob_buy_order_filled

                            midprice_integer_expectation = np.dot(prob_price, np.array(range(1,self.dim_midprice_grid+1)))

                            V_tmp = prob_buy_order_filled * (midprice_integer*self.env.tick_size/2 - idx_buy_price*self.env.tick_size  - self.env.transaction_cost) + \
                            (midprice_integer_expectation-midprice_integer)*(self.env.tick_size/2)*inventory - self.env.phi_inventory_risk*(inventory**2)*self.Delta + \
                            self.GAMMA_Delta * np.dot(prob_price.T ,np.dot(self.V_star, prob_inventory)) # the np.dot is doing the expectation

                            if V_tmp > V_max_at_this_state:
                                V_max_at_this_state = V_tmp
                                self.buy_price_star[idx_midprice , idx_inventory] = idx_buy_price
                                
    
                    elif inventory == self.env.bound_inventory: # then buy order is not allowed
                        action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2]
                        action_buy_price_list = [self.dim_action_buy_price-1] # do nothing for buy order
                        self.buy_price_star[idx_midprice , idx_inventory] = action_buy_price_list[0]
                        for idx_ask_price in action_ask_price_list:
                            prob_ask_order_filled = self.env.prob_executed( idx_ask_price*self.env.tick_size - midprice_integer*self.env.tick_size/2 )
                            
                            prob_price = self.env.trans_prob_matrix_midprice[idx_midprice, ] # probability distribution vector

                            prob_inventory = np.zeros( self.env.dim_inventory_grid ) # probability distribution vector
                            prob_inventory[idx_inventory] = 1 - prob_ask_order_filled
                            prob_inventory[idx_inventory - 1] = prob_ask_order_filled

                            midprice_integer_expectation = np.dot(prob_price, np.array(range(1,self.dim_midprice_grid+1)))
   
                            V_tmp = prob_ask_order_filled * (idx_ask_price*self.env.tick_size - midprice_integer*self.env.tick_size/2  - self.env.transaction_cost) + \
                            (midprice_integer_expectation-midprice_integer)*(self.env.tick_size/2)*inventory - self.env.phi_inventory_risk*(inventory**2)*self.Delta + \
                            self.GAMMA_Delta * np.dot(prob_price.T ,np.dot(self.V_star, prob_inventory)) # the np.dot is doing the expectation

                            if V_tmp > V_max_at_this_state:
                                V_max_at_this_state = V_tmp
                                self.ask_price_star[idx_midprice , idx_inventory] = idx_ask_price
                                
                    else: # then both sell and buy orders are allowed
                    
                        action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2] # the action is exactly equal to the index
                        action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                        
                        for idx_ask_price in action_ask_price_list:
                            for idx_buy_price in action_buy_price_list:
                                prob_ask_order_filled = self.env.prob_executed( idx_ask_price*self.env.tick_size - midprice_integer*self.env.tick_size/2 )
                                prob_buy_order_filled = self.env.prob_executed( midprice_integer*self.env.tick_size/2 - idx_buy_price*self.env.tick_size )  
    
                                prob_price = self.env.trans_prob_matrix_midprice[idx_midprice, ] # probability distribution vector
    
                                prob_inventory = np.zeros( self.dim_inventory_grid ) # probability distribution vector
                                prob_inventory[idx_inventory + 1] = ( 1 - prob_ask_order_filled )*prob_buy_order_filled
                                prob_inventory[idx_inventory] = ( 1 - prob_ask_order_filled )*( 1 - prob_buy_order_filled ) + prob_ask_order_filled * prob_buy_order_filled
                                prob_inventory[idx_inventory - 1] = ( 1 - prob_buy_order_filled )*prob_ask_order_filled

                                midprice_integer_expectation = np.dot(prob_price, np.array(range(1,self.dim_midprice_grid+1)))
    
                                V_tmp = prob_ask_order_filled * (idx_ask_price*self.env.tick_size - midprice_integer*self.env.tick_size/2 - self.env.transaction_cost) + \
                                prob_buy_order_filled * (midprice_integer*self.env.tick_size/2 - idx_buy_price*self.env.tick_size - self.env.transaction_cost) + \
                                (midprice_integer_expectation-midprice_integer)*(self.env.tick_size/2)*inventory - self.env.phi_inventory_risk*(inventory**2)*self.Delta + \
                                self.GAMMA_Delta * np.dot(prob_price.T ,np.dot(self.V_star, prob_inventory)) # the np.dot is doing the expectation
    
                                if V_tmp > V_max_at_this_state:
                                    V_max_at_this_state = V_tmp
                                    self.ask_price_star[idx_midprice , idx_inventory] = idx_ask_price
                                    self.buy_price_star[idx_midprice , idx_inventory] = idx_buy_price
                                    
                    self.V_star_converge_track[idx_midprice , idx_inventory, i+1] = V_max_at_this_state
                    V_max_increment = max(V_max_increment, abs(V_max_at_this_state-self.V_star[idx_midprice , idx_inventory]))
                    # print(f"V_max_increment: {V_max_increment}")
        self.Bellman_iter_steps_converge = i
            
    def print_true_values_and_plot_Bellman_iteration(self, ):                                    
        fig, axs = plt.subplots(self.dim_midprice_grid, self.dim_inventory_grid, figsize=(15, 15))

        global_min = np.min(self.V_star_converge_track)
        global_max = np.max(self.V_star_converge_track)
        
        for idx_midprice in range(self.dim_midprice_grid):
            for idx_inventory in range(self.dim_inventory_grid):
                axs[idx_midprice, idx_inventory].plot(self.V_star_converge_track[idx_midprice, idx_inventory, 0:self.Bellman_iter_steps_converge+1])
                
                axs[idx_midprice, idx_inventory].set_title(f"(price,inventory)=({idx_midprice},{idx_inventory})")
                axs[idx_midprice, idx_inventory].set_xlabel("Step")
                axs[idx_midprice, idx_inventory].set_ylabel("Value")
                axs[idx_midprice, idx_inventory].set_ylim(global_min-0.05, global_max+0.05)
        
        plt.tight_layout()
        plt.show()
        print(self.Bellman_iter_steps_converge)
        print('---------- true optimal value function V_star: ----------')
        print(self.V_star)
        print('---------- true optimal ask price: ----------')
        print(self.ask_price_star)
        print('---------- true optimal buy price: ----------')
        print(self.buy_price_star)
                                    
        
    def result_metrics(self, ):
        '''
        
        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
         : integer
            the number of wrong policies. It is equal to the sum of wrong ask price and wrong buy price.
            (the smaller it is, the better the RL algo is).

        '''
        # print('---------- the visiting number for each state: ----------')
        # print(self.state_counter_matrix)
        action_ask_price_RL = np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid) )
        action_buy_price_RL = np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid) )
        V_RL = np.zeros( (self.dim_midprice_grid, self.dim_inventory_grid) )
        # print('---------- the Q function for each state: ----------')
        for idx_midprice in range(self.dim_midprice_grid):
            for idx_inventory in range(self.dim_inventory_grid):
                midprice_integer = int(idx_midprice + 1)
                inventory = int(idx_inventory - self.env.bound_inventory)
                if inventory == -self.env.bound_inventory: # then sell order is not allowed
                    action_ask_price_list = [self.dim_action_ask_price-1] # do nothing for ask order
                    action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                    # print( f'(midprice_integer,inventory)={midprice_integer},{inventory}' )
                    # print(action_ask_price_list)
                    # print(action_buy_price_list)
                    # #print( state_action_counter_matrix[ idx_midprice, idx_inventory, action_ask_price_list, action_buy_price_list] )
                    # print( self.state_action_counter_matrix[ idx_midprice, idx_inventory, :, :] )
                    # print( self.Q_table[ idx_midprice, idx_inventory, :, :] )

                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( action_ask_price_list, action_buy_price_list )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_ask_price = self.dim_action_ask_price-1
                    idx_buy_price = action_buy_price_list[ idx_optimal_buy[0] ]

                elif inventory == self.env.bound_inventory: # then buy order is not allowed
                    action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2]
                    action_buy_price_list = [self.dim_action_buy_price-1] # do nothing for buy order
                    # print( f'(midprice_integer,inventory)={midprice_integer},{inventory}' )
                    # print(action_ask_price_list)
                    # print(action_buy_price_list)
                    # #print( state_action_counter_matrix[ idx_midprice, idx_inventory, action_ask_price_list, action_buy_price_list] )
                    # print( self.state_action_counter_matrix[ idx_midprice, idx_inventory, :, :] )
                    # print( self.Q_table[ idx_midprice, idx_inventory, :, :] )

                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( action_ask_price_list, action_buy_price_list )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_ask_price = action_ask_price_list[ idx_optimal_ask[0] ]
                    idx_buy_price = self.dim_action_buy_price-1

                else: # then both sell and buy orders are allowed
                    action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2] # the action is exactly equal to the index
                    action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
                    # print( f'(midprice_integer,inventory)={midprice_integer},{inventory}' )
                    # print(action_ask_price_list)
                    # print(action_buy_price_list)
                    # #print( state_action_counter_matrix[ idx_midprice, idx_inventory, action_ask_price_list, action_buy_price_list] )
                    # print( self.state_action_counter_matrix[ idx_midprice, idx_inventory, :, :] )
                    # print( self.Q_table[ idx_midprice, idx_inventory, :, :] )

                    Q_values_at_state = self.Q_table[idx_midprice, idx_inventory, :, :][np.ix_( action_ask_price_list, action_buy_price_list )]
                    idx_optimal_ask, idx_optimal_buy = np.where(Q_values_at_state == Q_values_at_state.max())
                    idx_ask_price = action_ask_price_list[ idx_optimal_ask[0] ]
                    idx_buy_price = action_buy_price_list[ idx_optimal_buy[0] ]

                V_RL[idx_midprice, idx_inventory] = Q_values_at_state.max()
                action_ask_price_RL[idx_midprice, idx_inventory] = idx_ask_price
                action_buy_price_RL[idx_midprice, idx_inventory] = idx_buy_price
        # print('---------- the learned value function and policy: ----------')
        # print(V_RL)
        # print(action_ask_price_RL)
        # print(action_buy_price_RL)
        # compute value function error

        #print(V_RL)
        #print(self.V_star)
        #self.V_error = np.max( abs( V_RL - self.V_star ) )
        return (action_ask_price_RL!=self.ask_price_star).sum()+(action_buy_price_RL!=self.buy_price_star).sum()
















