import numpy as np

class MarketEnvironment:
    def __init__(self, dim_price_grid, bound_inventory, dim_action_ask_price, dim_action_buy_price, Delta):
        # basic dimensions and state space grid
        self.dim_price_grid = dim_price_grid  # S_P
        self.bound_inventory = bound_inventory # N_Y
        self.dim_midprice_grid = 2*self.dim_price_grid-1  # S_X
        self.dim_inventory_grid = 2*self.bound_inventory+1 # S_Y
        self.dim_action_ask_price = dim_action_ask_price
        self.dim_action_buy_price = dim_action_buy_price
        self.price_list = np.array(list(range(self.dim_price_grid+1)))
        self.Delta = Delta # time increment

        self.tick_size = 1/3
        self.order_size = 1
        
        self.Delta_0 = 0.1
        
        # The two parameters below: for a limit order at price level p, the probability that this order is executed is 
        # equal to A*exp(-k1*|p-midprice|)*Delta, where |p-midprice| is the absolute distance between p and midprice.
        self.k1 = 2
        self.A = np.exp(1)*(1/self.Delta_0)*0.4

        # Transition probability matrix
        self.trans_prob_Q_matrix_midprice_upper_offdiagonal_entries = np.zeros((self.dim_midprice_grid-1, self.dim_midprice_grid))
        self.trans_prob_Q_matrix_midprice_lower_offdiagonal_entries = np.zeros((self.dim_midprice_grid-1, self.dim_midprice_grid))
        self._generate_transition_matrix()
        
        # Initial data for state variables
        # self.N_RL_iter = 10000  # # total steps of Q-learning iteration
        self.reset()

    def _generate_transition_matrix(self, lambda_seq=None):
        if lambda_seq is None:
            # generate N random lambda value
            # lambda_seq = np.random.uniform(0, 1, self.dim_midprice_grid-1)
            lambda_seq = np.array((self.dim_midprice_grid-1)*[1/3])
            lambda_seq[0] = 0.5
            lambda_seq *= (1/self.Delta_0)

        self.lambda_upper_offdiagonal_entries = lambda_seq
        self.lambda_lower_offdiagonal_entries = np.flip(lambda_seq, axis=0)
        
        for i in range(self.dim_midprice_grid-1):
            self.trans_prob_Q_matrix_midprice_lower_offdiagonal_entries[i,i] = self.lambda_lower_offdiagonal_entries[i] 
            self.trans_prob_Q_matrix_midprice_lower_offdiagonal_entries[i,i+1] = -self.lambda_lower_offdiagonal_entries[i]
            self.trans_prob_Q_matrix_midprice_upper_offdiagonal_entries[i,i] = -self.lambda_upper_offdiagonal_entries[i] 
            self.trans_prob_Q_matrix_midprice_upper_offdiagonal_entries[i,i+1] = self.lambda_upper_offdiagonal_entries[i]

        self.trans_prob_Q_matrix_midprice = np.zeros((self.dim_midprice_grid, self.dim_midprice_grid))
        self.trans_prob_Q_matrix_midprice[0:(self.dim_midprice_grid-1), 0:self.dim_midprice_grid] += self.trans_prob_Q_matrix_midprice_upper_offdiagonal_entries
        self.trans_prob_Q_matrix_midprice[1:self.dim_midprice_grid, 0:self.dim_midprice_grid] += self.trans_prob_Q_matrix_midprice_lower_offdiagonal_entries
        
        self.trans_prob_matrix_midprice = np.identity(self.dim_midprice_grid) + self.Delta*self.trans_prob_Q_matrix_midprice

    # For a limit order at price level p, the probability that this order is executed is 
    # equal to A*exp(-k1*|p-midprice|)*Delta, where |p-midprice| is the absolute distance between p and midprice.
    def prob_executed(self, price_distance):
        return self.A*np.exp(-self.k1*price_distance)*self.Delta

    def reset(self, ):
        """Reset the environment to its initial state."""
        # self.midprice_data = np.zeros(self.N_RL_iter+1)
        # self.inventory_data = np.zeros(self.N_RL_iter+1)
        
        self.midprice_data = self.dim_midprice_grid//2
        self.inventory_data = self.dim_inventory_grid//2

    def step(self, idx_ask_price, idx_buy_price, midprice_integer, inventory):
        # inventory is in [-dim_inventory_grid, -dim_inventory_grid+1,...,-1,0,1,...,dim_inventory_grid-1,dim_inventory_grid]
        # midprice = midprice_integer*(tick_size/2)
        idx_midprice = midprice_integer - 1
        if midprice_integer == 1:
            midprice_next = midprice_integer + np.dot( np.random.multinomial(1, [ self.trans_prob_matrix_midprice[idx_midprice,idx_midprice],self.trans_prob_matrix_midprice[idx_midprice,idx_midprice+1] ]),
                            np.array([0,1]) )
        elif midprice_integer == self.dim_midprice_grid:
            midprice_next = midprice_integer + np.dot( np.random.multinomial(1, [ self.trans_prob_matrix_midprice[idx_midprice,idx_midprice-1],self.trans_prob_matrix_midprice[idx_midprice,idx_midprice] ]),
                            np.array([-1,0]) )
        else:
            midprice_next = midprice_integer + np.dot( np.random.multinomial(1, [ self.trans_prob_matrix_midprice[idx_midprice,idx_midprice-1],self.trans_prob_matrix_midprice[idx_midprice,idx_midprice],self.trans_prob_matrix_midprice[idx_midprice,idx_midprice+1] ]),
                            np.array([-1,0,1]) )

        # inventory inventory variable update: Bernouli RV to simulate if the ask or buy order is executed or not:
        prob_ask_order_filled = self.prob_executed( idx_ask_price*self.tick_size - midprice_integer*self.tick_size/2 )
        prob_buy_order_filled = self.prob_executed( midprice_integer*self.tick_size/2 - idx_buy_price*self.tick_size )

        ask_order_change = self.order_size if np.random.uniform() <= prob_ask_order_filled else 0
        buy_order_change = self.order_size if np.random.uniform() <= prob_buy_order_filled else 0

        if inventory < -self.bound_inventory + self.order_size: # inventory hits/below the lower bound, then no sell order is allowed
            ask_order_change = self.bound_inventory - np.abs(inventory)
        elif inventory > self.bound_inventory - self.order_size: # inventory hits/over the upper bound, then no buy order is allowed
            buy_order_change = self.bound_inventory - np.abs(inventory)
    

        inventory_next = inventory - ask_order_change + buy_order_change
        # translate back from midprice_integer,inventory values to midprice_integer,inventory index
        idx_midprice_next = int(midprice_next - 1)
        idx_inventory_next = int(inventory_next + self.bound_inventory)
        
        if inventory_next == -self.bound_inventory: # then sell order is not allowed
            action_ask_price_list = [self.dim_action_ask_price-1] # do nothing for ask order
            action_buy_price_list = self.price_list[self.price_list<midprice_next/2] # the action is exactly equal to the index

        elif inventory_next == self.bound_inventory: # then buy order is not allowed
            action_ask_price_list = self.price_list[self.price_list>midprice_next/2]
            action_buy_price_list = [self.dim_action_buy_price-1] # do nothing for buy order

        else: # then both sell and buy orders are allowed
            action_ask_price_list = self.price_list[self.price_list>midprice_next/2] # the action is exactly equal to the index
            action_buy_price_list = self.price_list[self.price_list<midprice_next/2] # the action is exactly equal to the index
        
        # update the data of state variable
        self.midprice_data = idx_midprice_next # all are integers, i.e., 0,1,2,...,dim_midprice_grid-1
        self.inventory_data = idx_inventory_next # all are integers, i.e., 0,1,2,...,dim_inventory_grid-1


        reward = ( -midprice_integer*self.tick_size/2+idx_ask_price*self.tick_size )*ask_order_change + ( midprice_integer*self.tick_size/2-idx_buy_price*self.tick_size )*buy_order_change # - (inventory**2)*Delta + (midprice_next-midprice_integer)*inventory

        return reward, idx_midprice_next, idx_inventory_next, action_ask_price_list, action_buy_price_list
           

# Example Usage:
# market_env = MarketEnvironment()