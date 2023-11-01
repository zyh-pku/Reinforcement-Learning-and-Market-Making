import numpy as np

class MarketEnvironment:
    def __init__(self, N_P, N_Y, dim_action_a, dim_action_b, Delta):
        # basic dimensions and state space grid
        self.N_P = N_P
        self.N_Y = N_Y
        self.dim_X = 2*self.N_P-1
        self.dim_Y = 2*self.N_Y+1
        self.dim_action_a = dim_action_a
        self.dim_action_b = dim_action_b
        self.prices = np.array(list(range(self.N_P+1)))
        self.Delta = Delta

        self.tick = 1/3
        self.k1 = 2
        self.p0 = 0.4
        self.A = np.exp(1)*(1/self.Delta)*self.p0
        

        # Transition probability matrix
        self.Q_X_i_i1 = np.zeros((self.dim_X-1, self.dim_X))
        self.Q_X_i1_i = np.zeros((self.dim_X-1, self.dim_X))
        self.lambda_i_i1 = (1/self.Delta)*np.array([0.5,1/3])
        self.lambda_i1_i = (1/self.Delta)*np.array([1/3,0.5])
        self._init_transition_probs()
        
        # Initial data for state variables
        self.N_RL_iter = 10000  # # total steps of Q-learning iteration
        self.reset()
        
    def _init_transition_probs(self):
        for i in range(self.dim_X-1):
            self.Q_X_i1_i[i,i] = self.lambda_i1_i[i]
            self.Q_X_i1_i[i,i+1] = -self.lambda_i1_i[i]
            self.Q_X_i_i1[i,i] = -self.lambda_i_i1[i]
            self.Q_X_i_i1[i,i+1] = self.lambda_i_i1[i]

        self.Q_X = np.zeros((self.dim_X, self.dim_X))
        self.Q_X[0:(self.dim_X-1), 0:self.dim_X] += self.Q_X_i_i1
        self.Q_X[1:self.dim_X, 0:self.dim_X] += self.Q_X_i1_i
        
        self.P_X = np.identity(self.dim_X) + self.Delta*self.Q_X

    def lambda_e(self, D):
        return np.exp(-self.k1*D)*self.A

    def reset(self, ):
        """Reset the environment to its initial state."""
        # self.X_data = np.zeros(self.N_RL_iter+1)
        # self.Y_data = np.zeros(self.N_RL_iter+1)

        
        self.X_data = self.dim_X//2
        self.Y_data = self.dim_Y//2

    def step(self, p_a, p_b, x, y, idx_x):
        if x == 1:
            x_i1 = x + np.dot( np.random.multinomial(1, [ self.P_X[idx_x,idx_x],self.P_X[idx_x,idx_x+1] ]),
                            np.array([0,1]) )
        elif x == self.dim_X:
            x_i1 = x + np.dot( np.random.multinomial(1, [ self.P_X[idx_x,idx_x-1],self.P_X[idx_x,idx_x] ]),
                            np.array([-1,0]) )
        else:
            x_i1 = x + np.dot( np.random.multinomial(1, [ self.P_X[idx_x,idx_x-1],self.P_X[idx_x,idx_x],self.P_X[idx_x,idx_x+1] ]),
                            np.array([-1,0,1]) )

        # y inventory variable update: Bernouli RV to simulate if the ask or buy order is executed or not:
        p_ask_fill = self.lambda_e( -x*self.tick/2+p_a*self.tick )*self.Delta
        p_buy_fill = self.lambda_e( x*self.tick/2-p_b*self.tick )*self.Delta

        dna = 1 if np.random.uniform() <= p_ask_fill else 0
        dnb = 1 if np.random.uniform() <= p_buy_fill else 0

        if y == -self.N_Y:
            dna = 0
        if y == self.N_Y:
            dnb = 0

        y_i1 = y - dna + dnb
        # translate back from x,y values to x,y index
        idx_x_i1 = int(x_i1 - 1)
        idx_y_i1 = int(y_i1 + self.N_Y)
        if y_i1 == -self.N_Y: # then sell order is not allowed
            action_a_list = [self.dim_action_a-1] # do nothing for ask order
            action_b_list = self.prices[self.prices<x_i1/2] # the action is exactly equal to the index

        elif y_i1 == self.N_Y: # then buy order is not allowed
            action_a_list = self.prices[self.prices>x_i1/2]
            action_b_list = [self.dim_action_b-1] # do nothing for buy order

        else: # then both sell and buy orders are allowed
            action_a_list = self.prices[self.prices>x_i1/2] # the action is exactly equal to the index
            action_b_list = self.prices[self.prices<x_i1/2] # the action is exactly equal to the index
        
        # update the data of state variable
        self.X_data = idx_x_i1 # all are integers, i.e., 0,1,2,...,dim_X-1
        self.Y_data = idx_y_i1 # all are integers, i.e., 0,1,2,...,dim_Y-1


        reward = ( -x*self.tick/2+p_a*self.tick )*dna + ( x*self.tick/2-p_b*self.tick )*dnb # - (y**2)*Delta + (x_i1-x)*y

        return reward, idx_x_i1, idx_y_i1, action_a_list, action_b_list


# Example Usage:
# market_env = MarketEnvironment()
