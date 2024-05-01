import numpy as np
from market_2 import MarketEnvironment
from Nash_solver import get_action_from_nash, get_nash_value, get_policy_from_nash

class NashQLearningAgent:
    def __init__(self, env, dim_price_grid, Delta, 
                 N_learning_steps, N_RL_iter, V_RL_iter_initial, 
                 lr0, lr, lr_epoch, exp0, exp, exp_epoch, solver='LP'):
        
        self.dim_price_grid = dim_price_grid
        self.dim_midprice_grid = 2 * dim_price_grid - 1
        dim_action_ask_price = dim_price_grid+2
        dim_action_buy_price = dim_price_grid+2
        self.dim_action_ask_price = dim_action_ask_price
        self.dim_action_buy_price = dim_action_buy_price
        self.Delta = Delta
        self.env = env
        self.GAMMA = 0.95
        self.GAMMA_Delta = np.exp(-0.95 * Delta)
        self.N_learning_steps = N_learning_steps
        self.N_RL_iter = N_RL_iter
        self.Q_table_1 = dict()
        self.Q_table_2 = dict()
        self.Q_table_track_1 = dict()
        self.Q_table_track_2 = dict()
        self.state_counter_matrix = np.zeros(self.dim_midprice_grid)
        self.state_action_counter_matrix = dict()

        self.lr0 = lr0
        self.lr = lr
        self.lr_epoch = lr_epoch
        self.learning_rate_list = np.array([self.lr0 * ((self.lr) ** (i // self.lr_epoch)) for i in range(self.N_learning_steps)])
        self.learning_rate_matrix = dict()
        for idx_midprice in range(self.dim_midprice_grid):
            midprice_integer = idx_midprice + 1 # midprice=midprice_integer*tick_size/2 and midprice_integer is in (1,2,...,self.dim_midprice_grid)
            action_ask_price_list = self.env.price_list[self.env.price_list>midprice_integer/2] # the action is exactly equal to the index
            action_buy_price_list = self.env.price_list[self.env.price_list<midprice_integer/2] # the action is exactly equal to the index
            dim_action = len(action_ask_price_list)*len(action_buy_price_list)
            self.Q_table_1[midprice_integer] = np.zeros((dim_action, dim_action)) + V_RL_iter_initial
            self.Q_table_2[midprice_integer] = np.zeros((dim_action, dim_action)) + V_RL_iter_initial
            self.Q_table_track_1[midprice_integer] = np.zeros((dim_action, dim_action, self.N_learning_steps)) + V_RL_iter_initial
            self.Q_table_track_2[midprice_integer] = np.zeros((dim_action, dim_action, self.N_learning_steps)) + V_RL_iter_initial
            
            learning_rate_vector = self.learning_rate_list.reshape((1,1,self.N_learning_steps))         
            self.learning_rate_matrix[midprice_integer] = np.zeros((dim_action, dim_action, self.N_learning_steps))
            self.learning_rate_matrix[midprice_integer][:] = learning_rate_vector

            self.state_action_counter_matrix[midprice_integer] = np.zeros((dim_action, dim_action))
        self.exp0 = exp0
        self.exp = exp
        self.exp_epoch = exp_epoch
        self.explore_prob_list = np.array([self.exp0 * ((self.exp) ** (i // self.exp_epoch)) for i in range(self.N_learning_steps)])
        self.explore_prob_matrix = np.zeros((self.dim_midprice_grid, self.N_learning_steps))
        self.explore_prob_matrix[:] = self.explore_prob_list
        self.exp_smallest = 0.00001

        self.solver = solver

        self.V_learned_1 = np.zeros(self.dim_midprice_grid)
        self.V_learned_2 = np.zeros(self.dim_midprice_grid)
        self.policy_learned_1 = dict()
        self.policy_learned_2 = dict()

        # # Bellman iteration # we comment the below since we directly use the true values computed offline
        # self.Bellman_iter_threshold = 1e-4
        # self.N_Bellman_iter = 100
        # self.V_true_1 = np.zeros(self.dim_midprice_grid)
        # self.V_true_2 = np.zeros(self.dim_midprice_grid)
        # self.policy_true_1 = dict()
        # self.policy_true_2 = dict()
        # self.compute_true_nash_equilibrium()
        self.V_true_1 = np.array([3.81355605, 3.85030124, 3.81355605])
        self.V_true_2 = np.array([3.81355605, 3.85030124, 3.81355605])
        self.policy_true_1 ={1: np.array([[0.],[1.]]),
                            2: np.array([[1.]]),
                            3: np.array([[1., 0.]])}
        self.policy_true_2 ={1: np.array([[0.],[1.]]),
                            2: np.array([[1.]]),
                            3: np.array([[1., 0.]])}
        self.V_error_track_1 = np.zeros(self.N_RL_iter)
        self.V_error_track_2 = np.zeros(self.N_RL_iter)
        self.policy_error_track_1 = np.zeros(self.N_RL_iter)
        self.policy_error_track_2 = np.zeros(self.N_RL_iter)


    def update(self):
        self.env.reset()
        for i in range(self.N_RL_iter):
            idx_midprice = self.env.midprice_data
            midprice_integer = idx_midprice + 1
            action_ask_price_list = self.env.price_list[self.env.price_list > midprice_integer / 2]
            action_buy_price_list = self.env.price_list[self.env.price_list < midprice_integer / 2]
            count_state = int(self.state_counter_matrix[idx_midprice])
            self.state_counter_matrix[idx_midprice] += 1
            EPSILON = self.explore_prob_matrix[idx_midprice, count_state]
            EPSILON = max(EPSILON, self.exp_smallest)

            if np.random.uniform() < EPSILON:
                idx_ask_price_1 = np.random.randint(len(action_ask_price_list))
                idx_buy_price_1 = np.random.randint(len(action_buy_price_list))
                idx_ask_price_2 = np.random.randint(len(action_ask_price_list))
                idx_buy_price_2 = np.random.randint(len(action_buy_price_list))
                idx_optimal_1 = idx_ask_price_1 * len(action_buy_price_list) + idx_buy_price_1
                idx_optimal_2 = idx_ask_price_2 * len(action_buy_price_list) + idx_buy_price_2
                idx_ask_price_1 = action_ask_price_list[idx_ask_price_1]
                idx_buy_price_1 = action_buy_price_list[idx_buy_price_1]
                idx_ask_price_2 = action_ask_price_list[idx_ask_price_2]
                idx_buy_price_2 = action_buy_price_list[idx_buy_price_2]
            else:
                idx_optimal_1, idx_optimal_2 = get_action_from_nash(self.Q_table_1[midprice_integer], self.Q_table_2[midprice_integer], self.solver)
                idx_ask_price_1 = action_ask_price_list[idx_optimal_1 // len(action_buy_price_list)]
                idx_buy_price_1 = action_buy_price_list[idx_optimal_1 % len(action_buy_price_list)]
                idx_ask_price_2 = action_ask_price_list[idx_optimal_2 // len(action_buy_price_list)]
                idx_buy_price_2 = action_buy_price_list[idx_optimal_2 % len(action_buy_price_list)]

            count_state_action = int(self.state_action_counter_matrix[midprice_integer][idx_optimal_1, idx_optimal_2])
            self.state_action_counter_matrix[midprice_integer][idx_optimal_1, idx_optimal_2] = count_state_action + 1

            idx_midprice_next, reward_1, reward_2 = self.env.step(midprice_integer, idx_ask_price_1, idx_buy_price_1, idx_ask_price_2, idx_buy_price_2)
            midprice_integer_next = idx_midprice_next + 1
            Q_new_1, Q_new_2 = get_nash_value(self.Q_table_1[midprice_integer_next], self.Q_table_2[midprice_integer_next], self.solver)
            Q_old_1, Q_old_2 = self.Q_table_1[midprice_integer][idx_optimal_1, idx_optimal_2], self.Q_table_2[midprice_integer][idx_optimal_2, idx_optimal_1]
            self.Q_table_1[midprice_integer][idx_optimal_1, idx_optimal_2] = Q_old_1 + (reward_1 + self.GAMMA_Delta * Q_new_1 - Q_old_1) * self.learning_rate_matrix[midprice_integer][idx_optimal_1, idx_optimal_2, count_state_action]
            self.Q_table_2[midprice_integer][idx_optimal_2, idx_optimal_1] = Q_old_2 + (reward_2 + self.GAMMA_Delta * Q_new_2 - Q_old_2) * self.learning_rate_matrix[midprice_integer][idx_optimal_1, idx_optimal_2, count_state_action]
            self.Q_table_track_1[midprice_integer][:, :, count_state_action] = self.Q_table_1[midprice_integer][:, :]
            self.Q_table_track_2[midprice_integer][:, :, count_state_action] = self.Q_table_2[midprice_integer][:, :]

            ### track the error
            self.V_error_track_1[i], self.V_error_track_2[i], self.policy_error_track_1[i], self.policy_error_track_2[i] = self.result_metrics()

    def result_metrics(self):
        policy_error_1 = 0
        policy_error_2 = 0
        for idx_midprice in range(self.dim_midprice_grid):
            midprice_integer = idx_midprice + 1
            action_ask_price_list = self.env.price_list[self.env.price_list > midprice_integer / 2]
            action_buy_price_list = self.env.price_list[self.env.price_list < midprice_integer / 2]
            ne0, ne1 = get_policy_from_nash(self.Q_table_1[midprice_integer], self.Q_table_2[midprice_integer], self.solver)
            self.V_learned_1[idx_midprice], self.V_learned_2[idx_midprice] = get_nash_value(self.Q_table_1[midprice_integer], self.Q_table_2[midprice_integer], self.solver)
            self.policy_learned_1[midprice_integer] = ne0.reshape((len(action_ask_price_list), len(action_buy_price_list)))
            self.policy_learned_2[midprice_integer] = ne1.reshape((len(action_ask_price_list), len(action_buy_price_list)))
            policy_error_1 = max(policy_error_1, np.max(np.abs(self.policy_learned_1[midprice_integer] - self.policy_true_1[midprice_integer])))
            policy_error_2 = max(policy_error_2, np.max(np.abs(self.policy_learned_2[midprice_integer] - self.policy_true_2[midprice_integer])))
        return np.max( np.abs(self.V_learned_1-self.V_true_1) ), np.max( np.abs(self.V_learned_2-self.V_true_2) ), \
                    policy_error_1 , policy_error_2
            # print('midprice_integer: ', midprice_integer)
            # print('MM1 policy: ')
            # print(ne0)
            # print('MM2 policy: ')
            # print(ne1)
            # print('state_action_counter_matrix: ', self.state_action_counter_matrix[midprice_integer])

    def compute_true_nash_equilibrium(self,):
        V_max_increment = float('inf')
        V_true_converge_track_1 = np.zeros((self.dim_midprice_grid, self.N_Bellman_iter))
        V_true_converge_track_2 = np.zeros((self.dim_midprice_grid, self.N_Bellman_iter))
        Q_table_true_1 = dict()
        Q_table_true_2 = dict()
        for idx_midprice in range(self.dim_midprice_grid):
            midprice_integer = idx_midprice + 1
            action_ask_price_list = self.env.price_list[self.env.price_list > midprice_integer / 2]
            action_buy_price_list = self.env.price_list[self.env.price_list < midprice_integer / 2]
            dim_action = len(action_ask_price_list) * len(action_buy_price_list)
            Q_table_true_1[midprice_integer] = np.zeros((dim_action, dim_action))
            Q_table_true_2[midprice_integer] = np.zeros((dim_action, dim_action))

        i = -1
        while V_max_increment > self.Bellman_iter_threshold and i < self.N_Bellman_iter - 2:
            i += 1
            self.V_true_1 = V_true_converge_track_1[:, i]
            self.V_true_2 = V_true_converge_track_2[:, i]
            V_max_increment = 0
            for idx_midprice in range(self.dim_midprice_grid):
                midprice_integer = idx_midprice + 1
                action_ask_price_list = self.env.price_list[self.env.price_list > midprice_integer / 2]
                action_buy_price_list = self.env.price_list[self.env.price_list < midprice_integer / 2]
                dim_action = len(action_ask_price_list) * len(action_buy_price_list)

                midprice_prob_vector = self.env.trans_prob_matrix_midprice[idx_midprice, :]
                for idx_optimal_1 in range(dim_action):
                    for idx_optimal_2 in range(dim_action):
                        idx_ask_price_1 = action_ask_price_list[idx_optimal_1 // len(action_buy_price_list)]
                        idx_buy_price_1 = action_buy_price_list[idx_optimal_1 % len(action_buy_price_list)]
                        idx_ask_price_2 = action_ask_price_list[idx_optimal_2 // len(action_buy_price_list)]
                        idx_buy_price_2 = action_buy_price_list[idx_optimal_2 % len(action_buy_price_list)]
                        prob_ask_order_filled_1, prob_ask_order_filled_2 = self.env.prob_executed(midprice_integer * self.env.tick_size / 2, idx_ask_price_1 * self.env.tick_size, idx_ask_price_2 * self.env.tick_size, type="ask")
                        prob_buy_order_filled_1, prob_buy_order_filled_2 = self.env.prob_executed(midprice_integer * self.env.tick_size / 2, idx_buy_price_1 * self.env.tick_size, idx_buy_price_2 * self.env.tick_size, type="buy")
                        reward_expected_1 = (-midprice_integer * self.env.tick_size / 2 + idx_ask_price_1 * self.env.tick_size) * prob_ask_order_filled_1 + (midprice_integer * self.env.tick_size / 2 - idx_buy_price_1 * self.env.tick_size) * prob_buy_order_filled_1
                        reward_expected_2 = (-midprice_integer * self.env.tick_size / 2 + idx_ask_price_2 * self.env.tick_size) * prob_ask_order_filled_2 + (midprice_integer * self.env.tick_size / 2 - idx_buy_price_2 * self.env.tick_size) * prob_buy_order_filled_2
                        Q_table_true_1[midprice_integer][idx_optimal_1, idx_optimal_2] = reward_expected_1 + self.GAMMA_Delta * np.dot(self.V_true_1, midprice_prob_vector)
                        Q_table_true_2[midprice_integer][idx_optimal_2, idx_optimal_1] = reward_expected_2 + self.GAMMA_Delta * np.dot(self.V_true_2, midprice_prob_vector)

                V_new_1, V_new_2 = get_nash_value(Q_table_true_1[midprice_integer], Q_table_true_2[midprice_integer], self.solver)
                V_true_converge_track_1[idx_midprice, i + 1] = V_new_1
                V_true_converge_track_2[idx_midprice, i + 1] = V_new_2
                V_max_increment = max(V_max_increment, np.max(np.abs(V_new_1 - self.V_true_1[idx_midprice])), np.max(np.abs(V_new_2 - self.V_true_2[idx_midprice])))

        # print(V_true_1)
        # print(V_true_2)

        for idx_midprice in range(self.dim_midprice_grid):
            midprice_integer = idx_midprice + 1
            action_ask_price_list = self.env.price_list[self.env.price_list > midprice_integer / 2]
            action_buy_price_list = self.env.price_list[self.env.price_list < midprice_integer / 2]
            ne0, ne1 = get_policy_from_nash(Q_table_true_1[midprice_integer], Q_table_true_2[midprice_integer], self.solver)
            self.policy_true_1[midprice_integer] = ne0.reshape((len(action_ask_price_list), len(action_buy_price_list)))
            self.policy_true_2[midprice_integer] = ne1.reshape((len(action_ask_price_list), len(action_buy_price_list)))
            # print('midprice_integer: ', midprice_integer)
            # print('MM1 policy: ')
            # print(ne0)
            # print('MM2 policy: ')
            # print(ne1)
