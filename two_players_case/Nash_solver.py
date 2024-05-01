""" Solve equilibrium (Nash/Correlated) with Linear Programming in zero-sum games. """
import numpy as np
from pulp import *
import time
import matplotlib.pyplot as plt

def NashEquilibriumLPSolver(A, B=None, vlim=10, verbose=False):
    """ 
    Solve general-sum Nash equilibrium with Linear Programming, using pulp.
    If B=None, then it degrades to a zero-sum game.
    If the game is asymmetric, B should not be of the same shape as A. For example, if A is (m,n), then B is (n,m).
    One LP problem per player, solve one side, then the other side (with opposite and transpose payoff matrix).
    
    Args:
        vlim: this is the absolute value range of objective variable z, it should be of a similar magnitude of the payoff values. 
    """
    def solve_one_side(A):
        t0 = time.time()
        rows = A.shape[0]
        cols = A.shape[1]
        value_range = [-vlim, vlim]
        var_list = []
        for r in range(rows-1):  # the last prob is 1-sum(previous probs)
            var_list.append(LpVariable(f"x{r}", 0., 1.)) # probability variable is in [0., 1.]
        var_list.append(LpVariable("z", value_range[0], value_range[1]))

        prob = LpProblem("myProblem", LpMaximize)
        last_var = '(1'+''.join([f"-var_list[{r}]" for r in range(rows-1)])+')' # '1-x1-x2-...x(n-1)'

        # define objective 
        prob+=var_list[-1] # maximize z
        obj = "prob+=var_list[-1]"
        exec(obj)
        if verbose: print("Objective:\n", obj) 

        # define contraints
        if verbose: print("Constraints:")
        postfix = f">=var_list[-1]"
        for c in range(cols):
            constr = f"prob+=" + "+".join([f"{A[r][c]}*var_list[{r}]" for r in range(rows-1)]) + f'+{A[rows-1][c]}*{last_var}'+postfix
            exec(constr)
            if verbose: print(constr)
        # for r in range(rows-1):     
        #     constr = f"prob+=var_list[{r}]>=0" # each prob is non-negative (except the last)
        #     exec(constr)  
        #     if verbose:
        #         print(constr)
        constr = f"prob+={last_var}>=0" # last prob is non-negative
        exec(constr) 
        if verbose: print(constr)
        t1 = time.time()
        # solve the LP
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False)) 
        t2 = time.time()

        v_list = np.array([value(var_list[r]) for r in range(rows-1)])

        # handle the exception that sum of all vars except the last is larger than 1 (usually by small value e-8)
        partial_sum = sum(v_list)
        if partial_sum > 1.:
            v_list[np.where(v_list>0)[0]]-=(partial_sum-1.)  # reduce the value exceeding 1. from the first positive value
            v_list = np.append(v_list, 0)
        else:
            v_list = np.append(v_list, 1-sum(v_list))  # add the last one
        
        if verbose: print("Prob Values: ", v_list, ", Objective Value: ", value(var_list[-1]))
        # print('time: ',(t1-t0)/(t2-t0), (t2-t1)/(t2-t0))
        return v_list, value(var_list[-1])
    
    if verbose: print('Player 1:')
    v_list1, z1 = solve_one_side(A)
    if verbose: print('-------------\n Player 2:')
    if B is None:  # zero-sum game
        B=-A.T
    v_list2, z2 = solve_one_side(B)

    return (v_list1, v_list2)

def CoarseCorrelatedEquilibriumLPSolver(A, B=None, vlim=10, verbose=False):
    """ 
    Solve general-sum coarse correlated equilibrium (CCE) with Linear Programming, using pulp.
    If B=None, then it degrades to a zero-sum game.
    """

    rows = A.shape[0]
    cols = A.shape[1]
    value_range = [-vlim, vlim]
    marg_probs1, marg_probs2 = [], []  # marginal probabilities for two players
    joint_probs = []

    for r in range(rows-1):  # the last prob is 1-sum(previous probs)
        marg_probs1.append(LpVariable(f"x1_{r}", 0., 1.)) # probability value is in [0., 1.]
    for c in range(cols-1):  # the last prob is 1-sum(previous probs)
        marg_probs2.append(LpVariable(f"x2_{c}", 0., 1.)) # probability value is in [0., 1.]
    for r in range(rows):
        for c in range(cols):
            if not (r == rows-1 and c == cols-1): 
                joint_probs.append(LpVariable(f"x12_{r}{c}", 0., 1.)) # probability value is in [0., 1.]

    prob = LpProblem("myProblem", LpMaximize)
    last_marg_prob1 = '(1'+''.join([f"-marg_probs1[{r}]" for r in range(rows-1)])+')' # '1-x1_1-x1_2-...x1_(n-1)'
    last_marg_prob2 = '(1'+''.join([f"-marg_probs2[{c}]" for c in range(cols-1)])+')' # '1-x2_1-x2_2-...x2_(n-1)'
    last_joint_prob = '(1'+''.join([f"-joint_probs[{i}]" for i in range(len(joint_probs))])+')'
    
    # right hand side of the contraints
    A_ = A.reshape(-1)
    rhs1 = '+'.join([f"{v}*joint_probs[{i}]" for i, v in enumerate(A_[:-1])])
    rhs1 += f"+{A_[-1]}*{last_joint_prob}"
    if B is None: 
        B = -A.T
    B_ = B.T.reshape(-1)
    rhs2 = '+'.join([f"{v}*joint_probs[{i}]" for i, v in enumerate(B_[:-1])])  # B.T to match the order of joint_probs
    rhs2 += f"+{B_[-1]}*{last_joint_prob}"

    # define contraints
    if verbose: print("Constraints:")
    postfix = f"<={rhs1}"
    for r in range(rows): 
        constr = f"prob+=" + "+".join([f"{A[r][c]}*marg_probs2[{c}]" for c in range(cols-1)]) \
        + f'+{A[r][cols-1]}*{last_marg_prob2}'+postfix
        exec(constr)
        if verbose: print(constr)
    postfix = f"<={rhs2}"
    B_ = B.T
    for c in range(cols): 
        constr = f"prob+=" + "+".join([f"{B_[r][c]}*marg_probs1[{r}]" for r in range(rows-1)]) \
        + f'+{B_[rows-1][c]}*{last_marg_prob1}'+postfix
        exec(constr)
        if verbose: print(constr)            
    
    constr1 = f"prob+={last_marg_prob1}>=0" # last prob is non-negative
    exec(constr1) 
    constr2 = f"prob+={last_marg_prob2}>=0"
    exec(constr2)
    constr3 = f"prob+={last_joint_prob}>=0"
    exec(constr3)
    if verbose: 
        print(constr1)
        print(constr2)
        print(constr3)

    # solve the LP
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False)) # this will prevent unnecessary info report

    def append_last(l):
        """ 
        Need to check whether the sum of previous variables is larger than 1., since the constraint can be not strict.
        """
        # handle the exception that sum of all vars except the last is larger than 1 (usually by small value e-8)
        partial_sum = sum(l)
        if partial_sum > 1.:
            l[np.where(l>0)[0]]-=(partial_sum-1.)  # reduce the value exceeding 1. from the first positive value
            l = np.append(l, 0)
        else:
            l = np.append(l, 1-sum(l))  # add the last one
        return l

    result_marg_probs1 = np.array([value(marg_probs1[r]) for r in range(rows-1)])
    result_marg_probs1 = append_last(result_marg_probs1)
    result_marg_probs2 = np.array([value(marg_probs2[c]) for c in range(cols-1)])
    result_marg_probs2 = append_last(result_marg_probs2)
    result_joint_probs = np.array([value(joint_probs[i]) for i in range(len(joint_probs))])
    result_joint_probs = append_last(result_joint_probs)

    return result_marg_probs1, result_marg_probs2, result_joint_probs

def get_nash_value(payoff_1, payoff_2, solver):
    vlim = max( np.max(np.abs(payoff_1)), np.max(np.abs(payoff_2)) )+1
    if solver == 'LP':
        ne = NashEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
        ne_value_1, ne_value_2 = ne[0]@payoff_1@ne[1].T, ne[1]@payoff_2@ne[0].T
    elif solver == 'CCE':
        ne0, ne1, ne = CoarseCorrelatedEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
        ne_value_1, ne_value_2 = ne0@payoff_1@ne1.T, ne1@payoff_2@ne0.T
    else:
        raise NotImplementedError

    return ne_value_1, ne_value_2

def get_action_from_nash(payoff_1, payoff_2, solver):
    """ sample action from current 'nash' strategies. """
    vlim = max( np.max(np.abs(payoff_1)), np.max(np.abs(payoff_2)) )+1
    if solver == 'LP':
        ne = NashEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
    elif solver == 'CCE':
        ne0, ne1, ne = CoarseCorrelatedEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
        ne=(ne0, ne1)
    else:
        raise NotImplementedError

    actions = []
    for dist in ne: # sample actions from Nash distribution
        sample_hist = np.random.multinomial(1, dist)
        a = np.where(sample_hist>0) # hist is histogram
        actions.append(a[0][0])
    return actions[0], actions[1]

def get_policy_from_nash(payoff_1, payoff_2, solver):
    """ sample action from current 'nash' strategies. """
    vlim = max( np.max(np.abs(payoff_1)), np.max(np.abs(payoff_2)) )+1
    if solver == 'LP':
        ne = NashEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
    elif solver == 'CCE':
        ne0, ne1, ne = CoarseCorrelatedEquilibriumLPSolver(payoff_1, payoff_2, vlim=vlim)
        ne=(ne0, ne1)
    else:
        raise NotImplementedError

    return ne[0], ne[1]