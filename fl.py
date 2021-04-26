import csv
import sys
import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.example
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning

import numpy as pynp
import pandas as pypd
import matplotlib.pyplot as pymap
import seaborn as pysb
import time

Trans = {}
Rew = {}

#gamma is the discount factor
if len(sys.argv)>1:
    gamma = float(sys.argv[1])
else:
    gamma = 0.9

#the maximum error allowed in the utility of any state
if len(sys.argv)>2:
    epsilon = float(sys.argv[2])
else:
    epsilon = 0.001

def read_file():
    #read transitions from file and store it to a variable
    with open('/Users/uma/Desktop/Desktop/omscs/ml2021/assignments/ul/transitions.csv', 'r') as csvfile:
        input1 = csv.reader(csvfile, delimiter=',')
        for row in input1:
            if row[0] in Trans:
                if row[1] in Trans[row[0]]:
                    Trans[row[0]][row[1]].append((float(row[3]), row[2]))
                else:
                    Trans[row[0]][row[1]] = [(float(row[3]), row[2])]
            else:
                Trans[row[0]] = {row[1]:[(float(row[3]),row[2])]}

    #read rewards file and save it to a variable
    with open('/Users/uma/Desktop/Desktop/omscs/ml2021/assignments/ul/rewards.csv', 'r') as csvfile:
        input2 = csv.reader(csvfile, delimiter=',')
        for row in input2:
            Rew[row[0]] = float(row[1]) if row[1] != 'None' else None

read_file()

class MarkovDecisionProcess:

    def __init__(self, transition={}, reward={}, gamma=.9):
        #collect all nodes from the transition models
        self.states = transition.keys()
        #initialize transition
        self.transition = transition
        #initialize reward
        self.reward = reward
        #initialize gamma
        self.gamma = gamma

    def R(self, state):
        """return reward for this state."""
        return self.reward[state]

    def actions(self, state):
        """return set of actions that can be performed in this state"""
        return self.transition[state].keys()

    def T(self, state, action):
        """for a state and an action, return a list of (probability, result-state) pairs."""
        return self.transition[state][action]

#Initialize the MarkovDecisionProcess object
MKDP = MarkovDecisionProcess(transition=Trans, reward=Rew)

def value_iteration():

    start_time = time.time()
    states = MKDP.states
    actions = MKDP.actions
    T = MKDP.T
    R = MKDP.R

    #initialize value of all the states to 0 (this is k=0 case)
    V1 = {s: 0 for s in states}
    while True:
        V = V1.copy()
        delta = 0
        for s in states:
            #Bellman update, update the utility values
            V1[s] = R(s) + gamma * max([ sum([p * V[s1] for (p, s1) in T(s, a)]) for a in actions(s)])
            #calculate maximum difference in value
            delta = max(delta, abs(V1[s] - V[s]))

        #check for convergence, if values converged then return V
        if delta < epsilon * (1 - gamma) / gamma:
            return V

def best_policy(V):

    states = MKDP.states
    actions = MKDP.actions
    pi = {}
    for s in states:
        pi[s] = max(actions(s), key=lambda a: expected_utility(a, s, V))
    return pi


def expected_utility(a, s, V):
    T = MKDP.T
    return sum([p * V[s1] for (p, s1) in MKDP.T(s, a)])


#call value iteration
V = value_iteration()
print('State - Value')
for s in V:
    print(s, ' - ' , V[s])
pi = best_policy(V)
print('\nOptimal policy is \nState - Action')
for s in pi:
    print(s, ' - ' , pi[s])

###Value Iteration
start_time = time.time()
trans,rewards = hiive.mdptoolbox.example.forest(S=100)
data_cl = ['gamma', 'epsilon', 'tim', 'iter', 'rew','pol', 'err']
vi_data = pypd.DataFrame(0.0, index=pynp.arange(20), columns=data_cl)

r_i = 0
for g_iter in [0.1, 0.2, 0.4, 0.8, 0.9]:
    for e_iter in [1e-1, 1e-2, 1e-4, 1e-8,1e-10]:
        vi_run = ValueIteration(trans, rewards, gamma=g_iter, epsilon=e_iter, max_iter=1000)
        viruns = vi_run.run()
        t = viruns[-1]['Time']
        s = viruns[-1]['Iteration']
        r = viruns[-1]['Max V']
        vi_data['gamma'][r_i] = g_iter
        vi_data['epsilon'][r_i] = e_iter
        vi_data['tim'][r_i] = t
        vi_data['iter'][r_i] = s
        vi_data['rew'][r_i] = r

        #print('%.1f,\t%.1E,\t%.3f,\t%2d,\t%2f' % (g_iter, e_iter, t, s, r))
        r_i = r_i + 1
    vi_data.fillna(0, inplace=True)
    vi_data.head()
pt = vi_data.plot(x='gamma', y='rew',color='red',dashes=[4, 3], title="Gamma vs Rewards")
pt.set_xlabel("Gamma")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()
endTime = time.time() - start_time
print("total - time : ", endTime)

pt = vi_data.plot(x='iter', y='rew',color='green',dashes=[4, 3], title="Iterations vs Rewards")
pt.set_xlabel("Iterations")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()

pt = vi_data.plot(x='epsilon', y='rew',color='purple',dashes=[4, 3],title="Epsilon vs Rewards")
pt.set_xlabel("Epsilon")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()

pt = vi_data.plot(x='tim', y='rew',color='red',dashes=[4, 3],title="Time vs Rewards")
pt.set_xlabel("Time")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()

#Policy Iteration

start_time = time.time()
trans,rewards = hiive.mdptoolbox.example.forest(S=500)
data_cl = ['gamma', 'epsilon', 'tim', 'iter', 'rew','pol', 'err']
vi_data = pypd.DataFrame(0.0, index=pynp.arange(6), columns=data_cl)

r_i = 0
for g_iter in [0.1, 0.2, 0.4, 0.8, 0.9,0.99,0.999]:
    vi_run = PolicyIteration(trans, rewards, gamma=g_iter, max_iter=100000,eval_type='matrix')
    viruns = vi_run.run()
    t = viruns[-1]['Time']
    s = viruns[-1]['Iteration']
    r = viruns[-1]['Max V']
    vi_data['gamma'][r_i] = g_iter
    vi_data['tim'][r_i] = t
    vi_data['iter'][r_i] = s
    vi_data['rew'][r_i] = r
    print('%.1f,\t%.3f,\t%4d,\t%5f' % (g_iter, t, s, r))
    r_i = r_i + 1

vi_data.fillna(0, inplace=True)
vi_data.head()

endTime = time.time() - start_time
print("total time : ", endTime)

pt=vi_data.plot(x='gamma', y='rew',color='red',dashes=[4, 3], title='Gamma vs Rewards')
pt.set_xlabel("Gamma")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()

pt=vi_data.plot(x='iter', y='rew',color='green',dashes=[4, 3], title = 'Iteration vs Rewards')
pt.set_xlabel("Iteration")
pt.set_ylabel("Rewards")
pymap.grid(True)
pymap.show()


##QLearning


start_time = time.time()
trans,rewards = hiive.mdptoolbox.example.forest(S=1000)
gammas         = [0.1, 0.2, 0.4, 0.8]
alphas         = [0.01, 0.1, 0.2]
alpha_decays   = [0.1,0.3, 0.6, 0.9]
epsilon_decays = [0.1,0.3, 0.6, 0.9]
iterations     = [1e1, 1e2, 1e4]
data_cl = ['gamma', 'epsilon_decay', 'tim','avg_rew','m_rew','iter', 'rew','pol', 'err', 'alpha', 'alpha_decay']
vi_data = pypd.DataFrame(0.0, index=pynp.arange(588000), columns=data_cl)

r_i = 0
for g_iter in [0.1, 0.2, 0.4, 0.8]:
    for a_iter in [0.01, 0.1, 0.2]:
        for a_d_iter in [0.1,0.3, 0.6, 0.9]:
            for e_d_iter in [0.1,0.3, 0.6, 0.9]:
                for n_iters in range(1000):
                    vi_run = QLearning(trans, rewards, gamma=g_iter, alpha=a_iter, alpha_decay=a_d_iter, epsilon_decay=e_d_iter, n_iter=10001)
                    viruns = vi_run.run()
                    m_rew, avg_rew= [], []
                    for virun in viruns:
                        m_rew.append(virun['Max V'])
                        avg_rew.append(virun['Mean V'])

                    vi_data['gamma'][r_i] = g_iter
                    vi_data['alpha'][r_i] = a_iter
                    vi_data['alpha_decay'][r_i] = a_d_iter
                    vi_data['epsilon_decay'][r_i] = e_d_iter
                    vi_data['tim'][r_i] = viruns[-1]['Time']
                    vi_data['iter'][r_i] = viruns[-1]['Iteration']
                    vi_data['rew'][r_i] = viruns[-1]['Max V']
                    
                    r_i = r_i + 1

vi_data.fillna(0, inplace=True)
vi_data.head()

endTime = time.time() - start_time

# vi_data.plot(x='gamma', y='rew',color='red',dashes=[4, 3])
# pymap.grid(True)
# pymap.show()
#
# vi_data.plot(x='iter', y='rew',color='green',dashes=[4, 3])
# pymap.grid(True)
# pymap.show()
#
# vi_data.plot(x='epsilon', y='rew',color='purple',dashes=[4, 3])
# pymap.grid(True)
# pymap.show()

# # Plot time vs. Iterations
# pymap.figure()
# pymap.title('Time vs Iteration')
# pymap.plot(x=vi_data['iter'],y=vi_data['tim'])
# pymap.xlabel('Iterations')
# pymap.ylabel("Time")
# pymap.legend(loc="best")
# pymap.grid()
# pymap.show()


