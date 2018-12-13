import numpy as np
class Agent:
    
    def __init__(self, env):
        self.states = set()
        self.actions = ['H', 'S']
        for _ in range(10000):
            self.states.add(env.sample_state())
        self.states = list(self.states)
        self.Q = {}
        self.policy = {}
        for state in self.states:
            self.Q[state] = {}
            for action in self.actions:
                self.Q[state][action] = np.random.random()
        for state in list(self.Q.keys()):
            if self.Q[state]['H'] > self.Q[state]['S']:
                self.policy[state] = 'H'
            else:
                self.policy[state] = 'S'

    def take_action(self, state):
        if np.random.random() > 0.1:
            return self.policy[state]
        else:
            return np.random.choice(['H', 'S'])
            
    def train(self, env):
        returns = {}
        for state in self.states:
            returns[state] = {}
            for action in self.actions:
                returns[state][action] = []
        for _ in range(10000):
            episode = []
            seen_states = set()
            s, r, done = env.reset()
            while not done:
                a = self.take_action(s)
                sp, r, done = env.step(a)
                episode.append((s, a, r))
                s = sp
            G = 0
            for step in reversed(episode):
                s, a, r = step
                G = 0.9*G + r  
                if (s, a) not in seen_states:
                    seen_states.add((s, a))
                    returns[s][a].append(G)
                    self.Q[s][a] = np.mean(returns[s][a])
            self.gen_optimum_policy()
    
    def gen_optimum_policy(self):
        for state in list(self.Q.keys()):
            if self.Q[state]['H'] > self.Q[state]['S']:
                self.policy[state] = 'H'
            else:
                self.policy[state] = 'S'