from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in range(L):
            for j in range(S):
                if i==0:
                    alpha[j, i] = self.pi[j] * self.B[j, self.obs_dict[Osequence[i]]]
                else:
                    alpha[j,i]=self.B[j,self.obs_dict[Osequence[i]]]* sum([self.A[m, j] * alpha[m, i - 1] for m in range(S)])
        
                
        ###################################################
        
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        for i in range(L - 1,-1,-1):
            for j in range(S):
                if i==L-1:
                    beta[j, L - 1] = 1
                else:
                    beta[j, i] = sum([beta[m, i + 1] * self.A[j, m] * self.B[m, self.obs_dict[Osequence[i + 1]]] for m in range(S)])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        
        ###################################################
        # Edit here
        alpha=self.forward(Osequence)
        prob = np.sum(alpha,axis=0)[-1]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha=self.forward(Osequence) #num_state*L
        beta=self.backward(Osequence) #num_state*L
        prob=(alpha*beta)/np.sum(alpha,axis=0)[-1]
        
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        P=np.sum(alpha,axis=0)[-1]
        for t in range(0,L-1): 
            for i in range (0,S):
                for j in range (0,S):
                    prob[i,j,t]=(alpha[i,t]*self.A[i,j]*self.B[j,self.obs_dict[Osequence[t + 1]]]*beta[j,t+1])/P    
#        prob=
        
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = list()
        ###################################################
        S = len(self.pi)
        N = len(Osequence)
        delta = np.zeros([S, N])
        del_s = np.zeros([S, N])
        key_value=list(self.state_dict.keys()) 
        for i in range(0, N):
            for j in range(S):
                if i==0:
                    delta[j, i] = self.pi[j] * self.B[j, self.obs_dict[Osequence[i]]]
                    del_s[j, i] = 0
                else:
                    temp = [delta[m, i - 1] * self.A[m, j] for m in range(S)]
                    delta[j, i] = self.B[j, self.obs_dict[Osequence[i]]]*max(temp)
                    del_s[j, i] = np.argmax(temp)
        path.append(np.argmax(delta[:, -1]))
        for t in range(N - 1,0,-1):
            path.append(del_s[path[-1], t].astype('int'))
        path.reverse()
        paths=[]
        for i in path:
            paths.append(key_value[i])
        
        ###################################################
        return paths
