from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for i in range(L):
            if i == 0:
                alpha[:, :1] = self.pi.reshape(S, 1) * self.B[:, O[i]:O[i]+1]
            else:
                alpha[:, i:i+1] = self.B[:, O[i]: O[i]+1] * \
                    np.sum(self.A * alpha[:, i-1:i], axis=0).reshape(S, 1)

        return alpha


        

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for i in range(L-1, -1, -1):  # 6,5,4,3,2,1,0
            if i == L-1:
                beta[:, i:i+1] = 1
            else:
                beta[:, i:i+1] = np.sum(self.A.T * self.B[:, O[i+1]: O[i+1]+1]
                                        * beta[:, i+1:i+2], axis=0).reshape(S, 1)

        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        # P(X{1:T} = x{1:T}) = np.sum(alpha\s(t)* beta\s(t))
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        # true for any t. I am taking t = 0 (alpha[:, 1] means alpha at time t=0)
        prob = np.sum(alpha[:, 1]*beta[:, 1], axis=0)
        return prob



    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        L = len(Osequence)
        S = len(self.pi)
        gamma = np.zeros([S, L])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prob = np.sum(alpha[:, 1]*beta[:, 1], axis=0)
        for i in range(L):
            gamma[:, i:i+1] = alpha[:, i:i+1] * beta[:, i:i+1] / prob
        return gamma


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        O = self.find_item(Osequence)
        normalization = np.sum(alpha[:, 1]*beta[:, 1], axis=0)

        for t in range(L-1):
            for i in range(S):
                for j in range(S):
                    prob[i, j, t] = (alpha[i][t] * self.A[i][j] *
                                     self.B[j][O[t+1]] * beta[j][t+1]) / normalization

        # print(prob)
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        delta = np.zeros([S, L])
        smallDelta = np.zeros([S, L])

        for t in range(L):
            if t == 0:
                delta[:, t:t +
                      1] = self.pi.reshape(S, 1) * self.B[:, O[t]:O[t]+1]
                # print("delta \n", delta)
            else:
                #print("*" * 60)
                delta[:, t:t+1] = self.B[:, O[t]:O[t]+1] * \
                    np.amax(self.A * delta[:, t-1:t], axis=0).reshape(S, 1)

                smallDelta[:, t:t +
                           1] = np.argmax(self.A * delta[:, t-1:t], axis=0).reshape(S, 1)
                # print(smallDelta)

        #print("delta \n", delta)
        # Backtracking
        #print("Backtracking ...")
        # print(smallDelta)
        state_idx = np.argmax(delta[:, L-1: L], axis=0)[0]
        state = self.find_key(self.state_dict, state_idx)
        #print("state_idx state : ", state_idx, state)

        #print("state :", state)
        path.append(state)
        for t in range(L-1, 0, -1):
            state_idx = int(smallDelta[state_idx][t])
            state = self.find_key(self.state_dict, state_idx)
            #print("state_idx state :", state_idx, state)
            path.append(state)

        # Reverse path
        path.reverse()
        print("PATH : ", path)
        return path
        
        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
