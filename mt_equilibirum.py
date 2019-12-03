##
## Computes the equilibrium distribution of the MT system.
## For a N_L/N_R system, computes the probabilities and rates of
## position-dependent reattachment.

import numpy as np

def find_q0(N, alpha, beta_prime, beta, alpha_prime):
    """Find the probability of being in the 0-long decayed state.
    
    N - max length of MT is N (so K=N-1 "normal" lengths are possible)
    alpha - growth rate
    beta_prime - catastrophe rate
    beta - decay rate
    alpha_prime - rescure rate
    """
    
    K = N-1
    
    r = (1 + alpha_prime/beta)/(1 + beta_prime/alpha)
    u_1 = alpha*alpha_prime/(alpha + beta_prime)
    
    u_vals = np.array([u_1 * r**n for n in range(0, K)])
    p_vals = np.copy(u_vals/alpha) # prob. of being in polymerizing state
    p_vals = np.append(p_vals, [p_vals[-1]*(1+alpha_prime/beta)/alpha]) # prob in depoly state
    
    q_vals = np.append([1], u_vals/beta)
    
    total = q_vals.sum() + p_vals.sum()
    
    q_0 = 1/total # prob of 0-length state
    
    return q_0

def compute_probabilities(N, alpha, beta_prime, beta, alpha_prime):
    """Find the probability of being in each state.
    
    N - max length of MT is N (so K=N-1 "normal" lengths are possible)
    alpha - growth rate
    beta_prime - catastrophe rate
    beta - decay rate
    alpha_prime - rescure rate
    """
    
    K = N-1
    
    q_0 = find_q0(N, alpha, beta_prime, beta, alpha_prime)
    
    r = (1 + alpha_prime/beta)/(1 + beta_prime/alpha)
    u_1 = alpha*alpha_prime/(alpha + beta_prime)
    
    u_vals = np.array([u_1 * r**n for n in range(0, K)])
    p_vals = np.copy(u_vals/alpha)
    p_vals = np.append(p_vals, [p_vals[-1]*(1+alpha_prime/beta)/alpha])
    p_vals = np.append([0], p_vals)
    
    q_vals = np.append([1], u_vals/beta)
    q_vals = np.append(q_vals, [0])
    
    q_vals = q_0 * q_vals # prob. in poly. states 1,...,K
    p_vals = q_0 * p_vals # prob. in depoly states 0,...,N-1
    
    return p_vals, q_vals

def reattachment_probabilities(p_vals, q_vals):
    """Given q and p probabilities, probability the length of the MT is >= i.
    
    returns: array of probabilities. Excludes reattachment from the 0-long MT,
    as that's not physical."""
    
    N = len(p_vals)-1
    reattach_prob = []

    for i in range(1,N+1): # obviously could be sped up
        reattach_prob.append(sum(p_vals[i:]) + sum(q_vals[i:]))
    
    return np.array(reattach_prob)

def compute_reattachment_probabilities(N, alpha, beta_prime, beta, alpha_prime):
    """Computed probability the length of the MT is >= i.
    
    returns: array of probabilities. Excludes reattachment from the 0-long MT,
    as that's not physical."""
    
    p_vals, q_vals = compute_probabilities(N, alpha, beta_prime, beta, alpha_prime)
    
    return reattachment_probabilities(p_vals, q_vals)

def compute_walkoff_probabilities(N, alpha, beta_prime, beta, alpha_prime, last_prob=0):
    """Compute the probability of walkoff. We assume zero walkoff probability at last dimer.
    
    Walkoff prob. at last dimer may be set with optional argument last_prob = ..."""
    p_vals, q_vals = compute_probabilities(N, alpha, beta_prime, beta, alpha_prime)
    reattach_prob = reattachment_probabilities(p_vals, q_vals)
    
    walkoff_prob = []
    
    for i in range(1, N):
        prob = (p_vals[i] + q_vals[i])/reattach_prob[i-1] # reattach prob has no 0 length value
        walkoff_prob.append(prob)
    
    walkoff_prob.append(last_prob)
    
    return walkoff_prob