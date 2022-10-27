"""
    Defines the training procedure for the three models 
"""

# 1. collect 10,000 rollouts from a random policy

# 2. Train V to encode each frame into a latent vector z

# 3. Train M to model P(z[t+1] | z[t], a[t], h[t])

# 4. Train (evolve) C to maximize the expected cumulative reward
