import numpy as np

def effective_connectivity():
    A = np.array([
        [0.9, 0.0, 0.0], #Node 1: only self-connection
        [0.3, 0.8, 0.0], #Node 2: From  Node 1 and itself
        [0.0, 0.25, 0.85]  #Node 3: From Node 2 and itself
    ])

    C = np.array([
        [1.0],  # Node 1: driven by Input 
        [0.0],  # Node 2: no direct input
        [0.0]   # Node 3: no direct input
    ])

    return A, C

