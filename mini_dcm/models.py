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


#hypothesis to tests

def model_mask_chain_true(): #the model aligns with the true connectivity
    """
    Model 1: correct chain 1 -> 2 -> 3, with self-connections on all nodes.
    """
    mask = np.zeros((3, 3), dtype=bool) #the sturcture of the A matrix

    np.fill_diagonal(mask, True) #self-connections

    mask[1, 0] = True   # node 1 -> node 2
    mask[2, 1] = True   #node 2 -> node 3
    return mask


def model_mask_missing_2to3():
    """
    Model 2: same as chain, but without 2 -> 3 connection.
    """
    mask = np.zeros((3, 3), dtype=bool)

    np.fill_diagonal(mask, True)
    mask[1, 0] = True   # 1 -> 2
    return mask


def model_mask_wrong_3to1():
    """
    Model 3: chain plus wrong feedback 3 -> 1 instead of 2 -> 3.
    """
    mask = np.zeros((3, 3), dtype=bool)

    np.fill_diagonal(mask, True)
    mask[1, 0] = True   # 1 -> 2
    mask[0, 2] = True   # 3 -> 1
    # 2 -> 3 absent

    return mask