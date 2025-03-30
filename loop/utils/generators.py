import numpy as np

def generate_permutation(params):
    
    out_dict = {}

    for key in params().keys():
        out_dict[key] = np.random.choice(params()[key])

    return out_dict
