import numpy as np
from numpy.typing import NDArray

from ..env.gridworld import GridWorld

def print_value_map(V:NDArray):
    print("State value map:")
    print(np.round(V,2))

def print_policy_map(policy_list:list, env:GridWorld):
    print("Policy map:")
    shape=env.state.shape
    for y in range(shape[0]):
        row = []
        for x in range(shape[1]):
            state_index = y * shape[1] + x
            
            row.append(
                int(np.argmax(
                    policy_list[state_index]
                ))
            )
        print(row)