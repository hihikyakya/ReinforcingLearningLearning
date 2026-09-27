import numpy as np

from .tools.print import print_policy_map
from .policy.policy import policy_evaluation, policy_improvement, policy_iteration, value_iteration
from .env.gridworld import GridWorld

def policy_basic():
    env = GridWorld()

    policy = [np.array([0.25,0.25,0.25,0.25]) for s in range(env.grid_map.shape[0] * env.grid_map.shape[1])]

    V = policy_evaluation(env, policy)

    print(np.round(V, 2))


    print("---------"*3)

    improved_policy = policy_improvement(V, env)

    print_policy_map(improved_policy, env)


def policy_iteration_exec():
    env = GridWorld()
    policy = [np.array([0.25,0.25,0.25,0.25]) for s in range(env.grid_map.shape[0] * env.grid_map.shape[1])] # 갈 수 있는 모든 칸의 policy(pi)
        

    optimal_V, optimal_policy = policy_iteration(env, policy)

    print("\nOptimal value function: ")
    print(np.round(optimal_V, 2))

    print("\nOptimal policy:")
    print_policy_map(optimal_policy, env)

def value_iteration_exec():
    env = GridWorld()

    optimal_V_vi, optimal_policy_vi = value_iteration(env)

    print("\nOptimal value function (Value Iteration):")
    print(np.round(optimal_V_vi, 2))

    print("\nOptimal policy (Value Iteration):")
    print_policy_map(optimal_policy_vi, env)


if __name__ == "__main__":
    # policy_basic()

    policy_iteration_exec()

    # value_iteration_exec()