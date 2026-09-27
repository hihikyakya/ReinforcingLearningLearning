import numpy as np

from .policy.policy import policy_evaluation, policy_improvement
from .env.gridworld import GridWorld

def main():
    env = GridWorld()
    policy = [np.array([0.25]*4) for s in range(env.state.shape[0] * env.state.shape[1])] # 갈 수 있는 모든 칸의 policy(pi)
        

    V = policy_evaluation(env, policy)

    print(np.round(V, 2))


    print("---------"*3)

    improved_policy = policy_improvement(V, env)

    y_dim, x_dim=env.state.shape
    for y in range(y_dim):
        row=[]
        for x in range(x_dim):
            state_index = y * x_dim + x
            best_action = np.argmax(improved_policy[state_index])
            row.append(int(best_action))

        print(row) # 0: up, 1: down, 2: left, 3:right


if __name__ == "__main__":
    main()
