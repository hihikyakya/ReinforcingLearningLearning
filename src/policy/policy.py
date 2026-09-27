import numpy as np
from numpy.typing import NDArray

from ..env.gridworld import GridWorld
from ..tools.print import print_policy_map, print_value_map


def policy_evaluation(env:GridWorld, policy: list[np.ndarray]) -> NDArray:
    env.reset()
    theta = 1e-4
    loop_count=0

    while True:
        delta = 0
        dim_y, dim_x = env.grid_map.shape
        new_state = np.zeros_like(env.grid_map, dtype=np.float32)

        # {pi∑p (r + \gamma * v(s_t+1))}
        for i in range(dim_y * dim_x):
            v_s = 0.

            s_t_x, s_t_y= np.divmod( i, dim_y)
            s_t=(s_t_y,s_t_x)

            if s_t[0] == env.goal_pos[0] and s_t[1] == env.goal_pos[1]:
                new_state[s_t[0], s_t[1]] = 0
                continue # goal pass

            for a_t in env.action_space: 
                pi_a = policy[i][a_t]
                p_ss=1.0

                v_s_next, s_t1 = env.v_s_next(s_t,a_t)
                reward = env.rewarding(s_t,a_t,s_t1)

                v_s += (pi_a * p_ss * (reward + env.gamma * v_s_next))
                

            new_state[s_t[0], s_t[1]] = v_s
        

        value_delta = np.sum(np.abs(new_state - env.state)) # ∑pi∑p (r + \gamma * v(s_t+1))
        env.state = new_state

        delta = max(delta, value_delta)
        loop_count += 1
        print(f"[{loop_count}] Delta: {delta}")

        if (delta <= theta) or loop_count >= 500: # exit 조건
            break

    return env.state

# V -> q -> argmax a => pi[a*]=1.0 => [loop...]
def policy_improvement(V:NDArray, env: GridWorld):
    new_policy = []

    y_size, x_size = V.shape

    for y in range(y_size):
        for x in range(x_size):
            s_t = (y,x)

            pi = [0,0,0,0]
            q_values = []

            for a_t in env.action_space:
                s_next = env.s_next(list(s_t), a_t)
                reward = env.rewarding(s_t, a_t, s_next)

                q_value = reward + (env.gamma * V[s_next[0], s_next[1]])
                q_values.append(q_value)

            best_action = np.argmax(q_values) # argmax => a
            pi[best_action] = 1.0  #  max_a pi

            new_policy.append(pi)

    return new_policy



def policy_iteration(env:GridWorld, initial_policy, max_iterations=100):
    current_policy = [pi.copy() for pi in initial_policy]

    for iteration in range(max_iterations):
        # Policy evaluation
        V = policy_evaluation(env,current_policy)

        improved_policy = policy_improvement(V, env)

        policy_stable = all(
            np.array_equal(current_policy[i], improved_policy[i])
            for i in range(len(current_policy))
        )

        current_policy = improved_policy

        print(f"\n===== Policy iteration: {iteration + 1} =====")
        print_value_map(V)
        print_policy_map(current_policy, env)

        if policy_stable:
            print("\nPolicy is stable. Converaged.")
            break
    return V, current_policy



def value_iteration(env:GridWorld, max_iterations=500, theta=1e-4):
    env.reset()

    for iteration in range(max_iterations):
        delta = 0
        y, x = env.state.shape
        new_state = np.zeros_like(env.state, dtype=np.float32)
        for i in range(y*x):
            s_t_x,s_t_y=np.divmod(i,y)
            s_t=(s_t_y,s_t_x)

            q_values = []
            for a_t in env.action_space:
                v_s_next, s_t1 = env.v_s_next(s_t,a_t)
                reward = env.rewarding(s_t, a_t, s_t1)
                q_value = reward + env.gamma * v_s_next
                q_values.append(q_value)

            best_value = max(q_values) # 가장 괜찮은 action에 의한 value
            new_state[s_t[0], s_t[1]] = best_value

        value_delta = np.sum(np.abs(new_state - env.state))
        env.state = new_state

        delta = max(delta, value_delta)

        print(f"[{iteration + 1}] Delta: {delta:.3f}")

        if delta <= theta:
            print("\nValue iteration converaged.")
            break

    optimal_policy = policy_improvement(env.state, env)
    return env.state, optimal_policy

