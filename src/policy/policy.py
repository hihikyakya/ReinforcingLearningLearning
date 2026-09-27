import numpy as np
from numpy.typing import NDArray

from ..env.gridworld import GridWorld
from ..models.position import Position


def policy_evaluation( env:GridWorld, policy: list[np.ndarray]) -> NDArray:
    env.reset()
    theta = 1e-4
    loop_count=0

    while True:
        delta = 0
        dim_y, dim_x = env.state.shape
        new_state = np.zeros((dim_y,dim_x))

        # {pi∑p (r + \gamma * v(s_t+1))}
        for i in range(dim_y * dim_x - 1):
            v_s = 0

            s_t_y, s_t_x= np.divmod( i, dim_y)
            s_t=Position(s_t_x, s_t_y)
            
            for a_t in env.action_space: 
                pi_a = policy[i][a_t]
                p_ss = 1.0
                v_s_next, s_t1 = env.v_s_next(s_t,a_t)
                reward = env.rewarding(s_t,a_t,s_t1)

                v_s += pi_a * p_ss * (reward + env.gamma * v_s_next)

            new_state[s_t.y, s_t.x] = v_s

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
            s_t = Position(x,y)

            pi = [0,0,0,0]
            q_values = []

            for a_t in env.action_space:
                s_next = env.s_next(s_t, a_t)
                reward = env.rewarding(s_t, a_t, s_next)

                q_value = reward + env.gamma * V[s_next.y, s_next.x]
                q_values.append(q_value)

            best_action = np.argmax(q_values) # argmax => a
            pi[best_action] = 1.0  #  max_a pi

            new_policy.append(pi)

    return new_policy



