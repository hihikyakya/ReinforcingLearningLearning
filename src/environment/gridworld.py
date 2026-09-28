import numpy as np


class GridWorld:
    def __init__(self):
        # Initialize the grid world environment
        self.grid_map = np.array(
            [
                [1,1,1,1,0,1],
                [1,1,0,1,0,1],
                [1,1,0,1,0,1],
                [1,1,1,1,1,1],
                [1,1,1,0,1,1],
            ]
        )
        self.gamma=1.0

        self.goal_pos=(0, self.grid_map.shape[1]-1) # y,x
        self.start_pos=(0,0) # x,y
        self.action_space = [0,1,2,3] # P_a=[0.25]*4
        self.reset()
    def reset(self):
        # 환경 state초기화
        self.state=np.zeros_like(self.grid_map, dtype=np.float32)
        return self.state

    def s_next(self, s_t, a_t: int):
        # action_t에 따라 state 수정
        pos=s_t

        if a_t==0: # up
            pos = [max(s_t[0]-1,0), s_t[1]]
        elif a_t==1: # down
            pos = [min(s_t[0]+1, self.grid_map.shape[0]-1), s_t[1]]
        elif a_t==2: # left
            pos = [s_t[0], max(s_t[1]-1, 0)]    
        elif a_t==3: # right
            pos = [s_t[0], min(s_t[1]+1, self.grid_map.shape[1]-1)]

        
        if self.grid_map[pos[0],pos[1]]==0: # 벽 처리
            pos = s_t

        return pos

        
    def v_s_next(self, s_t, a_t:int):
        s_next = self.s_next(list(s_t), a_t) # s_{t+1}
        return self.state[s_next[0],s_next[1]], s_next
    def rewarding(self, s_t, a_t:int, s_next):
        if s_t[0] == self.goal_pos[0] and s_t[1] == self.goal_pos[1]:
            return 0
        else:
            return -1 if not (s_t[0]==s_next[0] and s_t[1]==s_next[1]) else -1.5
            