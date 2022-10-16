
import numpy as np

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
NOOP = 4

WIN_REWARD = 10
ACTION_PENALTY = {
    RIGHT: -1,
    UP: -1,
    LEFT: -1,
    DOWN: -1,
    NOOP: -0.1  # lower penalty for resting
}

class MultiAgentGrid(rllib.env.multi_agent_env.MultiAgentEnv):

    def __init__(self, n_agents, grid_size):
        super.__init__()
        
        self.n_agents = n_agents
        self.grid_size = grid_size

        self.observation_space = [(0,0) for i in range(n_agents)]
        self.rng = np.random.default_rng()

        self._target_location = np.array([0,0])
        self._agent_locations = [ np.array([0,0]) for i in range(self.n_agents) ]

        self._action_to_direction = {
            RIGHT: np.array([1, 0]),
            UP: np.array([0, 1]),
            LEFT: np.array([-1, 0]),
            DOWN: np.array([0, -1]),
            NOOP: np.array([0, 0])
        }


    def step(self, actions):

        # initialize step results
        rewards = [0 for i in range(self.n_agents)]
        dones = [False for i in range(self.n_agents)]
        infos = [{} for i in range(self.n_agents)]

        winners = []

        for i, action in enumerate(actions):

            # update the reward (movement penalty)
            rewards[i] += ACTION_PENALTY[action]

            # update location
            self._agent_locations[i] = \
                self._agent_locations[i] + self._action_to_direction[action]

            # determine if agent has reached the target
            if np.equal(self._agent_locations[i], self._target_location):
                winners.append(i)

        # randomly select a winner if there are more than one
        if winners:
            winner = self.rng.choice(len(winners), 1)
            dones[winner] = True
            rewards[winner] += WIN_REWARD
            
        return self._get_observations(), rewards, dones, infos

    def reset(self):

        # generation origin positions
        locations = self.rng.choice(
            self.grid_height * self.grid_width,
            1 + self.n_agents, replace=False
        )
        self._target_location = np.array([locations[0] % self.grid_size[0], locations[0] // self.grid_size[0]])
        

        self._agent_locations = np.array([
                np.array([
                    locations[i+1] % self.grid_size[0], # x-coordinate
                    locations[i+1] // self.grid_size[0] # y-coordinate
                ]) for i in range(self.n_agents)
        ])


    def _get_observations(self):
        return [{
            'target': self._target_location.copy(),
            'location': self._agent_locations[i].copy(),
            'other_agents': self._agent_locations[:i] + self._agent_locations[i+1:]
        } for i in range(self.n_agents)]
