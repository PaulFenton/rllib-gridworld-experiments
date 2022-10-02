import uuid
import numpy as np

import pygame
import gym

from gym import spaces
import random

from ray.rllib.env.env_context import EnvContext

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
NOOP = 4

agent_colors = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}

class MultiAgentGrid(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):

        self.size = config['size']  # The size of the square grid
        self.n_robots = config['n_robots']
        self.wins = [0 for i in range(self.n_robots)]
        self.cumulative_reward = 0

        self.window_size = 512  # The size of the PyGame window
        self.metadata = { "render_fps": 10 }

        #self.action_space = spaces.Discrete(4)
        action_space = [5 for n in range(self.n_robots)]
        self.action_space = spaces.MultiDiscrete(action_space)


        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Box(0, self.size - 1, shape=(self.n_robots*2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        self._action_to_direction = {
            RIGHT: np.array([1, 0]),
            UP: np.array([0, 1]),
            LEFT: np.array([-1, 0]),
            DOWN: np.array([0, -1]),
            NOOP: np.array([0, 0])
        }

        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(config.worker_index * config.num_workers)
        self.rng = np.random.default_rng()

        print("ENV CONFIG---- ", config)

        # rendering options
        self.window = None
        self.screen = None
        self.canvas = None
        self.clock = None
        self.render_this = False
        self.render_mode = config['render_mode']
        if self.render_mode == 'all':
            self.render_this = True
        elif self.render_mode == 'first' and config.worker_index == 0:
            self.render_this = True
        self.recording_name = f"recording_{config.worker_index}_{str(uuid.uuid4())[:6]}.mp4"
        self.frames = []

    def _get_obs(self):
        return { "agents": self._agent_locations, "target": self._target_location }

    def _get_info(self):
        return {}

    # def _get_distance(self):
    #     return np.sum(np.abs(self._agent_location - self._target_location))

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cumulative_reward = 0

        # choose relevant locations
        locations = self.rng.choice(self.size**2, 1 + self.n_robots, replace=False)

        self._target_location = np.array([locations[0] % self.size, locations[0] // self.size])
        agent_locations = []
        for i in range(self.n_robots):
            agent_locations.append(np.array([locations[i+1] % self.size, locations[i+1] // self.size]))
        self._agent_locations = np.array(agent_locations).flatten()


        if self.render_this:
            self._render_frame()
        
        return self._get_obs()


    def step(self, action):

        rewards = 0.0
        updated_locations = []
        current_locations = self._agent_locations.reshape((self.n_robots, 2))
        done = False

        for i in range(self.n_robots):
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action[i]]

            next_location = np.clip(
                current_locations[i] + direction, 0, self.size - 1
            )
            updated_locations.append(next_location)


            # An episode is done iff the agent has reached the target
            if np.array_equal(next_location, self._target_location):
                self.wins[i] += 1
                rewards += 10.0
                done = True
            else:
                if action[i] == NOOP:
                    rewards += -0.01
                else:
                    rewards += -1.0

        self._agent_locations = np.array(updated_locations).flatten()
        self.cumulative_reward += rewards
        if self.render_this:
            self._render_frame()

        return self._get_obs(), rewards, done, self._get_info()


    def _render_frame(self):
        TEXT_HEIGHT = 16
        PADDING = 5


        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.clock.tick(self.metadata["render_fps"])

        full_canvas = pygame.Surface((self.window_size, self.window_size + TEXT_HEIGHT + PADDING))
        full_canvas.fill((0, 0, 0))
        
        # write summary stats
        summary = f"Reward: {self.cumulative_reward:12.2f}, Wins: "
        for i in range(self.n_robots):
            if i != 0:
                summary += ", "
            summary += f"{i:4d}: {self.wins[i]}"

        font = pygame.font.SysFont("monospace", TEXT_HEIGHT)
        text = font.render(summary, True, (255, 255, 0), (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (text.get_width() / 2, text.get_height() / 2)
        full_canvas.blit(text, textRect)
        
        
        
        
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agents
        for i, agent in enumerate(self._agent_locations.reshape((self.n_robots, 2))):
            pygame.draw.circle(
                canvas,
                agent_colors[i],
                (agent + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (255,255,255),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (255,255,255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        full_canvas.blit(canvas, (0, TEXT_HEIGHT + PADDING))

        self.canvas = full_canvas
        self.frames.append(np.transpose(
                np.array(pygame.surfarray.pixels3d(full_canvas)), axes=(1, 0, 2)
            ))

    def seed(self, seed=None):
        random.seed(seed)

    def render(self):
        if not self.render_this:
            print("Skipping rendering on worker")
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        # if self.screen is None:
        #     pygame.init()
        #     pygame.display.init()
        #     self.screen = pygame.display.set_mode(
        #         (self.window_size, self.window_size)
        #     )
    


        # self.screen.blit(self.canvas, (0, 0))
        # pygame.event.pump()
        # pygame.display.flip()


    def close(self):
        if len(self.frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )
            clip = ImageSequenceClip(self.frames, fps=10)
            clip.write_videofile(self.recording_name)