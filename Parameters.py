
class Parameters:

    def __init__(self):
        # Learning parameters
        self.num_episodes = 100
        self.learning_rate = 0.1
        self.network_dims = (5,) # has to be tuple
        self.discount = 0.9
        self.epsilon = 0.1
        self.epsilon_decay = 0.9

        # World parameters
        self.gravity = 9.8
        self.pole_one_length, self.pole_two_length = 1, 1 # one | two
        self.pole_one_mass, self.pole_two_mass = 1, 1     # one | two
        self.timestep = 0.05 * 4
        self.force = 1