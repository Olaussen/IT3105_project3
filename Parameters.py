
class Parameters:

    def __init__(self):
        # Learning parameters
        self.num_episodes = 50
        self.learning_rate = 0.001
        self.network_dims = (5,) # has to be tuple
        self.network_activations = ("relu",) # has to be tuple and same length as network_dims
        self.discount = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9
        self.max_limit = 1000

        # Coarse coder parameters
        self.tilings = 3
        self.bins = [[10, 3, 12, 2], [3, 12, 3, 3], [3, 3, 3, 10]] #Bins per features
        self.offsets = [[0, 0, 0, 0], [-2.2, -2.2, -2.2, -2.2], [0.06, 0.06, 0.06, 0.06]] #Offsets per dimension
        self.features = 4
        self.feature_ranges = [[-12, 12], [-10, 10], [-12, 12], [-10, 10]]

        # World parameters
        self.gravity = 9.8
        self.pole_one_length, self.pole_two_length = 1, 1 # one | two
        self.pole_one_mass, self.pole_two_mass = 1, 1     # one | two
        self.timestep = 0.05
        self.force = 1