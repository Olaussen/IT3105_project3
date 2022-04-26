import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from Parameters import Parameters
tf.compat.v1.disable_eager_execution() #Makes tensorflow go faster

params = Parameters()

class NeuralNetwork():

    def __init__(self):
        self.initialize()

    def initialize(self):
        model = Sequential()

        #Input layer
        model.add(Dense(5, input_dim=5, activation="relu"))
        #Hidden layers
        for i in range(len(params.network_dims)):
            model.add(Dense(params.network_dims[i], activation=params.network_activations[i]))
        #Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss=MeanSquaredError(), optimizer=Adam(params.learning_rate))
        model.summary()
        self.network = model

    def evaluate(self, state, action):
        state = np.asarray([action] + state)
        return self.network.predict(np.array([state,]))
    
    def train(self, state, error):

        self.network.fit(np.array(state), error, epochs=1, verbose=1)

