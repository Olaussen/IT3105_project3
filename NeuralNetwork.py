import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from Parameters import Parameters
params = Parameters()
tf.compat.v1.disable_eager_execution() #Makes tensorflow go faster


class NeuralNetwork():
    def initialize(self):
        """
            This method will initialize a new neural network with the parameters given in Parameters.py
        """
        model = Sequential()

        #Input layer
        model.add(Dense(13, input_dim=13, activation="relu")) 

        #Hidden layer
        for i in range(len(params.network_dims)):
            model.add(Dense(params.network_dims[i], activation=params.network_activations[i])) 
        #Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss=MeanSquaredError(), optimizer=Adam(params.learning_rate),)
        self.network = model
        model.summary()

    def evaluate(self, state: list, action: int):
        """
            This method will use the network to predict the best move to perform in the given state
        """
        input = np.asarray([action] + state)
        return self.network.predict(np.array([input,]))
    
    def train(self, state: list, error: float):
        """
            Uses the given state and error to train the neural network.
        """
        self.network.fit(np.array(state), error, epochs=1, verbose=1)

