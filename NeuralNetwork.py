import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution() #Makes tensorflow go faster

class Neural_Network():

    def __init__(self):

        self.network = None

    def initializeNetwork(self):

        model = tf.keras.Sequential()

        initializer = tf.keras.initializers.RandomNormal(mean=0.2, stddev=1.2, seed=40)


        #Input layer
        model.add(tf.keras.layers.Dense(9, input_dim=9, activation="relu", kernel_initializer=initializer))

        #Hidden layer
        #model.add(tf.keras.layers.Dense(16, activation="relu"))

        #Output layer
        model.add(tf.keras.layers.Dense(1, activation="relu", kernel_initializer=initializer))

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=[tf.keras.metrics.Accuracy()])
        self.network = model
        model.summary()

    def evaluate(self, state, action):

        action = [action]
        action.extend(state)
        state = np.asarray(action)

        return self.network.predict(np.array([state,]))
    
    def train(self, state, error):

        self.network.fit(np.array(state), error, epochs=10, verbose=1)
