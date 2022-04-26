import random
import matplotlib.pyplot as plt
from Acrobat import Acrobat
from NeuralNetwork import Neural_Network
from CoarseCoder import CoarseCoder
import tensorflow as tf
import numpy as np


class ReinforcementLearner():

    def __init__(self, episodesToRun, game):
        self.episodesToRun = episodesToRun
        self.game = game
        self.state = None
        self.seen = False
        self.discount = 0.9
        self.epsilon = 1
        self.epsilondecay = 0.9
        random.seed(42)
        tf.random.set_seed(42)
        np.random.seed(42)

    #Returns best action according to network with value
    #If all values are equally good, return a random one with value
    def getBestNetworkAction(self, encodedState):

        actions = self.game.possibleActions()
        evaluations = {}
        seen = []

        for x in actions:
            eval = self.network.evaluate(encodedState, x)
            evaluations[str(x)] = eval
            seen.append(float(eval))

        if not self.seen:
            print(evaluations)
            self.seen = True

        #print(evaluations)


        if (len(set(seen)) > 1) and random.uniform(0,1) > self.epsilon:
            #print("hi")
            key = max(evaluations, key=evaluations.get)
            return int(key), evaluations[key]
        else:
            b = random.choice(actions)
            return b, evaluations[str(b)]


    def runAlgorithm(self):

        #Create a neural network
        self.network = Neural_Network()
        self.network.initializeNetwork()

        #Create encoder
        encoder = CoarseCoder()
        encoder.initializeTiling()

        #Keeps track of steps to win for stat display at the end
        episodes = []
        summer = 0

        #Play the game until all episodes are run
        for x in range(0, self.episodesToRun):

            self.epsilon = self.epsilon * self.epsilondecay
            
            #Initialize a new game
            self.game.initialize() 
            times = 0

            #Get current state
            self.state = self.game.currentState()

            #Encode state with coarse coding
            cutState = self.state
            encodedState = encoder.getEncoding(cutState)
            x_train = []
            y_train = []

            #Get best action given state
            action, prevActionValue = self.getBestNetworkAction(encodedState)

            while self.game.running():
                
                self.game.makeAction(action)

                #Get current state
                self.state = self.game.currentState()

                #Encode state with coarse coding
                cutState = self.state
                nextEncodedState = encoder.getEncoding(cutState)

                #Get best action given state
                nextAction, nextActionValue = self.getBestNetworkAction(nextEncodedState)

                #Get reward
                reward = self.game.reward()

                #Train chosen state on reward + discount * value of the next best move
                stateToUpdate = [action]
                stateToUpdate.extend(encodedState)
                #if (reward > 0): nextActionValue = 0
                #error = prevActionValue + (0.2*(reward + (self.discount * nextActionValue) - prevActionValue))
                error = reward + (self.discount * nextActionValue)

                #self.network.train(stateToUpdate, error)
                #if reward == -1:
                x_train.append(stateToUpdate)
                y_train.append(int(error))
                
                #Update for next loop
                action = nextAction
                encodedState = nextEncodedState
                times += 1
                summer += 1

                if times == 1000:
                    break
            
            new_x_train = []
            new_y_train = []

            for z in range(0, len(x_train)):
                if x_train[z] not in new_x_train:
                    new_x_train.append(x_train[z])
                    new_y_train.append(y_train[z])
                else:
                    key = self.search(new_x_train, x_train[z])
                    new_y_train[key] = (new_y_train[key] + y_train[z])/2

            self.network.train(x_train, y_train)
            print("training")
            episodes.append(times)
            print("Played game: " + str(x))
            print("Timesteps to win: " + str(times))
            print("Avg. timesteps to win: " + str(summer/len(episodes)))

            if times < 100:
                self.game.visualizeGame(save_animation=False)

        self.showDevelopment(episodes)

    def search(self, train, key):

        for x in range(0, len(train)):
            if train[x] == key:
                return x


    def showDevelopment(self, stats):
        episodes = []
        
        for x in range(0, len(stats)):
            episodes.append(x+1)

        plt.plot(episodes, stats)
        plt.ylabel("Timesteps")
        plt.xlabel("Episodes")
        plt.show()


learner = ReinforcementLearner(100, Acrobat())
learner.runAlgorithm()
