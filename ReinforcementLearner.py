import random
import matplotlib.pyplot as plt
from Acrobat import Acrobat
from NeuralNetwork import NeuralNetwork
from CoarseCoder import CoarseCoder
from Parameters import Parameters
params = Parameters()

class ReinforcementLearner():

    def __init__(self, game: Acrobat):
        self.game = game
        self.state = None
        self.seen = False
        self.epsilon = params.epsilon

    #Returns best action according to network with value
    #If all values are equally good, return a random one with value
    def get_best_network_action(self, state: tuple):
        actions = self.game.legal_actions()
        evaluations = {}
        seen = []

        for action in actions:
            eval = self.network.evaluate(state, action)
            evaluations[str(action)] = eval
            seen.append(float(eval))

        if not self.seen:
            print(evaluations)
            self.seen = True

        if (len(set(seen)) > 1) and random.random() > self.epsilon:
            key = max(evaluations, key=evaluations.get)
            return int(key), evaluations[key]
        action = random.choice(actions)
        return action, evaluations[str(action)]


    def run(self):
        #Create a neural network and coarse coder
        self.network = NeuralNetwork()
        encoder = CoarseCoder()
        #Keeps track of steps to win for stat display at the end
        episodes = []
        total = 0

        #Play the game until all episodes are run
        for i in range(params.num_episodes):
            timesteps_to_win = 0
            x_train = []
            y_train = []
            #Initialize a new game
            self.game.initialize() 
            #Get current state
            self.state = self.game.current_state()
            #Encode state with coarse coding
            encoded_state = encoder.get_encoding(self.state)
            stored_encoded = encoded_state
            #Get best action given state
            action, _ = self.get_best_network_action(encoded_state)

            while not self.game.is_end_state():
                self.game.perform_action(action)
                #Get current state
                self.state = self.game.current_state()
                #Encode state with coarse coding
                encoded_state = encoder.get_encoding(self.state)
                #Get best action given state
                next_action, next_action_evaluation = self.get_best_network_action(encoded_state)
                #Get reward
                reward = self.game.reward()
                #Train chosen state on reward + discount * value of the next best move
                error = reward + (params.discount * next_action_evaluation)
                x_train.append([action] + stored_encoded)
                y_train.append(int(error))
                
                #Update for next loop
                action = next_action
                timesteps_to_win += 1
                stored_encoded = encoded_state
                total += 1

                if timesteps_to_win == 1000:
                    break
            
            new_x_train = []
            new_y_train = []
            for i in range(len(x_train)):
                data = x_train[i]
                label = y_train[i]
                if data not in new_x_train:
                    new_x_train.append(data)
                    new_y_train.append(label)
                else:
                    key = self.search(new_x_train, data)
                    new_y_train[key] = (new_y_train[key] + label)/2

            self.network.train(x_train, y_train)
            episodes.append(timesteps_to_win)
            print(f"Played game: {i +1}")
            print(f"Timesteps to win: {timesteps_to_win}")
            print(f"Avg. timesteps to win: {total / len(episodes)}\n")

            if timesteps_to_win < 100:
                self.game.visualize_game(save_animation=False)

            self.epsilon *= params.epsilon_decay

        self.show_development(episodes)

    def search(self, train, key):
        for i in range(len(train)):
            if train[i] == key:
                return i

    def show_development(self, stats):
        episodes = []
        
        for i in range(len(stats)):
            episodes.append(i)

        plt.plot(episodes, stats)
        plt.ylabel("Timesteps")
        plt.xlabel("Episodes")
        plt.show()
