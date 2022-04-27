
from Acrobat import Acrobat
from ReinforcementLearner import ReinforcementLearner

if __name__ == "__main__":
    world = Acrobat()
    rl = ReinforcementLearner(world)
    rl.run()
    world.visualizeGame(save_animation=False)
