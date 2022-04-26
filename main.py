
from Acrobat import Acrobat
from ReinforcementLearner import ReinforcementLearner

if __name__ == "__main__":
    rl = ReinforcementLearner(Acrobat())
    rl.run()