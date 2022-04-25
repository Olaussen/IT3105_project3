import numpy as np
import random

class CoarseCoder():

    def __init__(self):

        self.tilings = 2
        self.bins = [[7, 7, 7, 7], [7, 7, 7, 7]] #Bins per features
        self.offsets = [[0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]] #Offsets per dimension
        self.features = 4
        self.feature_ranges = [[-10, 10], [-10, 10],[-10, 10], [-10, 10]]

    def initializeTiling(self):

        table = []

        for i in range(self.tilings):
            
            binn = self.bins[i]
            offset = self.offsets[i]

            tiling = []

            for y in range(self.features):
                newtile = np.linspace(self.feature_ranges[y][0], self.feature_ranges[y][1], binn[y] + 1)[1:-1] + offset[y]
                tiling.append(newtile)
            
            table.append(tiling)
        
        self.table = np.array(table)

    def getEncoding(self, state):

        codings = []

        for x in self.table:

            for y in range(self.features):

                feature = state[y]
                tiling = x[y]
                encoding = np.digitize(feature, tiling)
                codings.append(encoding)
        
        return codings

            
