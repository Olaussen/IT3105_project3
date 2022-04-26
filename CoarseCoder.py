import numpy as np
import random, math

class CoarseCoder():

    def __init__(self):

        self.tilings = 1
        self.bins = [[10, 10, 10, 10], [10, 10, 10, 10], [100, 100, 100, 100], [30, 30, 30, 30]] #Bins per features
        self.offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.23, 0.44, 0.32, 1.4]] #Offsets per dimension
        self.features = 4
        self.feature_ranges = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]

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

        #Used for testing non-tiling based approach
        '''takeover = []

        for x in state:
            takeover.append(round(x*10)) #100
        
        print(takeover)
        return takeover'''
        return codings
            
