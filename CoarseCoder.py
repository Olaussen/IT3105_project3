import numpy as np
import random, math
from Parameters import Parameters
params = Parameters()

class CoarseCoder():
    def initialize(self):
        """
            Initialized the course coder with the bins, offsets and ranges set in Parameters.py
        """
        table = []
        for i in range(params.tilings):
            binn = params.bins[i]
            offset = params.offsets[i]
            tiling = []
            for y in range(params.features):
                newtile = np.linspace(params.feature_ranges[y][0], params.feature_ranges[y][1], binn[y] + 1)[1:-1] + offset[y]
                tiling.append(newtile)
            table.append(tiling)
        self.table = np.array(table)

    def getEncoding(self, state):
        codings = []
        for x in self.table:
            for y in range(params.features):
                feature = state[y]
                tiling = x[y]
                encoding = np.digitize(feature, tiling)
                codings.append(encoding)
        return codings
            
