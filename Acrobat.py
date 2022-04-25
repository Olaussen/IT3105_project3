import math
import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Acrobat():
    def __init__(self):
        pass # initialize() called from learner

    def initialize(self):        
        if hasattr(self, 'geometryAllGames'): # save geometry from last run
            self.geometryAllGames.append(self.geometryCurrentGame)            
        else:
            self.geometryAllGames = []
        self.geometryCurrentGame = []

        self.angle1 = 0
        self.rotVel1 = 0
        self.angle2 = 0
        self.rotVel2 = 0
        self.phi2 = 0
        self.m2 = 1
        self.L1 = 1
        self.Lc2 = 1/2
        self.g = 9.8
        self.m1 = 1
        self.Lc1 = 1/2
        self.d2 = 0
        self.F = 0
        self.timestep = 0.05 * 4
        self.L2 = 1
        self.angle3 = 0
        self.xp1 = 10
        self.xp2 = 0
        self.yp1 = 10
        self.yp2 = 0
        self.xtip = 0
        self.ytip = 0
        
        self.ygoal = self.yp1 + self.L2 

        self.makeAction(0)

    def currentState(self):
        return [self.angle1, self.rotVel1, self.angle2, self.rotVel2]

    def currentGeometry(self):
        return [self.xp2, self.yp2, self.xtip, self.ytip] # p1 constant
    
    def running(self):
        return self.ytip < self.ygoal
        #return abs(self.angle2) < (math.pi/2) # debugging

    def makeAction(self, action):

        #Avoids sending negative values into neural network
        if action == 2:
            action == -1

        self.F = action

        # ok
        self.phi2 = self.m2 * self.Lc2 * self.g * math.cos(self.angle1 + self.angle2 - (math.pi/2))

        # small change for breaking over lines
        self.phi1 = (-self.m2 * self.L1 * self.Lc2 * self.rotVel2**2 * math.sin(self.angle2)) \
        - (2 * self.m2 * self.L1 * self.Lc2 * self.rotVel2 * self.rotVel1 * math.sin(self.angle2)) \
        + (((self.m1 * self.Lc1) + (self.m2 * self.L1)) * self.g * math.cos(self.angle1 - (math.pi/2))) + self.phi2

        # ok
        tmp = self.Lc2**2 + (self.L1 * self.Lc2 * math.cos(self.angle2))
        self.d2 = (self.m2 * tmp) + 1

        # ok
        tmp1 = self.L1**2 + self.Lc2**2 + (2*self.L1*self.Lc2*math.cos(self.angle2))
        self.d1 = (self.m1 * self.Lc1**2) + (self.m2 * tmp1) + 2

        
        # ok
        tmp2 = (self.m2 * self.Lc2**2) + (1 - (self.d2**2/self.d1))
        tmp3 = self.F + ((self.d2/self.d1)*self.phi1) - (self.m2 * self.L1 * self.Lc2 * self.rotVel1**2 * math.sin(self.angle2)) - self.phi2
        self.rotAcc2 = (1/tmp2) * tmp3
        self.rotAcc1 = -(((self.d2*self.rotAcc2) + self.phi1)/self.d1)
        

        # ok
        self.rotVel2 = self.rotVel2 + (self.timestep * self.rotAcc2)
        self.rotVel1 = self.rotVel1 + (self.timestep * self.rotAcc1)
        self.angle2 = self.angle2 + (self.timestep * self.rotVel2)
        self.angle1 = self.angle1 + (self.timestep * self.rotVel1)

        # ok
        self.angle3 = self.angle1 + self.angle2
        self.xp2 = self.xp1 + (self.L1 * math.sin(self.angle1))
        self.yp2 = self.yp1 - (self.L1 * math.cos(self.angle1))
        self.xtip = self.xp2 + (self.L2 * math.sin(self.angle3))
        self.ytip = self.yp2 - (self.L2 * math.cos(self.angle3))

        if self.rotVel1 > math.pi*9:
            self.rotVel1 = math.pi*9
        
        if self.rotVel1 < -math.pi*9:
            self.rotVel1 = -math.pi*9

        if self.rotVel2 > math.pi*9:
            self.rotVel2 = math.pi*9
        
        if self.rotVel2 < -math.pi*9:
            self.rotVel2 = -math.pi*9

        self.geometryCurrentGame.append(self.currentGeometry())

    def possibleActions(self):
        return [1, 2, 0] #-1 is called 2 so we don't send negative values into neural network

    def reward(self):
        if self.running():
            return 0
        else:
            return 150

    def visualizeGame( self, game=-1, save_animation=True, filename='animation.gif' ):
        c = { # color palette
            'medium-grey': ( 218/255, 218/255, 218/255),
            'dark-grey': ( 68/255, 79/255, 85/255),
            'blue': ( 0/255, 142/255, 194/255),
            'green': ( 93/255, 184/255, 46/255),
            'red': ( 237/255, 28/255, 46/255),
            'orange': ( 255/255, 150/255, 0/255),
            'yellow': ( 255/255, 213/255, 32/255),
            'purple': ( 112/255, 48/255, 160/255),
            'black': ( 0, 0, 0),
            'white': ( 1, 1, 1)
        }
        c_ind = ['dark-grey','black', 'red', 'medium-grey', 'white', 'blue']
        
        scale_xlim = 1.5
        scale_ylim = 1.5
        r_hinge = 9e-2
        r_hinge_inner = .7*r_hinge
        dx = r_hinge - r_hinge_inner
        repeat_last_frame = 50

        #self.ygoal
        max_len = self.L1 + self.L2
        scaled_max_len = max_len*scale_xlim
        x_lim_low = self.xp1-scaled_max_len
        x_lim_high = self.xp1+scaled_max_len        
        y_lim_low = self.yp1-max_len*scale_ylim
        y_lim_high = max(self.yp1+max_len*scale_ylim, self.ygoal*scale_ylim)
        
        scenes = self.geometryAllGames[game] if game!=-1 else self.geometryCurrentGame
        X = []
        Y = []

        # create data to plot
        for scene in scenes: # scene: [x2,y2,xt,yt]
            X.append( [ self.xp1, scene[0], scene[2] ] )
            Y.append( [ self.yp1, scene[1], scene[3] ] )
        
        for i in range(repeat_last_frame):
            X.append( [ self.xp1, scenes[-1][0], scenes[-1][2] ] )
            Y.append( [ self.yp1, scenes[-1][1], scenes[-1][3] ] )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlim=(x_lim_low, x_lim_high), ylim=(y_lim_low, y_lim_high))
        ax.axis( 'equal' )
        ax.axis( 'off' )
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
        # draw fixed elements
        ax.plot([self.xp1-max_len, self.xp1+max_len], [self.ygoal,self.ygoal], color=c[c_ind[2]], lw=1) # y_limit
        
        circle0 = plt.Circle( (self.xp1, self.yp1), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle1 = plt.Circle( (self.xp1, self.yp1), r_hinge_inner, color=c[c_ind[3]], zorder=5 )
        ax.add_patch( circle0 )
        ax.add_patch( circle1 )
        
        # draw acrobat
        line = ax.plot(X[0], Y[0], color=c[c_ind[0]], lw=6, zorder=1)[0]

        #p2 hinge
        circle2 = plt.Circle( (X[0][1], Y[0][1]), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle3 = plt.Circle( (X[0][1], Y[0][1]), r_hinge_inner, color=c[c_ind[3]], zorder=5 )
        ax.add_patch( circle2 )
        ax.add_patch( circle3 )

        # tip
        circle4 = plt.Circle( (X[0][2], Y[0][2]), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle5 = plt.Circle( (X[0][2], Y[0][2]), r_hinge_inner, color=c[c_ind[3]], zorder=5 )
        ax.add_patch( circle4 )
        ax.add_patch( circle5 )

        # progress bar
        w_full = self.xp1 - (self.xp1-max_len)
        height = 2*r_hinge
        w_inner = w_full-2*dx

        rect1 = plt.Rectangle( (self.xp1-max_len+dx, y_lim_low), w_full, height, color=c[c_ind[1]], zorder=1 ) # border
        rect2 = plt.Rectangle( (self.xp1-max_len+2*dx, y_lim_low+dx), w_inner, height-2*dx, color=c[c_ind[4]],zorder=3 ) # border
        rect3 = plt.Rectangle( (self.xp1-max_len+2*dx, y_lim_low+dx), 0, height-2*dx, color=c[c_ind[5]],zorder=5 )
        
        ax.add_patch( rect1 )
        ax.add_patch( rect2 )
        ax.add_patch( rect3 )

        # progress text
        t_width = len(str(len(scenes)))
        base_txt = 'control step '
        label = ax.text( self.xp1 + 20*dx, y_lim_low + r_hinge, base_txt + str(0).ljust(t_width) + '/' + str(len(scenes))  , ha='left', va='center', fontsize=12 )

        def animate(i):
            line.set_xdata(X[i])
            line.set_ydata(Y[i])

            circle2.center = X[i][1], Y[i][1] # hinge 2
            circle3.center = X[i][1], Y[i][1]

            circle4.center = X[i][2], Y[i][2] # tip
            circle5.center = X[i][2], Y[i][2]

            w_fract = min(1,i/len(scenes))
            rect3.set(width=w_fract*w_inner)

            step = min( i,len(scenes) )
            label.set_text(base_txt + str(step).ljust(t_width) + '/' + str(len(scenes)))

        anim = FuncAnimation(
            fig, animate, interval=100, frames=len(Y)-1)
        
        if save_animation:            
            anim.save(filename)
        else:
            plt.draw()
            plt.show()


if __name__=='__main__':
    moves = 200
    a = Acrobat()
    a.initialize()
    for i in range(moves):
        a.makeAction(random.choice(a.possibleActions()))
    a.visualizeGame( save_animation=False )
    