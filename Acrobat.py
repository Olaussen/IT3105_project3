import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Parameters import Parameters

params = Parameters()

class Acrobat:

    def initialize(self):        
        if hasattr(self, 'geometryAllGames'): # save geometry from last run
            self.geometryAllGames.append(self.geometryCurrentGame)            
        else:
            self.geometryAllGames = []
        self.geometryCurrentGame = []

        self.angle1 = 0
        self.angle2 = 0
        self.vel1 = 0
        self.vel2 = 0
        self.angle3 = 0
        self.xp1 = 10
        self.xp2 = 0
        self.yp1 = 10
        self.yp2 = 0
        self.xtip = 0
        self.ytip = 0
        
        self.ygoal = self.yp1 + params.pole_two_length

    def current_state(self) -> tuple:
        return (self.angle1, self.vel1, self.angle2, self.vel2)
    
    def is_end_state(self) -> bool:
        return self.ytip >= self.ygoal

    def perform_action(self, force: int):

        mass1 = params.pole_one_mass
        mass2 = params.pole_two_length
        length1 = params.pole_one_length
        length2 = params.pole_two_length
        center1 = length1 / 2
        center2 = length2 / 2
        gravity = params.gravity
        timestep = params.timestep

        # ok
        phi2 =  mass2 * center2 *  gravity * math.cos(self.angle1 + self.angle2 - (math.pi/2))

        # small change for breaking over lines
        phi1 = (-mass2 * length1 * center2 * self.vel2**2 * math.sin(self.angle2)) \
        - (2 * mass2 * length1 * center2 * self.vel2 * self.vel1 * math.sin(self.angle2)) \
        + (((mass1 * center1) + (mass2 * length1)) * gravity * math.cos(self.angle1 - (math.pi/2))) + phi2

        # ok
        inner = center2**2 + (length1 * center2 * math.cos(self.angle2))
        d2 = (mass2 * inner) + 1

        # ok
        inner = length1**2 + center2**2 + (2 * length1 * center2 * math.cos(self.angle2))
        d1 = (mass1 * center1**2) + (mass2 * inner) + 2

        
        # ok
        inner1 = (mass2 * center2**2) + (1 - (d2**2 / d1))
        inner2 = force + ((d2 / d1) * phi1) - (mass2 * length1 * center2 * self.vel1**2 * math.sin(self.angle2)) - phi2
        acc2 = (1 / inner1) * inner2
        acc1 = -((d2 * acc2) + phi1) / d1
        

        # ok
        self.vel2 += timestep * acc2
        self.vel1 += timestep * acc1
        self.angle2 += timestep * self.vel2
        self.angle1 += timestep * self.vel1

        # ok
        self.angle3 = self.angle1 + self.angle2
        self.xp2 = self.xp1 + (length1 * math.sin(self.angle1))
        self.yp2 = self.yp1 - (length1 * math.cos(self.angle1))
        self.xtip = self.xp2 + (length2 * math.sin(self.angle3))
        self.ytip = self.yp2 - (length2 * math.cos(self.angle3))

        if self.vel1 > math.pi * 9:
            self.vel1 = math.pi * 9
        
        if self.vel1 < -math.pi * 9:
            self.vel1 = -math.pi * 9

        if self.vel2 > math.pi * 9:
            self.vel2 = math.pi * 9
        
        if self.vel2 < -math.pi * 9:
            self.vel2 = -math.pi * 9

        if self.angle1 > math.pi * 2:
            self.angle1 = self.angle1 % math.pi * 2
        
        if self.angle1 < -math.pi * 2:
            self.angle1 = -(self.angle1 % math.pi * 2)

        if self.angle2 > math.pi * 2:
            self.angle2 = self.angle2 % math.pi * 2
        
        if self.angle2 < -math.pi * 2:
            self.angle2 = -(self.angle2 % math.pi * 2)

        self.geometryCurrentGame.append((self.xp2, self.yp2, self.xtip, self.ytip))

    def legal_actions(self) -> tuple:
        return (params.force, -params.force, 0) #-1 is called 2 so we don't send negative values into neural network

    def reward(self) -> int:
        return -1

    def visualize_game( self, game=-1, save_animation=True, filename='animation.gif' ):
        c = { # color palette
            'medium-grey': (218/255, 218/255, 218/255),
            'dark-grey': (68/255, 79/255, 85/255),
            'blue': (0/255, 142/255, 194/255),
            'green': (93/255, 184/255, 46/255),
            'red': (237/255, 28/255, 46/255),
            'orange': (255/255, 150/255, 0/255),
            'yellow': (255/255, 213/255, 32/255),
            'purple': (112/255, 48/255, 160/255),
            'black': (0, 0, 0),
            'white': (1, 1, 1)
        }
        c_ind = ['dark-grey','black', 'red', 'medium-grey', 'white', 'blue']
        
        scale_xlim = 1.5
        scale_ylim = 1.5
        r_hinge = 9e-2
        r_hinge_inner = .7*r_hinge
        dx = r_hinge - r_hinge_inner
        repeat_last_frame = 50

        #self.ygoal
        max_len = params.pole_one_length + params.pole_two_length
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
            X.append([self.xp1, scene[0], scene[2]])
            Y.append([self.yp1, scene[1], scene[3]])
        
        for i in range(repeat_last_frame):
            X.append([self.xp1, scenes[-1][0], scenes[-1][2]])
            Y.append([self.yp1, scenes[-1][1], scenes[-1][3]])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlim=(x_lim_low, x_lim_high), ylim=(y_lim_low, y_lim_high))
        ax.axis('equal')
        ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
        # draw fixed elements
        ax.plot([self.xp1-max_len, self.xp1+max_len], [self.ygoal,self.ygoal], color=c[c_ind[2]], lw=1) # y_limit
        
        circle0 = plt.Circle((self.xp1, self.yp1), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle1 = plt.Circle((self.xp1, self.yp1), r_hinge_inner, color=c[c_ind[3]], zorder=5)
        ax.add_patch(circle0)
        ax.add_patch(circle1)
        
        # draw acrobat
        line = ax.plot(X[0], Y[0], color=c[c_ind[0]], lw=6, zorder=1)[0]

        #p2 hinge
        circle2 = plt.Circle((X[0][1], Y[0][1]), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle3 = plt.Circle((X[0][1], Y[0][1]), r_hinge_inner, color=c[c_ind[3]], zorder=5)
        ax.add_patch(circle2)
        ax.add_patch(circle3)

        # tip
        circle4 = plt.Circle((X[0][2], Y[0][2]), r_hinge, color=c[c_ind[1]], zorder=5) # border
        circle5 = plt.Circle((X[0][2], Y[0][2]), r_hinge_inner, color=c[c_ind[3]], zorder=5)
        ax.add_patch(circle4)
        ax.add_patch(circle5)

        # progress bar
        w_full = self.xp1 - (self.xp1-max_len)
        height = 2*r_hinge
        w_inner = w_full-2*dx

        rect1 = plt.Rectangle((self.xp1-max_len+dx, y_lim_low), w_full, height, color=c[c_ind[1]], zorder=1) # border
        rect2 = plt.Rectangle((self.xp1-max_len+2*dx, y_lim_low+dx), w_inner, height-2*dx, color=c[c_ind[4]],zorder=3) # border
        rect3 = plt.Rectangle((self.xp1-max_len+2*dx, y_lim_low+dx), 0, height-2*dx, color=c[c_ind[5]],zorder=5)
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # progress text
        t_width = len(str(len(scenes)))
        base_txt = 'control step '
        label = ax.text(self.xp1 + 20*dx, y_lim_low + r_hinge, base_txt + str(0).ljust(t_width) + '/' + str(len(scenes))  , ha='left', va='center', fontsize=12)

        def animate(i):
            line.set_xdata(X[i])
            line.set_ydata(Y[i])

            circle2.center = X[i][1], Y[i][1] # hinge 2
            circle3.center = X[i][1], Y[i][1]

            circle4.center = X[i][2], Y[i][2] # tip
            circle5.center = X[i][2], Y[i][2]

            w_fract = min(1, i/len(scenes))
            rect3.set(width=w_fract*w_inner)

            step = min(i, len(scenes))
            label.set_text(base_txt + str(step).ljust(t_width) + '/' + str(len(scenes)))

        anim = FuncAnimation(fig, animate, interval=100, frames=len(Y)-1)
        
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
        a.perform_action(random.choice(a.legal_actions()))
    a.visualize_game(save_animation=False)
    