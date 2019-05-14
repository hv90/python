#Autor: hv90
#Data: 15/11/2018
#Proposta: simulação do sistema planetário da via láctea utilizando o
           conceito do problema de kepler para dois corpos.


#(Trabalhar em unidades de AU, massa solar e anos)
#G*M = 4(pi)^2
#a = T^(2/3)
#v' = -G*M*r / (|r|^3)
#v1 = v0 + a0 * t
#s1 = s0 + v0 * t
#a_x = 


#a = (GM(T^2)/4pi)^(1/3)
#a = T^(2/3)
#x(t) = a*cos(2pi(t-t0)/T)
#y(t) = a*sin(2pi(t-t0)/T)

#T = 1
#a = np.power(T, 2/3)
#v_orbit = np.sqrt(GM(2/r - 1/a))
#theta = (acos(x/r) + asin(y/r) )/2
#r_orbit0 = a(1-e^2) / (1 + e * cos theta)
#r = r0 + v0 * Dt

#F_x = (-GMm/r^3)*x
#F_y = (-GMm/r^3)*y


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(5, 5)
        

#---------------------------------------------------------------
class Body:
    #-----------------------------------------------------------    
    def __init__(self, name, position, velocity, mass, radius):  
        self.name   = name
        self.pos    = position
        self.vel    = velocity
        self.mass   = mass
        self.radius = radius
    #-----------------------------------------------------------
    def getName(self):
        #print("Body's name: ", self.name)
        return self.name
    #-----------------------------------------------------------
    def getPos(self):
        #print("Body's position: ", self.pos)
        return self.pos
    #-----------------------------------------------------------
    def getVel(self):
        #print("Body's velocity: ", self.vel)
        return self.vel
    #-----------------------------------------------------------
    def getMass(self):
        #print("Body's mass: ", self.mass, "kg")
        return self.mass
    #-----------------------------------------------------------
    def getRadius(self):
        #print("Body's radius: ", self.radius, "m")
        return self.radius
    #-----------------------------------------------------------
    def setPos(self, pos):
        self.pos = pos
    #-----------------------------------------------------------
    def setVel(self, vel):
        self.vel = vel
    #-----------------------------------------------------------
    def Runge_Kutta(self, Dt, pos, vel):
        [x, y] = pos
        [vx, vy] = vel
        #---------------------------------------------------
        def compute_Acceleration(x, y):
            def calculate_FSol(x, y):
                MSol   = 2*10**30
                G      = 6.67*10**(-11)
                r      = (x**2 + y**2)
                if(r == 0):
                    r = 0.01
                FSol = -G*MSol*self.mass/r
                FSol_X = FSol*x/r**(1/2)
                FSol_Y = FSol*y/r**(1/2)

                return FSol_X, FSol_Y
            [Fx, Fy] = calculate_FSol(x, y)
            ax = Fx/self.mass
            ay = Fy/self.mass
            
            return ax, ay
        #-----------------------------------------------------
        def K1(x, y, vx, vy):
            k1x = vx
            k1y = vy
                
            [k1vx, k1vy] = compute_Acceleration(x, y)
            
            return k1x, k1y, k1vx, k1vy
        #----------------------------------------------------
        def K2(x, y, vx, vy, k1, Dt):
            [k1x, k1y, k1vx, k1vy] = k1

            k2x = vx + Dt*k1vx / 2
            k2y = vy + Dt*k1vy / 2

            [k2vx, k2vy] = compute_Acceleration(x + Dt*k1x /2, y + Dt*k1y /2)
            
            return k2x, k2y, k2vx, k2vy
        #----------------------------------------------------
        def K3(x, y, vx, vy, k2, Dt):
            [k2x, k2y, k2vx, k2vy] = k2

            k3x = vx + Dt*k2vx / 2
            k3y = vy + Dt*k2vy / 2

            [k3vx, k3vy] = compute_Acceleration(x + Dt*k2x /2, y + Dt*k2y /2)

            return k3x, k3y, k3vx, k3vy
        #----------------------------------------------------
        def K4(x, y, vx, vy, k3, Dt):
            [k3x, k3y, k3vx, k3vy] = k3

            k4x = vx + Dt*k3vx / 2
            k4y = vy + Dt*k3vy / 2

            [k4vx, k4vy] = compute_Acceleration(x + Dt*k3x /2, y + Dt*k3y /2)

            return k4x, k4y, k4vx, k4vy            
        #---------------------------------------------------------
        k1 = K1(x, y, vx,vy)
        k2 = K2(x, y, vx, vy, k1, Dt)
        k3 = K3(x, y, vx, vy, k2, Dt)
        k4 = K4(x, y, vx, vy, k3, Dt)

        x  = x  + Dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        y  = y  + Dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6

        vx = vx + Dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
        vy = vy + Dt*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

        return [x, y], [vx, vy]
#----------------------------------------------------------------------------
Dt       = 86400
#----------------------------------------------------------------------------
#Terra:
posTerra = [-147095000000,0]
velTerra = [0, -30300]
mTerra   = 6 * 10**24
rTerra   = 6.4 * 10**8 #m
rTerraSol = 150 * 10**9
#----------------------------------------------------------------------------
#Sol:
posSol   = [0,0]
velSol   = [0,0]
mSol     = 2*10**30
rSol     = 700 * 10**8
#----------------------------------------------------------------------------
#Mercury:
posMerc = [-46 * 10**9, 0]
velMerc = [0, -58980]
mMerc   = 0.33 * 10**24
rMerc   = 2.4 * 10**8
#----------------------------------------------------------------------------
#Venus:
posVenus = [-107.5 * 10**9, 0]
velVenus = [0, -35260]
mVenus   = 4.8 * 10**24
rVenus   = 6 * 10**8
#----------------------------------------------------------------------------
#Marte:
posMarte = [-206.6 * 10**9, 0]
velMarte = [0, -26500]
mMarte   = 0.6 * 10**24
rMarte   = 3.4 * 10**8
#----------------------------------------------------------------------------
#Jupiter:
posJup = [-740.5 * 10**9, 0]
velJup = [0, -13720]
mJup   = 1898 * 10**24
rJup   = 71.5 * 10**8
#----------------------------------------------------------------------------
#Saturno:
posSat = [-1352.5 * 10**9, 0]
velSat = [0, -10180]
mSat   = 568 * 10**24
rSat   = 54.3 * 10**8
#----------------------------------------------------------------------------
#Urano:
posUra = [-2741.3 * 10**9, 0]
velUra = [0, -7110]
mUra   = 86.8 * 10**24
rUra   = 25 * 10**8
#----------------------------------------------------------------------------
#Netuno:
posNet = [-4444.45 * 10**9, 0]
velNet = [0, -5500]
mNet   = 102.4 * 10**24
rNet   = 24.3 * 10**8
#----------------------------------------------------------------------------

Terra = Body("Terra", posTerra, velTerra, mTerra, rTerra)
Sol   = Body("Sol", posSol, velSol, mSol, rSol)
Merc  = Body("Mercurio", posMerc, velMerc, mMerc, rMerc)
Venus = Body("Venus", posVenus, velVenus, mVenus, rVenus)
Marte = Body("Marte", posMarte, velMarte, mMarte, rMarte)
Jup = Body("Jupiter", posJup, velJup, mJup, rJup)
Sat = Body("Saturno", posSat, velSat, mSat, rSat)
Ura = Body("Urano", posUra, velUra, mUra, rUra)
Net = Body("Netuno", posNet, velNet, mNet, rNet)
#----------------------------------------------------------------------------
dist = -posNet[0]

axis  = plt.axes(xlim=(-dist*1.5, dist*1.5), ylim=(-dist*1.5, dist*1.5))
bg    = plt.Rectangle((-dist*1.5, -dist*1.5), 3*dist, 3*dist, color='white')
terra = plt.Circle((0, 0), rTerra, color='blue')
sol   = plt.Circle((0, 0), rSol, color='yellow')
merc  = plt.Circle((0, 0), rMerc, color='red')
venus  = plt.Circle((0, 0), rVenus, color='yellow')
marte  = plt.Circle((0, 0), rMarte, color='red')
jup    = plt.Circle((0, 0), rJup, color='orange')
sat    = plt.Circle((0, 0), rSat, color='orange') 
ura    = plt.Circle((0, 0), rUra, color='violet')
net    = plt.Circle((0, 0), rNet, color='blue')
#------------------------------------------------------------------------------
def init():
    bg.center = (0, 0)
    axis.add_patch(bg)
    axis.add_patch(terra)
    axis.add_patch(sol)
    axis.add_patch(merc)
    axis.add_patch(venus)
    axis.add_patch(marte)
    axis.add_patch(jup)
    axis.add_patch(sat)
    axis.add_patch(ura)
    axis.add_patch(net)

    return bg, sol, terra, merc, venus, marte, jup, sat, ura, net

#------------------------------------------------------------------------------
def animate(i):
    xBG, yBG       = bg.center
    xSol, ySol     = sol.center
    
    [posTerra, velTerra] = Terra.Runge_Kutta(Dt, Terra.getPos(), Terra.getVel())
    Terra.setPos(posTerra)
    Terra.setVel(velTerra)
    
    [posMerc, velMerc] = Merc.Runge_Kutta(Dt, Merc.getPos(), Merc.getVel())
    Merc.setPos(posMerc)
    Merc.setVel(velMerc)
    
    [posVenus, velVenus] = Venus.Runge_Kutta(Dt, Venus.getPos(), Venus.getVel())
    Venus.setPos(posVenus)
    Venus.setVel(velVenus)

    [posMarte, velMarte] = Marte.Runge_Kutta(Dt, Marte.getPos(), Marte.getVel())
    Marte.setPos(posMarte)
    Marte.setVel(velMarte)

    [posJup, velJup] = Jup.Runge_Kutta(Dt, Jup.getPos(), Jup.getVel())
    Jup.setPos(posJup)
    Jup.setVel(velJup)
    
    [posSat, velSat] = Sat.Runge_Kutta(Dt, Sat.getPos(), Sat.getVel())
    Sat.setPos(posSat)
    Sat.setVel(velSat)

    [posUra, velUra] = Ura.Runge_Kutta(Dt, Ura.getPos(), Ura.getVel())
    Ura.setPos(posUra)
    Ura.setVel(velUra)

    [posNet, velNet] = Ura.Runge_Kutta(Dt, Net.getPos(), Net.getVel())
    Net.setPos(posNet)
    Net.setVel(velNet)
    
    terra.center = (posTerra)
    merc.center  = (posMerc)
    venus.center = (posVenus)
    marte.center = (posMarte)
    jup.center   = (posJup)
    sat.center   = (posSat)
    ura.center   = (posUra)
    net.center   = (posNet)

    return bg, sol, terra, merc, venus, marte, jup, sat, ura, net

#--------------------------------------------------------------
anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=360, 
                               interval=20,
                               blit=True)

plt.show()

  

