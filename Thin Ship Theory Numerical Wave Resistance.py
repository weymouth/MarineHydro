# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:26:52 2020

@author: mclea
"""

# from stl import mesh
import numpy as np
from scipy import optimize
# from wave_components import wave_components
# from source_strength import source_strength
from stl import mesh as stl_mesh


class tank_properties:

    def __init__(self):
        self.g = 9.81  # Acceleration due to gravity
        self.c = (self.g/2)**0.5  # Ship speed
        self.k0 = self.g/self.c**2
        self.U = 2  # Water speed (m/s)
        self.B = 10  # Tank width (m)
        self.H = 20  # Tank depth
        self.m = 0  # Wave harmonic
        self.M = 2  # Max wave harmonic
        self.rho = 1000  # Water density
        self.form_factor = 1  # Form factor k
        self.mu = 1.139e-06  # viscosity


class create_hull:

    def __init__(self):
        self.file_loc = "5s.stl"
        self.mesh = stl_mesh.Mesh.from_file(self.file_loc)
        self.panel_centre = []
        self.panel_area = []

        self.load_hull()

    def load_hull(self):
        """
        Load the hull
        """
        panel_x = (self.mesh.v0[:, 0]+self.mesh.v1[:, 0]+self.mesh.v2[:, 0])/3
        panel_y = (self.mesh.v0[:, 1]+self.mesh.v1[:, 1]+self.mesh.v2[:, 1])/3
        panel_z = (self.mesh.v0[:, 2]+self.mesh.v1[:, 2]+self.mesh.v2[:, 2])/3
        self.panel_centre = np.array([panel_x, panel_y, panel_z]).transpose()

        # ====================================================================
        #       Calculate the area of each panel
        # ====================================================================
        AB = self.mesh.v1-self.mesh.v0
        AC = self.mesh.v2-self.mesh.v0
        self.panel_area = np.sum(0.5*abs(np.cross(AB, AC)), axis=1)

#    def panalize(self):
#        """
#        ''Panelize'' the hull. For each STL tringle, calculate the centre of
#        each panel and the area
#        """


class create_sources:

    def __init__(self, body, tank):
        self.body = body
        self.tank = tank
        self.strength = np.zeros(len(body.panel_centre))
        self.coords = body.panel_centre
        self.initialise_sources()

    def initialise_sources(self):
        self.strength = self.source_strength(self.body.mesh.normals,
                                             [self.tank.U, 0, 0],
                                             self.body.panel_area)

        # ====================================================================
        #       Remove sources above waterline
        # ====================================================================
        for i in range(len(self.strength)):
            if self.body.panel_centre[i][2] > 0:
                self.strength[i] = 0

        # ====================================================================
        #       Remove sources with negative y (body assumed symetric,
        #       accounted for in the math)
        # ====================================================================
        for i in range(len(self.strength)):
            if self.body.panel_centre[i][1] < 0:
                    self.strength[i] = 0

#        print(self.strength)

    def source_strength(self, n, U, A):
        """
        Returns the strength of a source, given the normal vector, the onset
        free stream vector and the panel area
        """
        return (-1/(2 * np.pi))*np.dot(n, U) * A


class wave_resistance:

    def __init__(self, sources, tank):
        self.sources = sources
        self.tank = tank
        self.elevation = []
        self.RWm = np.zeros(tank.M)

        self.calc_Rwm()

    def wave_components(self, m):
        """
        Takes wave harmonic n and reutrns the wave number km and angle theta θm
        using a newton raphson root finding method
        """
        X = m * np.pi/tank.B
        k = tank.k0 * (1 + (1 + (2 * X/tank.k0))**0.5) / 2  # Initial guess
        yn = optimize.newton(self.f, k, args=(m, tank))
        thetan = np.arcsin(((2*np.pi*m)/tank.B)/yn)

        return [yn, thetan]

    def f(self, x, n, tank):
        return x**2-tank.k0*x*np.tanh(x*tank.H)-1*((2*n*np.pi)/(tank.B))**2

    def elevation_terms(self, m):
        coeff = (16*np.pi*self.tank.U)/(self.tank.B*self.tank.g)

        kbar = self.tank.g/(self.tank.U**2)
        km, thetam = self.wave_components(m)

        # ====================================================================
        #       First Fraction
        # ====================================================================
        top_frac = (kbar+km*(np.cos(thetam)**2))
        bottom_frac = (1 + (np.sin(thetam)**2) - kbar*self.tank.H *
                       (1/(np.cosh(km*self.tank.H)**2)))
        first_frac = top_frac/bottom_frac

        # ====================================================================
        #        Summation Term
        # ====================================================================
        summation = 0

        for i in range(len(sources.strength)):
            strength_i = sources.strength[i]
            exp_term = np.exp(-1*km * self.tank.H)
            cosh_term = np.cosh(km*(self.tank.H * sources.coords[i][2]))
            matrix_term = np.array([[np.cos(km * sources.coords[i][0] *
                                            np.cos(thetam))],
                                    [np.sin(km*sources.coords[i][0] *
                                            np.cos(thetam))]])
            if m % 2 == 0:  # Even m
                multiplier = np.cos((m*np.pi*sources.coords[i][1])/self.tank.B)
            else:  # Odd m
                multiplier = np.sin((m*np.pi*sources.coords[i][1])/self.tank.B)

            summation += strength_i*exp_term*cosh_term*matrix_term*multiplier

        if m == 0:
            multiplier = multiplier*0.5

        return coeff*first_frac*summation

    def calc_Rwm(self):
        print("hi")
        coeff = 0.25*self.tank.rho*self.tank.g*self.tank.B

        k0, theta0 = self.wave_components(0)
        eta0, nu0 = self.elevation_terms(0)

        zeta0_squared = eta0**2 + nu0**2

# =============================================================================
#       First/Zeroth Wave Component
# =============================================================================

        if (2*k0*self.tank.H) > 50:
            frac_term = 0
        else:
            top_frac = 2*k0*self.tank.H
            bottom_frac = np.sinh(2*k0*self.tank.H)
            frac_term = top_frac/bottom_frac

#        print(coeff, zeta0_squared, frac_term)
        self.RWm[0] = coeff*zeta0_squared*(1-frac_term)

# =============================================================================
#       Other wave components
# =============================================================================

        for i in np.arange(1, self.tank.M):
            km, thetam = self.wave_components(i)
            etam, num = self.elevation_terms(i)
            zetam_squared = etam**2 + num**2
            frac1 = (np.cos(thetam)**2)/2

            if (2*km*self.tank.H) > 10:
                frac2 = 0
            else:
                top_frac = (2*km*self.tank.H)
                bottom_frac = (np.sinh(2*km*self.tank.H))
                frac2 = top_frac/bottom_frac

            self.RWm[i] = coeff*zetam_squared*(1-(frac1*(1+frac2)))


tank = tank_properties()
hull = create_hull()
# hull.mesh.translate([0,0.5,0.5])
# hull.load_hull()
sources = create_sources(hull, tank)
Rw = wave_resistance(sources, tank)
print(Rw.RWm)
# sources.initialise_sources()

# print(hull.panel_centre[-1])
# print(wave_components(10, tank))
# print(source_strength([[1,0,0]],[tank_properties.U,0,0],1))
