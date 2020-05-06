"""
Creates a virtual tank.

Classes:
    tank_properties

Methods:

Imports:

"""


class tank_properties:
    """
    tank_properties class

    Attributes:
        g -- Acceleration due to gravity
        c -- Ship speed
        K0 -- Fundimental wave number (g/c^2)
        U -- Ship speed
        B -- Tank width
        H -- Tank height
        M -- Maximum number of wave harmonics
        rho -- Water density
        mu -- Water viscosity
    """

    def __init__(self):
        self.g = 9.81  # Acceleration due to gravity
        self.c = (self.g/2)**0.5  # Ship speed
        self.k0 = self.g/self.c**2
        self.U = 2  # Water speed (m/s)
        self.B = 10  # Tank width (m)
        self.H = 20  # Tank depth
#        self.m = 0  # Wave harmonic
        self.M = 2  # Max wave harmonic
        self.rho = 1000  # Water density
#        self.form_factor = 1  # Form factor k
        self.mu = 1.139e-06  # viscosity
