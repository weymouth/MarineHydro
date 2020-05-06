"""
Classes:
    wave_resistance()

Methods:

Imports:
    numpy, optimize from scipy
"""
import numpy as np
from scipy import optimize


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
        X = m * np.pi/self.tank.B
        k = self.tank.k0 * (1 + (1 + (2 * X/self.tank.k0))**0.5) / 2  # Initial guess
        yn = optimize.newton(self.f, k, args=(m, self.tank))
        thetan = np.arcsin(((2*np.pi*m)/self.tank.B)/yn)

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
        
        if km*self.tank.H > 20:
            sechKH = 0
        else:
            sechKH = 1/np.cosh(km*self.tank.H)

        bottom_frac = (1 + (np.sin(thetam)**2) - kbar*self.tank.H*(sechKH**2))
        first_frac = top_frac/bottom_frac

        # ====================================================================
        #        Summation Term
        # ====================================================================
        summation = 0

        for i in range(len(self.sources.strength)):
            strength_i = self.sources.strength[i]
            exp_term = np.exp(-1*km * self.tank.H)
            cosh_term = np.cosh(km*(self.tank.H * self.sources.coords[i][2]))
            matrix_term = np.array([[np.cos(km * self.sources.coords[i][0] *
                                            np.cos(thetam))],
                                    [np.sin(km*self.sources.coords[i][0] *
                                            np.cos(thetam))]])
            if m % 2 == 0:  # Even m
                multiplier = np.cos((m*np.pi*self.sources.coords[i][1])/self.tank.B)
            else:  # Odd m
                multiplier = np.sin((m*np.pi*self.sources.coords[i][1])/self.tank.B)

            summation += strength_i*exp_term*cosh_term*matrix_term*multiplier

        if m == 0:
            multiplier = multiplier*0.5

        return coeff*first_frac*summation

    def calc_Rwm(self):
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
