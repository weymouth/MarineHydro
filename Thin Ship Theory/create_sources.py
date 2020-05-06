"""
Calculates the sources .

Classes:
    create_sources()

Methods:

Imports:
    numpy
"""
import numpy as np


class create_sources:

    def __init__(self, body, tank):
        """
        Initialise sources

        Inputs:
            body -- A create_hull object
            tank -- a tank object
        """
        self.body = body
        self.tank = tank
        self.strength = np.zeros(len(body.panel_centre))
        self.coords = body.panel_centre
        self.calc_sources()

    def calc_sources(self):
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
