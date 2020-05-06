"""
Creates a virtual hull.

Classes:
    create_hull()

Methods:

Imports:
    numpy, mesh from stl
"""

import numpy as np
from stl import mesh as stl_mesh


class create_hull:
    """
    Creates a hull object.

    Attributes:
        file_loc        -- File location of the hull
        mesh            -- stl object created from the file location
        panel_centre    -- An array of the coordinates of the centre of panels
        panel_area      -- An array of the areas of panels

    Methods:
        load_hull       -- Reload the hull after a change in the mesh. (e.g. a translation)
    """

    def __init__(self, file):
        """
        Initialize a hull

        Inputs:
            file -- The location of an stl file

        Outputs:
            A hull object
        """

        self.file_loc = file
        self.mesh = stl_mesh.Mesh.from_file(self.file_loc)
        self.panel_centre = []
        self.panel_area = []

        self.load_hull()

    def load_hull(self):
        """
        Reload the hull after a transformation of the mesh
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
