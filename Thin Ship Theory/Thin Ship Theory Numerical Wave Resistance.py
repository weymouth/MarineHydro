# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:26:52 2020

@author: mclea
"""

from tank_properties import tank_properties
from create_hull import create_hull
from create_sources import create_sources
from wave_resistance import wave_resistance

tank = tank_properties()
tank.M = 10
hull = create_hull('5s.stl')
sources = create_sources(hull, tank)
Rw = wave_resistance(sources, tank)

for i, value in enumerate(Rw.RWm):
    print("Rw for harmonic " + str(i) + " is " + str(value) + " N")
