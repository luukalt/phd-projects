# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:01:46 2022

@author: laaltenburg

y_plus(U_bulk) calculator for pipe
"""

#%% Import packages
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from premixed_flame_properties import *

#%% Start
plt.close("all")

T_u = 273.15 + 20
p_u = 101325 # 1 atm

phi = 0.4
H2_percentage = 100

mixture = PremixedFlame(phi, H2_percentage, T_u, p_u)
nu_u = mixture.nu_u

D = 25.16
D /= 1e3
U_bulk = 9

u_tau = (0.03955*U_bulk**(7/4)*nu_u**(1/4)*D**(-1/4))**0.5


y = 0.5
y /= 1e3
y_plus = u_tau*y/nu_u 

print("y_plus: {0:.1f}".format(y_plus))