import os
import numpy as np
import matplotlib.pyplot as plt

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings

# Parameters
angle = np.pi / 16  # 45 degrees
# angle = 0

# Generate original sine wave data
A = 3
y = np.linspace(-10 * np.pi, 10 * np.pi, 1000)
x = A*np.sin(y)

# Rotate the sine wave
x_rot = x * np.cos(angle) - y * np.sin(angle)
y_rot = x * np.sin(angle) + y * np.cos(angle)



length = 3

# Create a figure and axis
fig, ax = plt.subplots()

# Fill left side with blue
ax.fill_betweenx(y_rot, x1=-10, x2=x_rot, where=(x_rot <= np.inf), color='tab:blue', alpha=1)

# Fill right side with orange
ax.fill_betweenx(y_rot, x1=x_rot, x2=10, where=(x_rot >= -np.inf), color='tab:orange', alpha=1)

# Plot the rotated sine wave
# ax.plot(x_rot, y_rot, color='k')

# Add a vector arrow
start_x = -6.25
start_y = -6
arrow_dx = 0
arrow_dy = 2
ax.arrow(start_x, start_y, arrow_dx, arrow_dy, width=.25, head_width=1, head_length=1, fc='k', ec='k')


# Add vertical lines

line_ycenters = [-2.5, 6.]  # lengths of the vertical lines
for line_ycenter in line_ycenters:
    
    x_value = A*np.sin(line_ycenter)
    x_pos = x_value * np.cos(angle) - line_ycenter * np.sin(angle)
    y_pos = x_value * np.sin(angle) + line_ycenter * np.cos(angle)
    ax.vlines(x=x_pos, ymin=y_pos - length, ymax=y_pos, colors='k', linestyles='dashed' if line_ycenter == 6. else 'solid')
    ax.vlines(x=x_pos, ymin=y_pos, ymax=y_pos + length, colors='k', linestyles='solid' if line_ycenter == 6. else 'dashed')
    
    dxdy_value = A*np.cos(line_ycenter)

    y_value = line_ycenter
    x_value = A*np.sin(y_value)
    dydx_value = 1/dxdy_value

    x1 = np.linspace(x_value - length, x_value, 1000)
    x2 = np.linspace(x_value, x_value + length, 1000)
    
    y1 = dydx_value * (x1 - x_value) + y_value
    y2 = dydx_value * (x2 - x_value) + y_value
    
    # Rotate the sine wave
    x1_rot = x1 * np.cos(angle) - y1 * np.sin(angle)
    x2_rot = x2 * np.cos(angle) - y2 * np.sin(angle)
    
    y1_rot = x1 * np.sin(angle) + y1 * np.cos(angle)
    y2_rot = x2 * np.sin(angle) + y2 * np.cos(angle)
    
    
    ax.plot(x1_rot, y1_rot, color='k', ls='dashed') # if line_ycenter == 6. else 'solid')
    ax.plot(x2_rot, y2_rot, color='k', ls='solid') # if line_ycenter == 6. else 'dashed')
    

    x_value = A*np.sin(y_value)
    ax.plot(x_value * np.cos(angle) - y_value * np.sin(angle), x_value * np.sin(angle) + y_value * np.cos(angle), 'ko')


ax.text(-.85, 7.2, r'$\theta < 0$', fontsize=12,  ha='center', color='k')

ax.text(0, -4.1, r'$\theta > 0$', fontsize=12, ha='center')

ax.text(-6., start_y - 1.25, 'Direction of \n the bulk flow', fontsize=6, ha='center')

ax.text(-5.5, 1., r'Unburnt mixture', fontsize=6, ha='center')

ax.text(2.5, 5., 'Burnt mixture', fontsize=6, ha='center')



# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Diagonal Sine Wave with Blue and Orange Background')

# Set the limits
ax.set_xlim([-8, 5])
ax.set_ylim([-8, 10])
ax.set_aspect('equal')


# Hide axis and labels
ax.axis('off')

filename = 'theta_sketch'
eps_path = os.path.join('figures', f"{filename}.eps")
 
# Saving the figure in EPS format
fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')























