import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import ScalarFormatter, FuncFormatter, FormatStrFormatter
import matplotlib.ticker

from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import flame, piv_method, interpolation_method
from plot_params import colormap, fontsize

def copy_content(fig, old_ax, new_ax):
    # Copy lines
    for line in old_ax.lines:
        x, y = line.get_data()
        new_ax.plot(x, y, 
                    color=line.get_color(), 
                    linestyle=line.get_linestyle(), 
                    marker=line.get_marker(),
                    markersize=line.get_markersize(),
                    markerfacecolor=line.get_markerfacecolor(),
                    markeredgecolor=line.get_markeredgecolor(),
                    markeredgewidth=line.get_markeredgewidth(),
                    label=line.get_label())
        
    # Copy pcolor plots
    pcollections = [col for col in old_ax.collections if isinstance(col, PolyCollection) and not isinstance(col, Quiver)]
    for pc in pcollections:
        # Get the data and other properties from the original pcolor plot
        # cmap = pc.get_cmap()
        array = pc.get_array().data
        boundaries = pc.get_clim()
        
        # Check if old_ax has stored X and Y data (we need to ensure this)
        if hasattr(old_ax, '_X_data') and hasattr(old_ax, '_Y_data'):
            X = old_ax._X_data
            Y = old_ax._Y_data
            Z = old_ax._Z_data
            cmap = old_ax._cmap
            clim = old_ax._clim
            
            mesh = new_ax.pcolor(X, Y, Z, cmap=cmap)
            
            print(clim[1])
            mesh.set_clim(clim[0], clim[1])
            
            n_axes = len(old_ax.figure.axes)
            
            if n_axes > 1:
                
                for ax in old_ax.figure.axes:
                    
                    if ax.get_label() == '<colorbar>':
                        cbar_title = ax.get_ylabel()

                        cbar = new_ax.figure.colorbar(mesh, format='%.1e')
                        
                        # Define your custom colorbar tick locations and labels
                        num_ticks = 6
                        cbar_max = clim[1]
                        custom_cbar_ticks = np.linspace(0, cbar_max, num_ticks) # Replace with your desired tick positions
                        
                        print(custom_cbar_ticks)
                        # Create a ScalarFormatter with scientific notation and one decimal place
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((0, 0))
                        formatter.min_n_ticks = 5

                        formatter.set_useOffset(False)  # Prevent adding offset

                        # # Set the formatter for colorbar ticks
                        cbar.formatter = formatter
                        # Set the number of desired ticks
                        cbar.locator = matplotlib.ticker.MaxNLocator(num_ticks)

                        cbar.update_ticks()
                        
                        # if cbar_max < 1:
                        #     custom_cbar_tick_labels = [f'{tick:.3f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
                        # else:
                        #     custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
                        
                        # Set the colorbar ticks and labels
                        # cbar.set_ticks(custom_cbar_ticks)
                        # cbar.set_ticklabels(custom_cbar_tick_labels)
                        cbar.set_label(cbar_title, rotation=0, labelpad=25, fontsize=28) 
                        cbar.ax.tick_params(labelsize=fontsize)
                        # cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

    
    # Copy quiver plots (vectors)
    quivers = [col for col in old_ax.collections if isinstance(col, Quiver)] 
    for q in quivers:
        X, Y, U, V = q.X, q.Y, q.U, q.V
        new_ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=q.scale)
    
    
    # Set same limits and labels
    new_ax.set_xlim(old_ax.get_xlim())
    new_ax.set_ylim(old_ax.get_ylim())
    new_ax.set_title(old_ax.get_title())
    new_ax.set_xlabel(old_ax.get_xlabel())
    new_ax.set_ylabel(old_ax.get_ylabel())


# Load pickled figures
filename1 = 'H100_Re16000_fig8'
# filename1 = 'H0_Re4000_fig8'
# filename1 = 'H100_Re16000_fig4'
# filename1 = 'H0_Re4000_fig2_favre'

file_path1 = os.path.join('pickles', 'ls', f'{filename1}.pkl')

filename2 = 'H100_Re12500_fig8'
# filename2 = 'H0_Re3000_fig8'
# filename2 = 'H100_Re12500_fig4'
# filename2 = 'H0_Re4000_fig21'
# filename2 = 'H0_Re4000_fig4_favre'

file_path2 = os.path.join('pickles', 'ls', f"{filename2}.pkl")

with open(file_path1, 'rb') as f:
    fig1 = pickle.load(f)

with open(file_path2, 'rb') as f:
    fig2 = pickle.load(f)

# Create a new figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(9, 6)) # fig 4 and 8
# fig, axs = plt.subplots(1, 2, figsize=(7, 6)) # fig 13 and 21

fontsize = fontsize

# Copy contents
copy_content(fig, fig1.axes[0], axs[0])
copy_content(fig, fig2.axes[0], axs[1])

axs[0].set_aspect('equal')
axs[1].set_aspect('equal')

fig.subplots_adjust(wspace=-.8)

# fig.subplots_adjust(right=3)
#
# Remove ylabel
axs[1].set_ylabel('')

# Turn off y-axis labels for the second subplot (ax), but keep the ticks
axs[1].tick_params(axis='y', labelleft=False)

# Get the current x-axis label text
current_xlabel = axs[0].get_xlabel()
current_ylabel = axs[0].get_ylabel()

# Set the x-axis label with the same text and a different fontsize
axs[0].set_xlabel(current_xlabel, fontsize=fontsize)  # Adjust the fontsize as needed
axs[1].set_xlabel(current_xlabel, fontsize=fontsize)  # Adjust the fontsize as needed
axs[0].set_ylabel(current_ylabel, fontsize=fontsize)  # Adjust the fontsize as needed

axs[0].tick_params(axis='both', labelsize=fontsize)
axs[1].tick_params(axis='both', labelsize=fontsize)

# Add textbox
left, width = .075 , .0
bottom, height = .225, .72
right = left + width
top = bottom + height

# Re_D_check = 4000
Re_D_check = 16000

if Re_D_check in [3000, 4000]:
    fuel_type = 'DNG'
elif Re_D_check in [12500, 16000]:
    fuel_type = 'H$_{2}$'

text = f'{fuel_type}-{Re_D_check}'
# text = 'a'

axs[0].text(left, top, text, 
        horizontalalignment='left',
        verticalalignment='center',
        transform=axs[0].transAxes,
        fontsize=16,
        bbox=dict(facecolor="w", edgecolor='k', boxstyle='round')
        )

# Re_D_check = 3000
Re_D_check = 12500

if Re_D_check in [3000, 4000]:
    fuel_type = 'DNG'
elif Re_D_check in [12500, 16000]:
    fuel_type = 'H$_{2}$'

text = f'{fuel_type}-{Re_D_check}'
# text = 'b'
        
axs[1].text(left, top, text, 
        horizontalalignment='left',
        verticalalignment='center',
        transform=axs[1].transAxes,
        fontsize=16,
        bbox=dict(facecolor='w', edgecolor='k', boxstyle='round')
        )

fig.tight_layout()

filename = f'{filename1}_{filename2}'
# Constructing the paths
eps_path = os.path.join('figures', f'{filename}.eps')
fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')

#%% Save images
# Get a list of all currently opened figures
# figure_ids = plt.get_fignums()
# figure_ids = [4, 13, 21]

# # Apply tight_layout to each figure
# for fid in figure_ids:
#     fig = plt.figure(fid)
#     fig.tight_layout()
#     filename = f'test'
    
#     # Constructing the paths
#     eps_path = os.path.join('figures', f"{filename}.eps")
#     # pkl_path = os.path.join('pickles', f"{filename}.pkl")
    
#     # Saving the figure in EPS format
#     fig.savefig(eps_path, format="eps", dpi=300, bbox_inches="tight")
    
#     # Pickling the figure
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(fig, f)
