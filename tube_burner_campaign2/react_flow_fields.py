# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:25:15 2023

@author: laaltenburg

"""

#%% IMPORT PACKAGES
import os
import sys

# Add the 'main' folder to sys.path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
flame_front_detection_directory = os.path.abspath(os.path.join(parent_folder, 'flame_front_detection'))
flame_simulations_directory = os.path.abspath(os.path.join(parent_folder, 'flame_simulations'))
plot_parameters_directory = os.path.abspath(os.path.join(parent_folder, 'plot_parameters'))

# Add the flame_object_directory to sys.path
sys.path.append(parent_folder)
sys.path.append(flame_front_detection_directory)
sys.path.append(flame_simulations_directory)
sys.path.append(plot_parameters_directory)

import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from custom_colormaps import parula
from premixed_flame_properties import PremixedFlame
from cone_angle import cone_angle
from intersect import intersection

from ns_terms import ns_incomp_terms, ns_comp_terms
from nonreact_flow_fields import non_react_dict
from favre_averaging import process_df

import flame_object
import premixed_flame_properties
import rc_params_settings

#%% CLOSE ALL FIGURES
plt.close('all')

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 14.0}
#     )

#%% SET MATPLOTLIB PARAMETERS
# set_mpl_params()

# Set color map
# colormap = 'jet' 
# colormap = 'viridis'
colormap = parula

#%% FUNCTIONS

class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines

def plot_pressure_along_streamline(ax1, ax2, dpdr, dpdx, r_norm_values, x_norm_values, lines, incomp_indices, colors, p):
    
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Pressure along stream line')
    ax1.grid(True)
    
    styles_react = ['solid', 'solid', 'solid']
    styles_nonreact = ['dashed', 'dashed', 'dashed']
    
    for line, incomp_index, color in zip(lines, incomp_indices, colors):
        
        line = line[incomp_index]
        line_x, line_y = line[:,0], line[:,1]
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
          
        dpdr_values = dpdr.values.flatten()
        dpdx_values = dpdx.values.flatten()
        
        dpdr_along_line = griddata((r_norm_values, x_norm_values), dpdr_values, line, method='linear')
        dpdx_along_line = griddata((r_norm_values, x_norm_values), dpdx_values, line, method='linear')
        
        p_along_line = np.zeros(dpdr_along_line.shape[0])
    
        for i in range(1, dpdr_along_line.shape[0]):
            
            dr = line_x[i] - line_x[i-1]
            dx = line_y[i] - line_y[i-1]
            
            # dr *= D_in*1e-3
            # dx *= D_in*1e-3
            
            p_along_line[i] = p_along_line[i-1] + 0.5 * (dpdr_along_line[i] + dpdr_along_line[i-1]) * dr + 0.5 * (dpdx_along_line[i] + dpdx_along_line[i-1]) * dx
            
        
        dpds = np.gradient(p_along_line, cumulative_distances)
        
        if p == 0:
            ls = 'solid'
        elif p == 1:
            ls = 'dashed'
            
        ax1.plot(cumulative_distances, p_along_line, c=color, marker='None', ls=ls)
        
        ax2.plot(cumulative_distances, dpds, c=color, marker='None', ls=ls)
    
    # make proxy artists
    # make list of one line -- doesn't matter what the coordinates are
    dummy_line = [[(0, 0)]]
    # set up the proxy artist
    lc_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_react, colors=colors)
    lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
    
    fontsize_label = 20
    
    x_label = r'$s/D$'
    # y_label = r'$\frac{\overline{p}}{\rho_{u}U_{b}^2}$'
    # ax1.set_ylabel(y_label, rotation=0, labelpad=24, fontsize=26)
    
    y_label = r'$\overline{p^*} - \overline{p_{0}^*}$'
    ax1.set_ylabel(y_label, fontsize=fontsize_label)
    
    ax1.set_xlabel(x_label, fontsize=fontsize_label)
    # ax1.set_ylabel(y_label)
    ax1.set_xlim([0, 1])  # replace with your desired x limits
    ax1.set_ylim([-.01, .075])  # replace with your desired x limits
    ax1.grid(True)
    # ax1.legend()
    
    custom_y_ticks = [.0, .02, .04, .06]
    # custom_y_tick_labels = ['$}$', '$\overline{p_{0}^*}$+.02', '$\overline{p_{0}^*}$+.04', '$\overline{p_{0}^*}$+.06']
    ax1.set_yticks(custom_y_ticks)
    # ax1.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
    
    ax1.tick_params(axis='both', labelsize=fontsize_label)
    
    x_label = r'$s/D$'
    y_label = r'$\frac{dp^{*}}{ds^{*}}$'
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label, rotation=0, labelpad=15, fontsize=14)
    ax2.grid(True)
    # ax2.legend()
    
    # create the legend
    
    if flame.Re_D == 4000:
        ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
                    handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, .65))
    elif flame.Re_D == 16000:
        ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
                        handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, 1.))
        
    ax2.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
               handlelength=3, handleheight=3)
    
def plot_pressure_along_arbitrary_line(p_field, r_norm_values, x_norm_values, line):
    
    fig1, ax1 = plt.subplots()
    ax1.set_title('Pressure along arbitrary line')
    ax1.grid(True)
    
    # Compute the distances between consecutive points on the line
    distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
    
    # Compute the cumulative distances along the line
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    
    p_field_values = p_field.flatten()   
    term_along_line = griddata((r_norm_values, x_norm_values), p_field_values, line, method='linear')
    
    ax1.scatter(cumulative_distances, term_along_line, marker='o', label=f'pressure')

    ax1.legend()
    
def plot_mass_cons(p, ax, ax_inset, mass_cons, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Conservation of mass')
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\rho^*\mathbf{u^{*}}}$'
    
    with open('exported_data.pkl', 'rb') as file:
        imported_data = pickle.load(file)
    
    # First mass conservation
    for i, (line, flame_front_index, color) in enumerate(zip(lines, flame_front_indices, colors)):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method='linear')
                
                if p == 0:
                    ax.plot(cumulative_distances[:], term_along_line[:], c=color, ls='dashed', marker='None')
                    ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms3, mec='k')
                    ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                    ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                    
                    ax_inset.plot(cumulative_distances[:], term_along_line[:], c=color, ls='dashed', marker='None')
                    ax_inset.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms3, mec='k')
                    ax_inset.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                    ax_inset.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                
                if p == 1:
                    ax.plot(cumulative_distances[:], term_along_line[:], c=color, ls='dotted', marker='None')
                    ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                    ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                    
                    ax_inset.plot(cumulative_distances[:], term_along_line[:], c=color, ls='dotted', marker='None')
                    ax_inset.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                    ax_inset.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                
                # Include Favre averaging
                cumulative_distances = imported_data['cumulative_distances'][i]
                term_along_line = imported_data['term_along_line'][i]
                flame_front_index = imported_data['flame_front_index'][i]
                
                ax.plot(cumulative_distances[:], term_along_line[:], c=color, ls='solid', marker='None')
                ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms3, mec='k')
                ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                
                ax_inset.plot(cumulative_distances[:], term_along_line[:], c=color, ls='solid', marker='None')
                ax_inset.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms3, mec='k')
                ax_inset.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms6)
                ax_inset.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms6, mec='k')
                
    
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20) 
    
    # Your custom markers
    # marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, mec='k', mfc='None', linestyle='')
    # marker_dummy2 = Line2D([0], [0], label=r'Constant density limit [$\mathbf{\nabla^*} \cdot \overline{\mathbf{u^{*}}} = 0.2]$', marker='s', markersize=ms5, mec='k', mfc='None', linestyle='')
    
    # Automatic legends (if any)
    # handles, labels = ax.get_legend_handles_labels()
    
    # marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, 
    #      mec='k', mfc='None', linestyle='')
    
    # marker_dummy2 = Line2D([0], [0], label=r'Incompressibility limit [$\mathbf{\nabla^*} \cdot \overline{\mathbf{u^{*}}} = 0.2]$', marker='s', markersize=ms1, 
    #      mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    # handles, labels = ax.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    # handles.extend([marker_dummy1])
    
    # ax.legend(handles=handles,  loc='upper right')
    
    # # Your custom line styles
    # styles_re_react = ['dashed', 'dashed', 'dashed']
    # styles_favre_react = ['solid', 'solid', 'solid']
    # styles_nonreact = ['dotted', 'dotted', 'dotted']
    # dummy_line = [[(0, 0)]]
    # lc_re_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_re_react, colors=colors)
    # lc_favre_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_favre_react, colors=colors)
    # lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
    
    # # Combine all the legend entries
    # all_handles = handles + [lc_re_react, lc_favre_react, lc_nonreact]
    # all_labels = labels + ['Average flame front location', 'reacting flow [Re-average]', 'reacting flow [Favre-average]', 'non-reacting flow']
    
    # # Create a custom handler map for special legends (if required)
    # handler_map = {type(lc_re_react): HandlerDashedLines()}  # Add other handlers if required
    
    # # Display the legend
    # ax.legend(handles=all_handles, labels=all_labels, handler_map=handler_map, handlelength=3, handleheight=3, title='', loc="upper left", ncol=1, prop={"size": 12})
       
def plot_ns_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    width, height = 6, 6
    fig1, ax1 = plt.subplots(figsize=(width, height))
    
    # ax1.set_title('Conservation of mass')
    ax1.grid(True)
    ax1.set_ylim([-.25, 8.])  # replace with your desired x limits
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\mathbf{u^{*}}}$'
    
    incomp_indices = []
    
    incomp_threshold = 0.2
    
    # First mass conservation
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method='linear')
                
                if j == 0 and k == 0:
                    
                    # Divergence of u is almost zero
                    valid_indices = np.where(np.abs(term_along_line) <= incomp_threshold)[0]

                    breaks_in_valid_indices = np.where(np.diff(valid_indices) != 1)[0]
                    
                    if breaks_in_valid_indices.size:
                        valid_indices = valid_indices[:breaks_in_valid_indices[0]+1]
                    
                    incomp_indices.append(valid_indices) 
                
                ax1.plot(cumulative_distances[:], term_along_line[:], c=color, marker='None')
                ax1.plot(cumulative_distances[valid_indices[-1]], term_along_line[valid_indices[-1]], c=color, lw=1, marker='s', mec='k', ms=ms6)
                
                ax1.plot(cumulative_distances[0], term_along_line[0], c=color, marker='o', ls='None', mec='k', mfc='None', ms=ms6)
                ax1.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ls='None', mec='k', mfc=color, ms=ms6)
                ax1.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms5, mec='k')
                
                ax1.set_xlabel(x_label)
                ax1.set_ylabel(y_label)
    
    # styles_react = ['None', 'None']
    # # make list of one line -- doesn't matter what the coordinates are
    # dummy_line = [[(0, 0)]]
    
    marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms5, 
         mec='k', mfc='None', linestyle='')
    
    marker_dummy2 = Line2D([0], [0], label=r'Incompressibility limit [$\mathbf{\nabla^*} \cdot \overline{\mathbf{u^{*}}} = 0.2]$', marker='s', markersize=ms5, 
         mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    handles, labels = ax1.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    handles.extend([marker_dummy1, marker_dummy2])
    
    ax1.legend(handles=handles)

    # Second momentum equations
    mom_x_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    '[4] Viscous diffusion',
                    '[5] Reynolds normal stress [axial]  ',
                    '[6] Reynolds shear stress [radial] '
                    ]
    
    mom_r_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    '[4] Viscous diffusion',
                    '[5] Reynolds shear stress [axial]   ',
                    '[6] Reynolds normal stress [radial] '
                    ]
    
    mom_markers = ['v', '^', 'o', 's', 'p', 'd']
    
    linestyles = ['-', '-', '--', '-', '-', '-']
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fontsize_label = 24
    
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        width, height = 9, 9
        fig2, ax2 = plt.subplots(figsize=(width, height))
        # fig2, ax2 = plt.subplots()
        
        # ax2.set_title('Conservation of axial momentum for streamline ' + str(round(line[0][0], 1)))
        ax2.grid(True)
        ax2.set_xlabel(x_label, fontsize=fontsize_label)
        ax2.set_ylabel(r'Non-dimensionalized terms', fontsize=fontsize_label)
        
        fig3, ax3 = plt.subplots(figsize=(width, height))
        # ax3.set_title('Conservation of radial momentum r=' + str(round(line[0][0], 1)))
        ax3.grid(True)
        ax3.set_xlabel(x_label, fontsize=fontsize_label)
        ax3.set_ylabel(r'Non-dimensionalized terms', fontsize=fontsize_label)
        
        ax2.set_xlim([-.025, .5])  # replace with your desired x limits
        ax3.set_xlim([-.025, .5])  # replace with your desired x limits
        
        ax2.set_ylim([-.16, .16])  # replace with your desired y limits
        ax3.set_ylim([-.16, .16])  # replace with your desired y limits
        
        for j, terms in enumerate([mass_cons, mom_x, mom_r]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method='linear')
                
                if j == 0 and k == 0:
                    
                    # Divergence of u is almost zero
                    valid_indices = np.where(np.abs(term_along_line) <= incomp_threshold)[0]

                    breaks_in_valid_indices = np.where(np.diff(valid_indices) != 1)[0]
                    
                    if breaks_in_valid_indices.size:
                        valid_indices = valid_indices[:breaks_in_valid_indices[0]+1]
                
                color, linestyle, mom_marker = map(lambda x: x[k], [colors, linestyles, mom_markers])

                if j == 1:
                    
                    ax2.plot(cumulative_distances[valid_indices], term_along_line[valid_indices], c=colors[k], ls=linestyles[k], marker=mom_markers[k], ms=ms4, label=mom_x_labels[k])
                
                if j == 2:
    
                    ax3.plot(cumulative_distances[valid_indices], term_along_line[valid_indices], c=colors[k], ls=linestyles[k], marker=mom_markers[k], ms=ms4, label=mom_r_labels[k])
        
        
        # ax2.legend(title='Axial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        # ax3.legend(title='Radial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.225, .45))
        # ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.0, .425))
        
        ax3.legend(loc='best', prop={'size': 18})
        
        ax2.tick_params(axis='both', labelsize=20)
        ax3.tick_params(axis='both', labelsize=20)
        
        # ax2.legend()
        
        # ax2.legend(title='Terms', ncol=1, prop={"size": 12})
        # ax3.legend(title='Terms', ncol=1, prop={"size": 12})
    
    return incomp_indices

def plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform):
    
    strms = []
    streamlines = []
    flame_front_indices = []
    paths = []
    
    fig1, ax1 = plt.subplots()
    
    # Cone angle
    ax1.plot(r_range_left, poly_left_fit, c='k', ls='dashed')
    ax1.plot(r_range_right, poly_right_fit, c='k', ls='dashed')
    
    colors = []
    
    for (r_start, x_start) in start_points:
        
        start_point = np.array([[r_start, x_start]])
        strm = ax1.streamplot(r_uniform, x_uniform, u_r_uniform, u_x_uniform, linewidth=1,
                              start_points=start_point, 
                              # broken_streamlines=False, 
                              density=1,
                              )
        
        skip = 4
        ax1.quiver(r_uniform[::skip], x_uniform[::skip], u_r_uniform[::skip], u_x_uniform[::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
        
        colors.append(strm.lines.get_color())
        strms.append(strm)
       
    for c, strm in enumerate(strms):
        
        segments = strm.lines.get_segments()
        line = np.array([segment[0] for segment in segments[:-1]] + [segments[-1][1]])
        
        line_x, line_y = line[:,0], line[:,1]
        distance = np.cumsum(np.sqrt( np.ediff1d(line_x, to_begin=0)**2 + np.ediff1d(line_y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        
        fx, fy = interp1d(distance, line_x), interp1d(distance, line_y)
        
        n_interp_points = np.linspace(0, 1, 501)
        x_interp, y_interp = fx(n_interp_points), fy(n_interp_points)
        
        line = np.column_stack((x_interp, y_interp))
        
        streamline_x_max = .5
        streamline_y_max = 2.2
        streamline = line[(np.abs(line[:,0]) < streamline_x_max) & (line[:,1] > 0.1) & (line[:,1] < streamline_y_max)]
        streamline_x, streamline_y = streamline[:,0], streamline[:,1]
        streamline = np.column_stack((streamline_x, streamline_y))
        
        ax1.plot(streamline_x, streamline_y, color=colors[c], marker='None', ls='-', lw=2)
        ax1.plot(streamline_x[0], streamline_y[0], color=colors[c], marker='^', ls='None', mec='k', ms=ms5)
        ax1.plot(streamline_x[-1], streamline_y[-1], color=colors[c], marker='o', ls='None', mec='k', ms=ms5)
        
        # ax1.set_title(name)
        fontsize = 20
        ax1.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax1.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        ax1.tick_params(axis='both', labelsize=fontsize)
        
        # ax1.set_ylabel(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        ax1.grid(False)
        ax1.set_xlim([-.55, .55])  # replace with your desired x limits
        ax1.set_ylim([0., 2.25])  # replace with your desired x limits
        # ax1.legend()
        ax1.set_aspect('equal')
        
        # Find intersecion between streamline and average cone angle
        
        if streamline_x[0] < 0:
            x_intersect, y_intersect = intersection(r_range_left, poly_left_fit, streamline_x, streamline_y)

        else:
            x_intersect, y_intersect = intersection(r_range_right, poly_right_fit, streamline_x, streamline_y)
        
        intersect_point = np.array([x_intersect[0], y_intersect[0]])
        
        # Mask out y-values in y2_interp that are greater than y_intersection
        valid_indices = np.where(streamline_y <= y_intersect)[0]
        
        streamline_point_difference = streamline[valid_indices] - intersect_point
        distances_to_front = np.linalg.norm(streamline_point_difference, axis=1)
        
        # Get the index of the minimum distance
        min_distance_idx = valid_indices[np.argmin(distances_to_front)]
        
        # Retrieve the closest coordinates
        x_closest, y_closest = streamline[min_distance_idx]
        # y_closest = streamline_y[min_distance_idx]
        
        ax1.plot(x_intersect, y_intersect, marker='*', ms=ms3, c=colors[c], mec='k')
        # ax1.plot(x_closest, y_closest, 'sr')
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        flame_front_indices.append(min_distance_idx)
        streamlines.append(streamline)
        paths.append(cumulative_distances)
    
    # ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 12})
            
    return streamlines, paths, flame_front_indices, colors


def plot_streamlines_nonreacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform):
    
    strms = []
    streamlines = []
    paths = []
    
    fig1, ax1 = plt.subplots()
    
    colors = []
    
    for (r_start, x_start) in start_points:
        
        start_point = np.array([[r_start, x_start]])
        strm = ax1.streamplot(r_uniform, x_uniform, u_r_uniform, u_x_uniform, linewidth=1,
                              start_points=start_point, 
                              # broken_streamlines=False, 
                              density=1,
                              )
        
        skip = 4
        ax1.quiver(r_uniform[::skip], x_uniform[::skip], u_r_uniform[::skip], u_x_uniform[::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
        
        colors.append(strm.lines.get_color())
        strms.append(strm)
       
    for c, strm in enumerate(strms):
        
        segments = strm.lines.get_segments()
        line = np.array([segment[0] for segment in segments[:-1]] + [segments[-1][1]])
        
        line_x, line_y = line[:,0], line[:,1]
        distance = np.cumsum(np.sqrt( np.ediff1d(line_x, to_begin=0)**2 + np.ediff1d(line_y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        
        fx, fy = interp1d(distance, line_x), interp1d(distance, line_y)
        
        n_interp_points = np.linspace(0, 1, 501)
        x_interp, y_interp = fx(n_interp_points), fy(n_interp_points)
        
        line = np.column_stack((x_interp, y_interp))
        
        streamline_x_max = .5
        streamline_y_max = 2.2
        streamline = line[(np.abs(line[:,0]) < streamline_x_max) & (line[:,1] >= 0.1) & (line[:,1] < streamline_y_max)]
        streamline_x, streamline_y = streamline[:,0], streamline[:,1]
        streamline = np.column_stack((streamline_x, streamline_y))
        
        ax1.plot(streamline_x, streamline_y, color=colors[c], marker='None', ls='dashed', lw=2)
        ax1.plot(streamline_x[0], streamline_y[0], color=colors[c], marker='^', ls='None', mec='k', ms=ms5)
        ax1.plot(streamline_x[-1], streamline_y[-1], color=colors[c], marker='o', ls='None', mec='k', ms=ms5)
        
        # ax1.set_title(name)
        fontsize = 20
        ax1.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax1.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        ax1.tick_params(axis='both', labelsize=fontsize)
        
        # ax1.set_ylabel(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        ax1.grid(False)
        ax1.set_xlim([-.55, .55])  # replace with your desired x limits
        ax1.set_ylim([0., 2.25])  # replace with your desired x limits
        
        ax1.set_aspect('equal')
        
        # Find intersecion between streamline and average cone angle
        # Cone angle
        # ax1.plot(r_range_left, poly_left_fit, 'k--')
        # ax1.plot(r_range_right, poly_right_fit, 'k--')
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        streamlines.append(streamline)
        paths.append(cumulative_distances)
    
    dummy_indices = [-1 for _ in range(len(streamlines))]
    
    return streamlines, paths, dummy_indices, colors
    

def plot_cartoons(flame, fig, ax, image_nr, recording, piv_method):
    
    piv_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', recording, piv_method, 'Export')
    piv_file = os.path.join(piv_dir, f'B{image_nr:04d}.csv')
    
    # Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')

    df_piv = pd.read_csv(piv_file)
    
    df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
    
    # df_piv['x_shift [mm]'] = df_piv['x [mm]'] - (D_in/2 - offset_to_wall_center)
    # df_piv['y_shift [mm]'] = df_piv['y [mm]'] + offset
    
    # df_piv['x_shift_norm'] = df_piv['x_shift [mm]']/D_in
    # df_piv['y_shift_norm'] = df_piv['y_shift [mm]']/D_in
    
    # df_piv['x_shift [m]'] = df_piv['x_shift [mm]']*1e-3
    # df_piv['y_shift [m]'] = df_piv['y_shift [mm]']*1e-3
    
    # Get the column headers
    headers = df_piv.columns
    
    bottom_limit = -100
    top_limit = 100
    left_limit = -100
    right_limit = 100
    index_name = 'y_shift_norm'
    column_name = 'x_shift_norm'
    df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
    
    pivot_u_r = pd.pivot_table(df_piv_cropped, values=headers[u_r_col_index], index=index_name, columns=column_name)
    pivot_u_x = pd.pivot_table(df_piv_cropped, values=headers[u_x_col_index], index=index_name, columns=column_name)
    pivot_V_abs = pd.pivot_table(df_piv_cropped, values=headers[V_abs_col_index], index=index_name, columns=column_name)
    
    # Create x-y meshgrid
    r_norm_array = pivot_u_r.columns
    x_norm_array = pivot_u_r.index
    r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    # r_norm_values = r_norm.flatten()
    # x_norm_values = x_norm.flatten()
    
    # Construct file path
    raw_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Frame0', 'Export')
    raw_file = os.path.join(raw_dir, f'B{image_nr:04d}.csv')

    df_raw = pd.read_csv(raw_file)
    
    df_raw = process_df(df_raw, D_in, offset_to_wall_center, offset)
    
    # df_raw['x_shift [mm]'] = df_raw['x [mm]'] - (D_in/2 - offset_to_wall_center)
    # df_raw['y_shift [mm]'] = df_raw['y [mm]'] + offset
    
    # df_raw['x_shift_norm'] = df_raw['x_shift [mm]']/D_in
    # df_raw['y_shift_norm'] = df_raw['y_shift [mm]']/D_in
    
    # df_raw['x_shift [m]'] = df_raw['x_shift [mm]']*1e-3
    # df_raw['y_shift [m]'] = df_raw['y_shift [mm]']*1e-3

    headers_raw = df_raw.columns
    
    df_raw_filtered = df_raw[(df_raw[index_name] > bottom_limit) & (df_raw[index_name] < top_limit) & (df_raw[column_name] > left_limit) & (df_raw[column_name] < right_limit)]
    
    # Read intensity
    pivot_intensity = pd.pivot_table(df_raw_filtered, values=headers_raw[2], index=index_name, columns=column_name)
    r_raw_array = pivot_intensity.columns
    x_raw_array = pivot_intensity.index
    r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
    n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
    window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
    
    r_left_raw = r_raw_array[0]
    r_right_raw = r_raw_array[-1]
    x_bottom_raw = x_raw_array[0]
    x_top_raw = x_raw_array[-1]
    
    contour_nr = image_nr - 1
    contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
    
    contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    vector_scale = 20
    vector_width = 0.005
    vector_skip = 2
    box_size = .7
    
    toggle_zoom = True
    
    if flame.Re_D == 3000:
        
        # Re_D = 3000
        # x_left_zoom = -.475
        # y_bottom_zoom = .35
        # clims = (0.9, 1.3)
        # brighten_factor = 4
        # custom_x_ticks = [-.4, -.2, .0, .2]# Replace with your desired tick positions
        
        x_left_zoom = -.55
        y_bottom_zoom = .25
        clims = (0.9, 1.3)
        brighten_factor = 4
        custom_x_ticks = [-.4, -.2, .0, .2]# Replace with your desired tick positions
        
    elif flame.Re_D == 12500:
    
        # Re_D = 12500
        x_left_zoom =  -.2
        y_bottom_zoom = .8
        clims = (0.9, 1.3)
        brighten_factor = 16
        custom_x_ticks = [-.1, .1, .3, .5]# Replace with your desired tick positions
    
    x_right_zoom = x_left_zoom + box_size
    y_top_zoom = y_bottom_zoom + box_size
    
    # fig1, ax1 = plt.subplots()
    
    # ax = ax1
    fontsize = 20
    
    flow_field = ax.pcolor(r_norm, x_norm, pivot_V_abs.values/u_bulk_measured, cmap=colormap)
    
    flow_field.set_clim(clims[0], clims[1])
    cbar = ax.figure.colorbar(flow_field)
    cbar.set_label(cbar_titles[1], rotation=0, labelpad=25, fontsize=28) 
    cbar.ax.tick_params(labelsize=fontsize)
    
    skip = vector_skip
    ax.quiver(r_norm[::skip], x_norm[::skip], pivot_u_r[::skip]/u_bulk_measured, pivot_u_x[::skip]/u_bulk_measured, angles='xy', scale_units='xy', scale=vector_scale, width=vector_width, color='k')
    
    ax.set_xlabel(r'$r/D$', fontsize=fontsize)
    ax.set_ylabel(r'$x/D$', fontsize=fontsize)
    
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_aspect('equal')
    
    custom_x_tick_labels =  [f'{tick:.1f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    ax.set_xticks(custom_x_ticks)
    ax.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    y_tick_step = .2
    custom_y_ticks = np.linspace(y_bottom_zoom, y_top_zoom, 1 + int((y_top_zoom - y_bottom_zoom)/y_tick_step)) # Replace with your desired tick positions
    custom_y_tick_labels =  [f'{tick:.1f}' for tick in custom_y_ticks] # Replace with your desired tick labels
    ax.set_yticks(custom_y_ticks)
    ax.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
    
    # Define your custom colorbar tick locations and labels
    num_ticks = 5
    custom_cbar_ticks = np.linspace(clims[0], clims[1], num_ticks) # Replace with your desired tick positions
    custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
    cbar.set_ticks(custom_cbar_ticks)
    cbar.set_ticklabels(custom_cbar_tick_labels)
    
    # plot flame front contour
    ax.plot(contour_x, contour_y, color='r', ls='solid')
    ax.set_aspect('equal')
    
    fig2, ax2 = plt.subplots()
    
    # brighten_factor = 16
    z = pivot_intensity.values
    raw_field = ax2.pcolor(r_raw, x_raw, z, cmap='gray', vmin=np.min(z.flatten())/brighten_factor, vmax=np.max(z.flatten())/brighten_factor)
    # raw_field.set_clim(1.25, 3)
    
    # cbar = ax2.figure.colorbar(raw_field)
    # cbar.set_label(r'$I_p$', rotation=0, labelpad=15, fontsize=20)
    fontsize = 20
    ax2.set_xlabel(r'$r/D$', fontsize=fontsize)
    ax2.set_ylabel(r'$x/D$', fontsize=fontsize)
    
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize)
    
    ax2.set_aspect('equal')
    
    # plot flame front contour
    ax2.plot(contour_x, contour_y, color='r', ls='solid')
    
    if toggle_zoom:
        
        ax.set_xlim(x_left_zoom, x_right_zoom)
        ax.set_ylim(y_bottom_zoom, y_top_zoom)
        ax2.set_xlim(x_left_zoom, x_right_zoom)
        ax2.set_ylim(y_bottom_zoom, y_top_zoom)
        
    
    ax2.set_xticks(custom_x_ticks)
    ax2.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    ax2.set_yticks(custom_y_ticks)
    ax2.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
    
    # fig1.tight_layout()
    # fig1.savefig(f"figures/H{flame.H2_percentage}_Re{flame.Re_D}_B{image_nr}_V.eps", format="eps", dpi=300, bbox_inches="tight")
    
    # fig2.tight_layout()
    # fig2.savefig(f"figures/H{flame.H2_percentage}_Re{flame.Re_D}_B{image_nr}_Ip.png", format="png", dpi=300, bbox_inches="tight")
    
    # # Add textbox with timestamp
    # left, width = .25, .7
    # bottom, height = .25, .7
    # right = left + width
    # top = bottom + height
    
    # timestamp = (image_nr - image_nrs[0])*(1e3/flame.image_rate)
    # ax.text(right, top,  r'$t = {:.0f}$ ms'.format(timestamp), 
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes,
    #         fontsize=fontsize,
    #         bbox=dict(facecolor="w", edgecolor='k', boxstyle='round')
    #         )

   
def contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0):
    
    segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
    segmented_contour_r = segmented_contour[:, 0, 0]
    segmented_contour_x = segmented_contour[:, 0, 1]
    
    # x and y coordinates of the discretized (segmented) flame front 
    contour_r_corrected = segmented_contour_r*window_size_r_raw + r_left_raw
    contour_x_corrected = segmented_contour_x*window_size_x_raw + x_top_raw
    
    # Non-dimensionalize coordinates by pipe diameter
    contour_r_corrected /= 1 #D_in
    contour_x_corrected /= 1 #D_in
    
    contour_r_corrected_array = np.array(contour_r_corrected)
    contour_x_corrected_array = np.array(contour_x_corrected)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    contour_corrected_coords = np.array([contour_r_corrected_array, contour_x_corrected_array]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    contour_corrected = np.zeros((len(contour_r_corrected_array), 1, 2))
    contour_corrected[:, 0, :] = contour_corrected_coords
    
    return contour_corrected    

def process_df(df, D_in, offset_to_wall_center, offset):
    """
    Process the DataFrame by shifting and normalizing coordinates.

    :param df: DataFrame to process.
    :param D_in: Diameter for normalization.
    :param offset_to_wall_center: Offset for x-coordinate shifting.
    :param offset: Offset for y-coordinate shifting.
    :return: Processed DataFrame.
    """
    df['x_shift [mm]'] = df['x [mm]'] - (D_in/2 - offset_to_wall_center)
    df['y_shift [mm]'] = df['y [mm]'] + offset

    df['x_shift_norm'] = df['x_shift [mm]']/D_in
    df['y_shift_norm'] = df['y_shift [mm]']/D_in

    df['x_shift [m]'] = df['x_shift [mm]']*1e-3
    df['y_shift [m]'] = df['y_shift [mm]']*1e-3

    return df

#%% START OF CODE
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == '__main__':
    
    data_dir = 'U:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'
    
    frame_nr = 0
    segment_length_mm = 1 # units: mm
    window_size = 31 # units: pixels
    
    # Marker sizes
    ms1 = 18
    ms2 = 16
    ms3 = 14
    ms4 = 12
    ms5 = 10
    ms6 = 8
    
    #%% Define cases
    react_names_ls =    [
                        # ('react_h0_c3000_ls_record1', 57),
                        # ('react_h0_s4000_ls_record1', 58),
                        # ('react_h100_c12000_ls_record1', 61),
                        # ('react_h100_c12500_ls_record1', 61),
                        # ('react_h100_s16000_ls_record1', 62)
                        ]
    
    react_names_hs =    [
                        # ('react_h0_f2700_hs_record1', 57),
                        ('react_h0_c3000_hs_record1', 57),
                        # ('react_h0_s4000_hs_record1', 58),
                        # ('react_h100_c12500_hs_record1', 61),
                        # ('react_h100_s16000_hs_record1', 62)
                        ]
    
    
    if react_names_ls:
        spydata_dir = os.path.join(parent_folder, 'spydata\\udf')
    elif react_names_hs:
        spydata_dir = os.path.join(parent_folder, 'spydata')
        
    react_names = react_names_ls + react_names_hs
    
    # Create an empty dictionary for the (non)reacting cases
    react_dict = {}
    flames = []
    
    # piv_method = 'PIV_MP(3x16x16_75%ov_ImgCorr)'
    piv_method = 'PIV_MP(3x16x16_0%ov_ImgCorr)'
    
    for name, nonreact_run_nr in react_names:
    
        fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'
    
        with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
            flame = pickle.load(f)
        
        flames.append(flame)
        
        name = flame.name
        session_nr = flame.session_nr
        recording = flame.record_name
        piv_method = piv_method
        run_nr = flame.run_nr
        Re_D_set = flame.Re_D
        u_bulk_set = flame.u_bulk_measured
        u_bulk_measured = flame.u_bulk_measured
        
        if flame.H2_percentage == 0:
            # Colorbar limits for DNG-AIR
            cbar_max = [2,
                        2,
                        .01, 
                        .02,
                        .05,
                        .1,
                        ]
        elif flame.H2_percentage == 100:
        # Colorbar limits for H2-AIR
            cbar_max = [2,
                        2,
                        .25, 
                        .02,
                        .04,
                        .02,
                        ]
        
        react_dict[run_nr] = [name, session_nr, recording, piv_method, Re_D_set, u_bulk_set, u_bulk_measured, cbar_max, nonreact_run_nr]
        
        distances_above_tube = [.25, .75, 1.25, ]
        r_range_left, poly_left_fit, r_range_right, poly_right_fit, alpha = cone_angle(spydata_dir, name, distances_above_tube)
        
        cone_left_line = np.column_stack((r_range_left, poly_left_fit))
        cone_right_line = np.column_stack((r_range_right, poly_right_fit))
       
    #%% PIV file column indices
    u_r_col_index = 2
    u_x_col_index = 3
    V_abs_col_index = 4
    du_rdr_col_index = 5
    du_rdx_col_index = 6
    du_xdr_col_index = 7
    du_xdx_col_index = 8
    R_xx_col_index = 15 # RADIAL DIRECTION
    R_xy_col_index = 16
    R_yy_col_index = 17 # AXIAL DIRECTION
    TKE_col_index = 20
    
    col_indices =   [
                    u_x_col_index,
                    V_abs_col_index,
                    R_xx_col_index, 
                    R_xy_col_index, 
                    R_yy_col_index,
                    TKE_col_index,
                    ]
    
    cbar_titles =   [
                    r'$\frac{\overline{u_{x}}}{U_{b}}$',
                    r'$\frac{|V|}{U_{b}}$',
                    r'$\frac{R_{rr}}{U_{b}^2}$', # \overline{v\'v\'
                    r'$\frac{R_{rx}}{U_{b}^2}$', # \overline{u\'v\'
                    r'$\frac{R_{xx}}{U_{b}^2}$', # \overline{v\'v\'
                    r'$\frac{k}{U_{b}^2}$', # \overline{v\'v\'
                    ]
    
    #%% Calibration details
    D_in = flame.D_in # Inner diameter of the quartz tube, units: mm
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    
    #%% Start of cases loop
    for key, values in react_dict.items():
        
        axs = []
        
        for q in range(0, len(col_indices)):
            
            fig, ax = plt.subplots()
            
            axs.append(ax)
            
        
        #%%% Lists where index:0 (reactive) and index:1 (non-reacive)
        # Non-reactive: index 1
        
        cbar_max = values[7]
        nonreact_run_nr = values[8]
        
        values_list = [values]
        
        values = non_react_dict[nonreact_run_nr]
        
        values_list.append(values)
        
        headers_list = []
        df_piv_cropped_list = []
        df_piv_list = []
        u_bulk_measured_list = []
        Re_D_list = []
        
        pivot_u_r_norm_list = []
        pivot_u_x_norm_list = []
        pivot_u_r_norm_values_list = []
        pivot_u_x_norm_values_list = []
        pivot_V_abs_norm_values_list = []
        
        r_norm_array_list, r_norm_list, r_norm_values_list, = [], [], []
        x_norm_array_list, x_norm_list, x_norm_values_list = [], [], []
        
        r_uniform_list, x_uniform_list = [], []
        u_r_uniform_list, u_x_uniform_list = [], []
        
        #%%% Start of react + non-react loop
        for ii, values in enumerate(values_list):
            
            name = values[0]
            session_nr = values[1] 
            recording = values[2]
            piv_method = values[3]
            Re_D = values[4]
            u_bulk_set = values[5]
            u_bulk_measured = values[6]
            
            print(name)
            
            Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')
            
            if ii == 0:
                
                # image_nrs = [3167, 3169, 3171, 3173, 3175,  3177]
                # image_nrs = [2306, 2308, 2310, 2312, 2314, 2316] #[4624] #2314 #[4496]
                # image_nrs = [1737, 1738] #[4624] #2314 #[4496]
                image_nrs = [2297, 2299] #[4624] #2314 #[4496]
                
                fig_i, ax_i = plt.subplots()
                for image_nr in image_nrs:
                    plot_cartoons(flame, fig_i, ax_i, image_nr, recording, piv_method)
                
                # fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                
                # for image_i, ax_i in enumerate(axs.ravel()):
                    
                #     image_nr = image_nrs[image_i]
                #     plot_cartoons(flame, ax_i, image_nr, recording, piv_method)
                    
                #     if image_i in [1, 3, 5]:
                #         ax_i.set_ylabel('')
                #         ax_i.tick_params(axis='y', labelleft=False)
                    
                #     if image_i in [0, 1, 2, 3]:
                #         ax_i.set_xlabel('')
                #         ax_i.tick_params(axis='x', labelbottom=False)
                
                # fig.subplots_adjust(wspace=-.75)
                
                # fig.tight_layout()
                
                # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_sequence2'
                # png_path = os.path.join('figures', f"{filename}.png")
                # fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
                        
            df_piv = pd.read_csv(Avg_Stdev_file)
            
            df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
            
            # Get the column headers
            headers = df_piv.columns
            
            bottom_limit = -.5
            top_limit = 2.25
            left_limit = -0.575
            right_limit = 0.575
            index_name = 'y_shift_norm'
            column_name = 'x_shift_norm'
            df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
            
            pivot_u_r = pd.pivot_table(df_piv_cropped, values=headers[u_r_col_index], index=index_name, columns=column_name)
            pivot_u_x = pd.pivot_table(df_piv_cropped, values=headers[u_x_col_index], index=index_name, columns=column_name)
            pivot_V_abs = pd.pivot_table(df_piv_cropped, values=headers[V_abs_col_index], index=index_name, columns=column_name)
            
            # Create x-y meshgrid
            r_norm_array = pivot_u_r.columns
            x_norm_array = pivot_u_r.index
            r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
            r_norm_values = r_norm.flatten()
            x_norm_values = x_norm.flatten()
            
            pivot_u_r_norm = pivot_u_r/u_bulk_measured
            pivot_u_x_norm = pivot_u_x/u_bulk_measured
            pivot_V_abs_norm = pivot_V_abs/u_bulk_measured
            
            pivot_u_r_norm_list.append(pivot_u_r_norm)
            pivot_u_x_norm_list.append(pivot_u_x_norm)
            
            pivot_u_r_norm_values = pivot_u_r_norm.values.flatten()
            pivot_u_x_norm_values = pivot_u_x_norm.values.flatten()
            pivot_V_abs_norm_values = pivot_V_abs_norm.values.flatten()
            
            # Create a uniform grid
            r_uniform = np.linspace(r_norm_values.min(), r_norm_values.max(), len(r_norm_array))
            x_uniform = np.linspace(x_norm_values.min(), x_norm_values.max(), len(x_norm_array))
            r_uniform, x_uniform = np.meshgrid(r_uniform, x_uniform)
            
            # Interpolate the velocity components to the uniform grid
            u_r_uniform = griddata((r_norm_values, x_norm_values), pivot_u_r_norm_values, (r_uniform, x_uniform), method='linear')
            u_x_uniform = griddata((r_norm_values, x_norm_values), pivot_u_x_norm_values, (r_uniform, x_uniform), method='linear')
            
            headers_list.append(headers)
            df_piv_cropped_list.append(df_piv_cropped)
            df_piv_list.append(df_piv)
            u_bulk_measured_list.append(u_bulk_measured)
            Re_D_list.append(Re_D)
            
            pivot_u_r_norm_values_list.append(pivot_u_r_norm_values)
            pivot_u_x_norm_values_list.append(pivot_u_x_norm_values)
            pivot_V_abs_norm_values_list.append(pivot_V_abs_norm_values)
            
            r_norm_array_list.append(r_norm_array)
            r_norm_list.append(r_norm)
            r_norm_values_list.append(r_norm_values)
            
            x_norm_array_list.append(x_norm_array)
            x_norm_list.append(x_norm)
            x_norm_values_list.append(x_norm_values)
            
            r_uniform_list.append(r_uniform)
            x_uniform_list.append(x_uniform)
            u_r_uniform_list.append(u_r_uniform)
            u_x_uniform_list.append(u_x_uniform)
        
   
        for i, col_index in enumerate(col_indices):
            
            ax = axs[i]
            
            pivot_var_list = []
            pivot_var_values_list = []
            
            for p in [0, 1]:
                
                df_piv = df_piv_cropped_list[p]
                headers = headers_list[p]
                
                pivot_var = pd.pivot_table(df_piv, values=headers[col_index], index=index_name, columns=column_name)
                
                # Apply correction for Davis approach for TKE 2D
                if col_index == TKE_col_index:
                    pivot_var *= 2/3
                    
                pivot_var_values = pivot_var.values.flatten() 
                
                pivot_var_list.append(pivot_var)
                pivot_var_values_list.append(pivot_var_values)
                
            if i == 0:
                
                normalize_value = u_bulk_measured_list[0]
                
                # Define radial locations
                r_norm_lines = [.0,]
                
                # Define x-limits
                x_norm_min, x_norm_max = bottom_limit, top_limit
                
                # Define vline stepsize
                vline_step = 0.05
                
                #%%%% Streamlines
                r_start_right = 0.05
                r_step = 0.05
                # r_starts = np.arange(-r_start_right, r_start_right + r_step, r_step)
                r_starts = [.1, .2, .3]
                # r_starts = [.2,]
                
                x_starts = np.linspace(0.2, 0.2, len(r_starts))
                start_points = [(r_starts[i], x_starts[i]) for i in range(len(r_starts))]
                
                width, height = 6, 6
                
                fig, ax6 = plt.subplots()
                
                fig, ax7 = plt.subplots(figsize=(width + 1, height))
                
                fig, ax8 = plt.subplots(figsize=(width + 1, height))
                
                fig9, ax9 = plt.subplots(figsize=(width, height))
                ax9.grid(True)
                
                # Define the region of interest for zooming
                ax9_x1, ax9_x2, ax9_y1, ax9_y2 = -.05, 2.15, -.25, .25  # for example, zoom in on this region

                # Draw a rectangle or any other shape to indicate the zoom area
                ax9.add_patch(plt.Rectangle((ax9_x1, ax9_y1), ax9_x2 - ax9_x1, ax9_y2 - ax9_y1, fill=False, color='k', linestyle='solid', lw=2))
                
                ax9_inset = inset_axes(ax9, width="100%", height="400%", loc='upper left',
                                  # bbox_to_anchor=(x1, y1, x2 - x1, y2 - y1),
                                  bbox_to_anchor=(ax9_x1, ax9_y1 + 7.7, ax9_x2 - ax9_x1, ax9_y2 - ax9_y1),
                                  bbox_transform=ax9.transData,
                                  borderpad=0)
                
                # Set the limits for the inset axes
                ax9_inset.set_xlim(ax9_x1, ax9_x2)
                ax9_inset.set_ylim(ax9_y1, ax9_y2)
                
                ax9_inset.set_xticks([])
                for spine in ax9_inset.spines.values():
                    spine.set_linewidth(2)
    
                for p in [0, 1]:
                    
                    r_uniform = r_uniform_list[p]
                    x_uniform = x_uniform_list[p]
                    u_r_uniform = u_r_uniform_list[p]
                    u_x_uniform = u_x_uniform_list[p]
                    
                    df_piv = df_piv_cropped_list[p]
                    u_bulk_measured = u_bulk_measured_list[p]
                    Re_D = Re_D_list[p]
                    r_norm_values = r_norm_values_list[p]
                    x_norm_values = x_norm_values_list[p]
                    
                    mass_cons, mom_x, mom_r, = ns_incomp_terms(df_piv, D_in, u_bulk_measured, Re_D)
                    
                    if p == 0:
                        streamlines, paths, flame_front_indices, colors = plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform)
                        incomp_indices = plot_ns_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
                        input_indices = flame_front_indices
                    else:
                        streamlines, paths, dummy_indices, colors = plot_streamlines_nonreacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform)
                        incomp_indices = plot_ns_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, dummy_indices, colors)
                        input_indices = dummy_indices
                    
                    plot_mass_cons(p, ax9, ax9_inset, mass_cons, r_norm_values, x_norm_values, streamlines, input_indices, colors)
                    
                    ax9.set_xlim([-.5, 2.2])  # replace with your desired x limits
                    ax9.set_ylim(top=8.25)  # replace with your desired x limits
                    
                    for streamline, path, flame_front_index, color, incomp_index in zip(streamlines, paths, input_indices, colors, incomp_indices):
                        
                        # Use scipy's griddata function to interpolate the velocities along the line
                        velocities = griddata((r_norm_values_list[p], x_norm_values_list[p]), pivot_V_abs_norm_values_list[p], streamline, method='linear')
                        
                        if p == 0:
                            ls = 'solid'
                        elif p == 1:
                            ls = 'dashed'
                            
                        plot_line, = ax6.plot(path, velocities, c=color, marker='None', ls=ls)
                        
                        if p == 0:
                            ax6.plot(path[flame_front_index], velocities[flame_front_index], c=color, marker='*', mec='k', ms=ms5)
                            ax6.plot(path[incomp_index[:]], velocities[incomp_index[:]], lw=5, c=color, marker='None')
                            ax6.plot(path[incomp_index[-1]], velocities[incomp_index[-1]], ls='None', c=color, marker='s', mec='k')
                            
                        # plot_line, = ax6.plot(path, velocities, c=color, marker='None')
                                                                  
                        ax6.set_xlabel(r'$s/D$', fontsize=20)
                        ax6.set_ylabel(cbar_titles[1], rotation=0, labelpad=15, fontsize=24)
                        ax6.grid(True)
                        
                        dpdx = mom_x[2] #*(u_bulk_measured**2)/(D_in*1e-3)
                        dpdr = mom_r[2] #*(u_bulk_measured**2)/(D_in*1e-3)
                        plot_pressure_along_streamline(ax7, ax8, dpdr, dpdx, r_norm_values, x_norm_values, streamlines, incomp_indices, colors, p)
                        
                    styles_react = ['solid', 'solid', 'solid']
                    styles_nonreact = ['dashed', 'dashed', 'dashed']
                    # make list of one line -- doesn't matter what the coordinates are
                    dummy_line = [[(0, 0)]]
                    # set up the proxy artist
                    lc_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_react, colors=colors)
                    lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
                    
                    # create the legend
                    ax6.legend([lc_react, lc_nonreact], ['reacting flow', 'non reacting flow'], handler_map={type(lc_react): HandlerDashedLines()},
                               handlelength=3, handleheight=3)
                    
                    ax6.set_xlim(-0.05, 2.25)
                    ax6.set_ylim(0.9, 1.7)
                    
                    p = 0
                    df_piv = df_piv_cropped_list[p]
                    u_bulk_measured = u_bulk_measured_list[p]
                    Re_D = Re_D_list[p]
                    r_norm_values = r_norm_values_list[p]
                    x_norm_values = x_norm_values_list[p]
                
            elif i == 1:
                
                normalize_value = u_bulk_measured_list[0]

            else:
                
                normalize_value = u_bulk_measured_list[0]**2
            
            # fig9.tight_layout()
            
            # Mean absolute velocity
            flow_field = ax.pcolor(r_norm_list[0], x_norm_list[0], pivot_var_list[0].values/normalize_value, cmap=colormap)
            
            ax._X_data = r_norm_list[0]
            ax._Y_data = x_norm_list[0]
            ax._Z_data = pivot_var_list[0].values/normalize_value
            ax._cmap = colormap
            ax._clim = [0, cbar_max[i]]
            
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            
            if i == 1:
                skip = 4
                ax.quiver(r_norm_list[0][::skip], x_norm_list[0][::skip], pivot_u_r_norm_list[0][::skip], pivot_u_x_norm_list[0][::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
                
            
            if i == 3:
                flow_field.set_clim(-cbar_max[i], cbar_max[i])
            else:
                flow_field.set_clim(0, cbar_max[i])
            
            
            if i == 5:
                
                image_nr = 1
                raw_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Resize', 'Frame0', 'Export')
                raw_file = os.path.join(raw_dir, f'B{image_nr:04d}.csv')

                df_raw = pd.read_csv(raw_file)
                
                df_raw = process_df(df_raw, D_in, offset_to_wall_center, offset)

                headers_raw = df_raw.columns
                
                # Read intensity
                pivot_intensity = pd.pivot_table(df_raw, values=headers_raw[2], index=index_name, columns=column_name)
                r_raw_array = pivot_intensity.columns
                x_raw_array = pivot_intensity.index
                r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
                n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
                window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
                
                r_left_raw = r_raw_array[0]
                r_right_raw = r_raw_array[-1]
                x_bottom_raw = x_raw_array[0]
                x_top_raw = x_raw_array[-1]
                
                if flame.Re_D == 3000:
                    image_nrs = [10, 1112, 2018, 2259]
                
                elif flame.Re_D == 4000:
                    image_nrs = [12, 1879, 2204, 2361]
                
                elif flame.Re_D == 12500:
                    image_nrs = [471, 1251, 1473] # Re=12500
                    
                elif flame.Re_D == 16000:
                    image_nrs = [1500, 1600, 2171]
                    
                for image_nr in image_nrs:
                    contour_nr = image_nr - 1
                    contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
                    contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
                    ax.plot(contour_x, contour_y, color='r', ls='solid') 
            
            # Define your custom colorbar tick locations and labels
            num_ticks = 6
            custom_cbar_ticks = np.linspace(0, cbar_max[i], num_ticks) # Replace with your desired tick positions
            
            if cbar_max[i] < 1:
                custom_cbar_tick_labels = [f'{tick:.2f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
            else:
                custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
            
            fontsize = 20
            
            # Set the colorbar ticks and labels
            if flame.Re_D in [3000, 12500]: 
                cbar = ax.figure.colorbar(flow_field)
                cbar.set_ticks(custom_cbar_ticks)
                cbar.set_ticklabels(custom_cbar_tick_labels)
                cbar.set_label(cbar_titles[i], rotation=0, labelpad=25, fontsize=28) 
                cbar.ax.tick_params(labelsize=fontsize)
            
            ax.set_aspect('equal')
            ax.set_xlabel(r'$r/D$', fontsize=fontsize)
            ax.set_ylabel(r'$x/D$', fontsize=fontsize)
            
            ax.set_xlim([-.55, .55])  # replace with your desired x limits
            ax.set_ylim([0/D_in, 2.2])  # replace with your desired y limits
            # ax.set_xlim(x_limits)  # replace with your desired x limits
            # ax.set_ylim(y_limits)  # replace with your desired y limits
            
            # Cone angle
            ax.plot(r_range_left, poly_left_fit, c='k', ls='dashed') 
            ax.plot(r_range_right, poly_right_fit, c='k', ls='dashed')
            
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            
            if i == 1:
                
                # Determine shared colorbar limits
                vmin = 0 
                vmax = 2
                # vmax= 1.25
                
                # Create a figure and a 3D axis
                fig0 = plt.figure()
    
                # =============
                # set up the axes for the first plot
                z_bottom = 0.
                z = pivot_var_list[1].values/u_bulk_measured_list[1]
                z[z < z_bottom] = np.nan
                
                ax = fig0.add_subplot(1, 1, 1, projection='3d')
                surf = ax.plot_surface(r_norm_list[1], x_norm_list[1], z, cmap=colormap, vmin=vmin, vmax=vmax, edgecolors='k', lw=0.1)
                
                # Adjust the limits, add labels, title, etc.
                ax.set_xlabel(r'$r/D$', labelpad=5)
                ax.set_ylabel(r'$x/D$', labelpad=5)
                # ax.set_zlabel('Z Label')
                ax.set_title('Non-reactive flow')
                ax.set_aspect('equal')
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # x-axis
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # y-axis
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # z-axis
                ax.set_zlim(z_bottom, vmax)
                cbar = fig0.colorbar(surf, pad=0.01)
                cbar.set_label(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
                ax.set_xticks([-.5, 0., .5])
                ax.set_yticks([0, 1, 2])
                ax.set_zticks([])
                fig0.tight_layout()
                
                # Create a figure and a 3D axis
                fig1 = plt.figure()
                z = pivot_var_list[0].values/normalize_value
                z[z < z_bottom] = np.nan
                
                z_values = z.flatten()
                
                cone_left_line_3d = griddata((r_norm_values, x_norm_values), z_values, cone_left_line, method='linear')
                cone_right_line_3d = griddata((r_norm_values, x_norm_values), z_values, cone_right_line, method='linear')
                
                ax = fig1.add_subplot(1, 1, 1, projection='3d')
                ax.plot(cone_left_line[:,0], cone_left_line[:,1], cone_left_line_3d, c='k', ls='dashed', zorder=10)
                ax.plot(cone_right_line[:,0], cone_right_line[:,1], cone_right_line_3d, c='k', ls='dashed', zorder=10)
                
                surf = ax.plot_surface(r_norm_list[0], x_norm_list[0], z, cmap=colormap, vmin=vmin, vmax=vmax, edgecolors='k', lw=0.1, zorder=-1)
                
                # Adjust the limits, add labels, title, etc.
                ax.set_xlabel(r'$r/D$', labelpad=5)
                ax.set_ylabel(r'$x/D$', labelpad=5)
                # ax.set_zlabel(cbar_titles[i])
                ax.set_title('Reactive flow')
                ax.set_aspect('equal')
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # x-axis
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # y-axis
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # z-axis
                ax.set_zlim(z_bottom, vmax)
                cbar = fig1.colorbar(surf, pad=0.01)
                cbar.set_label(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
                ax.set_xticks([-.5, 0., .5])
                ax.set_yticks([0, 1, 2])
                ax.set_zticks([])
                fig1.tight_layout()

    # print(f'The cone angle: {alpha}')

# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()

#%% Save images
# Get a list of all currently opened figures
# figure_ids = plt.get_fignums()
# figure_ids = [1, 2, 12, 13, 21]

# if react_names_ls:
#     folder = 'ls'
# else:
#     folder = 'hs'
    
# # Apply tight_layout to each figure
# for fid in figure_ids:
#     fig = plt.figure(fid)
#     fig.tight_layout()
#     filename = f'H{flame.H2_percentage}_Re{Re_D_list[0]}_fig{fid}'
    
#     # Constructing the paths
#     if fid == 1:
        
#         png_path = os.path.join('figures', f'{folder}', f'{filename}.png')
#         # pkl_path = os.path.join('pickles', f'{folder}', f'{filename}.pkl')
        
#         # Saving the figure in EPS format
#         fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
#     else:
        
#         eps_path = os.path.join('figures', f'{folder}', f'{filename}.eps')
#         # pkl_path = os.path.join('pickles', f'{folder}', f'{filename}.pkl')
        
#         # Saving the figure in EPS format
#         fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    
    
    # Pickling the figure
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(fig, f)
        













    
    