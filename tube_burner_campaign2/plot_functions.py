# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:27:52 2024

@author: luuka
"""

#%% IMPORT PACKAGES
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.collections as mcol
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

#%% IMPORT USER DEFINED PACKAGES
from parameters import flame, interpolation_method, fontsize, ms1, ms2, ms3, ms4, ms5, ms6
from cone_angle import cone_angle
from functions import intersection

#%% OBJECTS
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
    
#%%  FUNCTIONS [fans_terms.py]
def plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform, start_points):
    
    strms = []
    streamlines = []
    flame_front_indices = []
    paths = []
    
    fig1, ax1 = plt.subplots()
    
    # Cone angle
    distances_above_tube = [.25, .75, 1.25,]
    r_range_left, poly_left_fit, r_range_right, poly_right_fit, alpha = cone_angle(distances_above_tube)

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
        
        skip = 2
        ax1.quiver(r_uniform[::skip], x_uniform[::skip], u_r_uniform[::skip], u_x_uniform[::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
        
        colors.append(strm.lines.get_color())
        strms.append(strm)
       
    for c, strm in enumerate(strms):
        
        segments = strm.lines.get_segments()
        line = np.array([segment[0] for segment in segments[:-1]] + [segments[-1][1]])
        
        line_r, line_x = line[:,0], line[:,1]
        distance = np.cumsum(np.sqrt(np.ediff1d(line_r, to_begin=0)**2 + np.ediff1d(line_x, to_begin=0)**2))
        
        distance = distance/distance[-1]
        
        fr, fx = interp1d(distance, line_r), interp1d(distance, line_x)
        
        n_interp_points = np.linspace(0, 1, 501)
        r_interp, x_interp = fr(n_interp_points), fx(n_interp_points)
        
        line = np.column_stack((r_interp, x_interp))
        
        # Filter line points where x > x_norm_min
        streamline_x_start = .1
        streamline_r_max = .5
        streamline_x_max = 2.2
        # streamline = line[(np.abs(line[:,0]) < streamline_r_max) & (line[:,1] > 0.05) & (line[:,1] < streamline_x_max)]
        streamline = line[(np.abs(r_interp) < streamline_r_max) & (x_interp >= streamline_x_start) & (x_interp < streamline_x_max)]
        
        streamline_r, streamline_x = streamline[:,0], streamline[:,1]
        streamline = np.column_stack((streamline_r, streamline_x))
        
        ax1.plot(streamline_r, streamline_x, color=colors[c], marker='None', ls='-', lw=2)
        ax1.plot(streamline_r[0], streamline_x[0], color=colors[c], marker='^', ls='None', mec='k', ms=ms5)
        ax1.plot(streamline_r[-1], streamline_x[-1], color=colors[c], marker='o', ls='None', mec='k', ms=ms5)
        
        # ax1.set_title(name)
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
        
        if streamline_r[0] < 0:
            r_intersect, x_intersect = intersection(r_range_left, poly_left_fit, streamline_r, streamline_x)

        else:
            r_intersect, x_intersect = intersection(r_range_right, poly_right_fit, streamline_r, streamline_x)
        
        intersect_point = np.array([r_intersect[0], x_intersect[0]])
        
        # Mask out y-values in y2_interp that are greater than x_intersection
        valid_indices = np.where(streamline_x <= x_intersect)[0]
        
        streamline_point_difference = streamline[valid_indices] - intersect_point
        distances_to_front = np.linalg.norm(streamline_point_difference, axis=1)
        
        # Get the index of the minimum distance
        min_distance_idx = valid_indices[np.argmin(distances_to_front)]
        
        # Retrieve the closest coordinates
        r_closest, x_closest = streamline[min_distance_idx]
        # y_closest = streamline_x[min_distance_idx]
        
        ax1.plot(r_intersect, x_intersect, marker='*', ms=ms2, c=colors[c], mec='k')
        # ax1.plot(r_closest, x_closest, 'sr')
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        flame_front_indices.append(min_distance_idx)
        streamlines.append(streamline)
        paths.append(cumulative_distances)
    
    # ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 12})
            
    return streamlines, paths, flame_front_indices, colors 


def plot_mass_cons(mass_cons, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    # colors = ['r', 'g', 'b']
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Conservation of mass')
    fig, ax = plt.subplots()
    ax.grid(True)
    # ax.set_ylim([-.25, 8.])  # replace with your desired x limits
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\rho^*\mathbf{u^{*}}}$'
    
    export_data = {
        'cumulative_distances': [],
        'term_along_line': [],
        'flame_front_index': []
    }
    
    # First mass conservation
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                ax.plot(cumulative_distances[:], term_along_line[:], ls='solid', marker='None', label=k)
                
                ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ms=ms4, mec='k')
                ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                
                ax.set_xlabel(x_label, fontsize=20)
                ax.set_ylabel(y_label, fontsize=20) 
                
                # Append data to each key in export_data
                export_data['cumulative_distances'].append(cumulative_distances.tolist())
                export_data['term_along_line'].append(term_along_line.tolist())
                export_data['flame_front_index'].append(flame_front_index)
    
    # Exporting the data
    with open('exported_data.pkl', 'wb') as file:
        pickle.dump(export_data, file)
        
    # marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    # handles, labels = ax.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    # handles.extend([marker_dummy1])
    
    # ax.legend(handles=handles)


def plot_fans_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    width, height = 9, 9
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True)
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\rho^*\mathbf{u^{*}}}$'
    
    # First mass conservation
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                ax.plot(cumulative_distances[:], term_along_line[:], ls='solid', marker='None', label=k)
                
                ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms4)
                ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                
                ax.set_xlabel(x_label, fontsize=20)
                ax.set_ylabel(y_label, fontsize=20) 
    
    marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, 
         mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    handles, labels = ax.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    handles.extend([marker_dummy1])
    
    ax.legend(handles=handles)
    
    # Second momentum equations
    mom_x_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    # '[4] Viscous diffusion',
                    '[4] Favre normal stress',
                    '[5] Favre shear stress'
                    ]
    
    mom_r_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    # '[4] Viscous diffusion',
                    '[4] Favre shear stress',
                    '[5] Favre normal stress'
                    ]
    
    mom_markers = ['v', '^', 'o', 's', 'p', 'd']
    mom_markers = ['None', 'None', 'None', 'None', 'None', 'None']
    
    
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
        ax2.set_ylabel(r'Dimensionless terms', fontsize=fontsize_label)
        
        fig3, ax3 = plt.subplots(figsize=(width, height))
        # ax3.set_title('Conservation of radial momentum r=' + str(round(line[0][0], 1)))
        ax3.grid(True)
        ax3.set_xlabel(x_label, fontsize=fontsize_label)
        ax3.set_ylabel(r'Dimensionless terms', fontsize=fontsize_label)
        
        bandwith = .35
        ax2.set_ylim([-2*bandwith, bandwith])  # replace with your desired y limits
        ax3.set_ylim([-2*bandwith, bandwith])  # replace with your desired y limits
        
        for j, terms in enumerate([mom_x, mom_r]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                color, linestyle, mom_marker = map(lambda x: x[k], [colors, linestyles, mom_markers])

                if j == 0:
                    
                    ax2.plot(cumulative_distances, term_along_line, c=colors[k], ls=linestyles[k], marker=mom_markers[k], label=mom_x_labels[k], lw=2)
                    ax2.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                    ax2.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms4)
                    ax2.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                    
                if j == 1:
    
                    ax3.plot(cumulative_distances, term_along_line, c=colors[k], ls=linestyles[k], marker=mom_markers[k], ms=ms1, label=mom_r_labels[k], lw=2)
                    ax3.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                    ax3.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>',  mec='k', ms=ms4)
                    ax3.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                    
        # ax2.legend(title='Axial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        # ax3.legend(title='Radial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        # ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.225, .45))
        # ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.0, .425))
        
        ax2.legend(loc='lower left', prop={'size': 18})
        ax3.legend(loc='lower left', bbox_to_anchor=(0, .1), prop={'size': 18})
        
        ax2.tick_params(axis='both', labelsize=20)
        ax3.tick_params(axis='both', labelsize=20)
        
        
def plot_pressure_along_streamline(dpdr, dpdx, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    width, height = 6, 6
    fig1, ax1 = plt.subplots(figsize=(width, height))
    
    fontsize_label = 24
    
    x_label = r'$s/D$'
    y_label = r'$\overline{p^*} - \overline{p_{0}^*}$'
    
    ax1.set_xlabel(x_label, fontsize=fontsize_label)
    ax1.set_ylabel(y_label, fontsize=fontsize_label)
    
    # Define the region of interest for zooming
    ax1_x1, ax1_x2, ax1_y1, ax1_y2 = -.1, .5, -.01, .06  # for example, zoom in on this region

    # Draw a rectangle or any other shape to indicate the zoom area
    ax1.add_patch(plt.Rectangle((ax1_x1, ax1_y1), ax1_x2 - ax1_x1, ax1_y2 - ax1_y1, fill=False, color='k', linestyle='solid', lw=2))
    
    if flame.Re_D == 3000:
        y_lims = [-.325, .125]
        bbox_to_anchor=(ax1_x1 + 1.725 , ax1_y1 + .055, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="225%"
        height="225%"
    elif flame.Re_D == 4000:
        y_lims = [-.325, .125]
        # bbox_to_anchor=(ax1_x1 + .15 , ax1_y1 -.225, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        bbox_to_anchor=(ax1_x1 + 1.725 , ax1_y1 + .055, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="225%"
        height="225%"
    elif flame.Re_D == 12500:
        y_lims = [-.12, .1]
        bbox_to_anchor=(ax1_x1 + 2., ax1_y1 + .035, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="175%"
        height="175%"
    elif flame.Re_D == 16000:
        y_lims = [-.12, .1]
        bbox_to_anchor=(ax1_x1 + 2., ax1_y1 + .035, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="175%"
        height="175%"
        
    ax1_inset = inset_axes(ax1, width=width, height=height, loc='upper left',
                      # bbox_to_anchor=(x1, y1, x2 - x1, y2 - y1),
                      bbox_to_anchor=bbox_to_anchor,
                      bbox_transform=ax1.transData,
                      borderpad=0)
    
    
    # Set the limits for the inset axes
    ax1_inset.set_xlim(ax1_x1, ax1_x2)
    ax1_inset.set_ylim(ax1_y1, ax1_y2)
    
    ax1_inset.set_xticks([.0, .2, .4])
    ax1_inset.set_yticks([.0, .02, .04, .06])
    
    # ax_inset = fig1.add_axes([.575, .55, .3, .3]) # x, y, width, height (in figure coordinate)
    
    styles_react = ['solid', 'solid', 'solid']
    styles_nonreact = ['dashed', 'dashed', 'dashed']
    
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        line_r, line_x = line[:,0], line[:,1]
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
          
        dpdr_values = dpdr.values.flatten()
        dpdx_values = dpdx.values.flatten()
        
        dpdr_along_line = griddata((r_norm_values, x_norm_values), dpdr_values, line, method=interpolation_method)
        dpdx_along_line = griddata((r_norm_values, x_norm_values), dpdx_values, line, method=interpolation_method)
        
        p_along_line = np.zeros(dpdr_along_line.shape[0])
    
        for i in range(1, dpdr_along_line.shape[0]):
            
            ds = cumulative_distances[i] - cumulative_distances[i-1]
            dr = line_r[i] - line_r[i-1]
            dx = line_x[i] - line_x[i-1]
            
            drds = dr/ds
            dxds = dx/ds
            
            
            # dr *= D_in*1e-3
            # dx *= D_in*1e-3
            
            p_along_line[i] = p_along_line[i-1] + 0.5 * (dpdr_along_line[i] + dpdr_along_line[i-1]) * dr + 0.5 * (dpdx_along_line[i] + dpdx_along_line[i-1]) * dx
            
        ax1.plot(cumulative_distances, p_along_line, c=color, marker='None', ls='solid')
        ax1.plot(cumulative_distances[flame_front_index], p_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
        ax1.plot(cumulative_distances[0], p_along_line[0], color=color, marker='>', ls='None', mec='k', ms=ms4)
        ax1.plot(cumulative_distances[-1], p_along_line[-1], color=color, marker='o', ls='None', mec='k', ms=ms5)
        
        # Create an inset with zoomed-in plot
        ax1_inset.plot(cumulative_distances[0], p_along_line[0], color=color, marker='>', ls='None', mec='k', ms=ms4)
        ax1_inset.plot(cumulative_distances, p_along_line, c=color, marker='None', ls='solid')
        
    # make proxy artists
    # make list of one line -- doesn't matter what the coordinates are
    # dummy_line = [[(0, 0)]]
    # set up the proxy artist
    # lc_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_react, colors=colors)
    # lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
    
    ax1.set_xlim(right=3)  # replace with your desired x limits
    ax1.set_ylim(y_lims)  # replace with your desired x limits
    ax1.grid(True)
    
    # ax_inset.set_xlim(x1, x2)
    # ax_inset.set_ylim(y1, y2)
    
    # Hide the x and y axis ticks
    # ax_inset.set_xticks([])
    # ax_inset.set_yticks([])
    
    ax1.tick_params(axis='both', labelsize=20)
    
    x_label = r'$s/D$'
    y_label = r'$\frac{dp^{*}}{ds^{*}}$'
    
    # # create the legend
    # if flame.Re_D == 4000:
    #     ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #                 handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, .65))
    # elif flame.Re_D == 16000:
    #     ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #                     handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, 1.))
        
    # ax2.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #            handlelength=3, handleheight=3)  
    

#%%  FUNCTIONS [reacting_flow_fields.py]

def plot_pressure_along_streamline_old(ax1, ax2, dpdr, dpdx, r_norm_values, x_norm_values, lines, incomp_indices, colors, p):
    
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
        
        dpdr_along_line = griddata((r_norm_values, x_norm_values), dpdr_values, line, method=interpolation_method)
        dpdx_along_line = griddata((r_norm_values, x_norm_values), dpdx_values, line, method=interpolation_method)
        
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
    term_along_line = griddata((r_norm_values, x_norm_values), p_field_values, line, method=interpolation_method)
    
    ax1.scatter(cumulative_distances, term_along_line, marker='o', label=f'pressure')

    ax1.legend()
    
def plot_mass_cons_old(p, ax, ax_inset, mass_cons, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
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
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
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
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
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
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
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



def plot_streamlines_nonreacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform, start_points):
    
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
        ax1.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax1.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        ax1.tick_params(axis='both', labelsize=fontsize)
        
        # ax1.set_ylabel(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        ax1.grid(False)
        ax1.set_xlim(left=-.55, right=.55)
        ax1.set_ylim(bottom=0, top=2.25)
        
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
    

def plot_cartoons(flame, image_nrs, recording, piv_method):
    
    # Vector settings for quiver plot
    vector_scale = 20
    vector_width = 0.005
    vector_skip = 1
    box_size = .7
    
    # Zoom settings
    # Toggle zoom
    toggle_zoom = True
    
    # Set zoom settings
    if flame.Re_D == 3000:
        
        # Re_D = 3000
        # x_left_zoom = -.475
        # y_bottom_zoom = .35
        # clims = (0.9, 1.3)
        # brighten_factor = 4
        # custom_x_ticks = [-.4, -.2, .0, .2]# Replace with your desired tick positions
        
        box_size = .45
        x_left_zoom = -.5
        y_bottom_zoom = .5
        clims = (0.9, 1.3)
        brighten_factor = 4
        custom_x_ticks = [-.4, -.3, -.2, -.1] # Replace with your desired tick positions
        
    elif flame.Re_D == 12500:
    
        # Re_D = 12500
        x_left_zoom =  -.2
        y_bottom_zoom = .8
        clims = (0.9, 1.3)
        brighten_factor = 16
        custom_x_ticks = [-.1, .1, .3, .5] # Replace with your desired tick positions
    
    x_right_zoom = x_left_zoom + box_size
    y_top_zoom = y_bottom_zoom + box_size
    
    fig1, ax1 = plt.subplots()
    
    fig2, ax2 = plt.subplots()
    
    colors = ['r', 'lime']
    linestyles = ['-', '--']
    lw = 2
    
    for ls, color, image_nr in zip(linestyles, colors, image_nrs):
        
        # PIV directory
        piv_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', recording, piv_method, 'Export')
        piv_file = os.path.join(piv_dir, f'B{image_nr:04d}.csv')
        
        # Read the PIV file and add coordinate system translation
        df_piv = pd.read_csv(piv_file)
        df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
        
        # Get the column headers of the PIV file
        headers = df_piv.columns
        
        # Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
        bottom_limit = -100
        top_limit = 100
        left_limit = -100
        right_limit = 100
        index_name = 'y_shift_norm'
        column_name = 'x_shift_norm'
        
        # Cropped PIV dataframe based on non-dimensional limits in r- (left, right) and x-direction (bottom, top)
        df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
        
        # Obtain velocity fields
        pivot_u_r = pd.pivot_table(df_piv_cropped, values=headers[u_r_col_index], index=index_name, columns=column_name)
        pivot_u_x = pd.pivot_table(df_piv_cropped, values=headers[u_x_col_index], index=index_name, columns=column_name)
        pivot_V_abs = pd.pivot_table(df_piv_cropped, values=headers[V_abs_col_index], index=index_name, columns=column_name)
        
        # Create r,x PIV grid
        r_norm_array = pivot_u_r.columns
        x_norm_array = pivot_u_r.index
        r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    
        # Raw Mie-scattering directory
        raw_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Frame0', 'Export')
        raw_file = os.path.join(raw_dir, f'B{image_nr:04d}.csv')
        
        # Read the raw Mie-scattering image and add coordinate system translation
        df_raw = pd.read_csv(raw_file)
        df_raw = process_df(df_raw, D_in, offset_to_wall_center, offset)
        
        # Get the column headers of the raw Mie-scattering image file
        headers_raw = df_raw.columns
        
        # Cropped raw Mie-scattering image based on Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
        df_raw_filtered = df_raw[(df_raw[index_name] > bottom_limit) & (df_raw[index_name] < top_limit) & (df_raw[column_name] > left_limit) & (df_raw[column_name] < right_limit)]
        
        # Obtain intensity field
        pivot_intensity = pd.pivot_table(df_raw_filtered, values=headers_raw[2], index=index_name, columns=column_name)
        
        # Create r,x raw Mie scattering grid
        r_raw_array = pivot_intensity.columns
        x_raw_array = pivot_intensity.index
        r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
        n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
        window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
        
        # Parameters for correcting contours from pixel coordinates to physical coordinates
        r_left_raw = r_raw_array[0]
        r_right_raw = r_raw_array[-1]
        x_bottom_raw = x_raw_array[0]
        x_top_raw = x_raw_array[-1]
        
        # Contour correction (raw -> world)
        contour_nr = image_nr - 1
        contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
        contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
        
        if image_nr == image_nrs[0]:
            
            # Figure 1: Plot velocity contour
            flow_field = ax1.pcolor(r_norm, x_norm, pivot_V_abs.values/u_bulk_measured, cmap=colormap)
            
            flow_field.set_clim(clims[0], clims[1])
            cbar = ax1.figure.colorbar(flow_field)
            cbar.set_label(cbar_titles[1], rotation=0, labelpad=25, fontsize=28) 
            cbar.ax.tick_params(labelsize=fontsize)
            
            # Figure 1: Plot velocity vectors
            skip = vector_skip
            ax1.quiver(r_norm[::skip], x_norm[::skip], pivot_u_r[::skip]/u_bulk_measured, pivot_u_x[::skip]/u_bulk_measured, angles='xy', scale_units='xy', scale=vector_scale, width=vector_width, color='k')
    
            
            # Figure 2: Plot raw Mie-scattering image
            pivot_intensity_values = pivot_intensity.values
            raw_field = ax2.pcolor(r_raw, x_raw, pivot_intensity_values, cmap='gray', 
                                   vmin=np.min(pivot_intensity_values.flatten())/brighten_factor, 
                                   vmax=np.max(pivot_intensity_values.flatten())/brighten_factor
                                   )
            
            # raw_field.set_clim(1.25, 3)
            
            # cbar = ax2.figure.colorbar(raw_field)
            # cbar.set_label(r'$I_p$', rotation=0, labelpad=15, fontsize=20)
            
        # Figure 1: Plot flame front contour
        ax1.plot(contour_x, contour_y, color=color, ls=ls, lw=lw)
        
        # Figure 2: Plot flame front contour
        ax2.plot(contour_x, contour_y, color=color, ls=ls, lw=lw)
        
    # Set labels for both figures
    ax1.set_xlabel(r'$r/D$', fontsize=fontsize)
    ax1.set_ylabel(r'$x/D$', fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax1.set_aspect('equal')
    
    ax2.set_xlabel(r'$r/D$', fontsize=fontsize)
    ax2.set_ylabel(r'$x/D$', fontsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.set_aspect('equal')
        
    if toggle_zoom:
        
        ax1.set_xlim(x_left_zoom, x_right_zoom)
        ax1.set_ylim(y_bottom_zoom, y_top_zoom)
        ax2.set_xlim(x_left_zoom, x_right_zoom)
        ax2.set_ylim(y_bottom_zoom, y_top_zoom)
    
    custom_x_tick_labels =  [f'{tick:.1f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    
    y_tick_step = .1
    custom_y_ticks = np.linspace(y_bottom_zoom, y_top_zoom, 1 + int((y_top_zoom - y_bottom_zoom)/y_tick_step)) # Replace with your desired tick positions
    custom_y_tick_labels =  [f'{tick:.1f}' for tick in custom_y_ticks] # Replace with your desired tick labels
    
    # Define your custom colorbar tick locations and labels
    num_ticks = 5
    custom_cbar_ticks = np.linspace(clims[0], clims[1], num_ticks) # Replace with your desired tick positions
    custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
    cbar.set_ticks(custom_cbar_ticks)
    cbar.set_ticklabels(custom_cbar_tick_labels)
    
    ax1.set_xticks(custom_x_ticks)
    ax1.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    ax1.set_yticks(custom_y_ticks)
    ax1.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels

    ax2.set_xticks(custom_x_ticks)
    ax2.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    ax2.set_yticks(custom_y_ticks)
    ax2.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
        
    fig1.tight_layout()
    fig1.savefig(f"figures/H{flame.H2_percentage}_Re{flame.Re_D}_B{image_nrs[0]}_V.eps", format="eps", dpi=300, bbox_inches="tight")
    
    fig2.tight_layout()
    fig2.savefig(f"figures/H{flame.H2_percentage}_Re{flame.Re_D}_B{image_nrs[0]}_Ip.png", format="png", dpi=300, bbox_inches="tight")
    
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

   
# def contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0):
    
#     segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
#     segmented_contour_r = segmented_contour[:, 0, 0]
#     segmented_contour_x = segmented_contour[:, 0, 1]
    
#     # x and y coordinates of the discretized (segmented) flame front 
#     contour_r_corrected = segmented_contour_r*window_size_r_raw + r_left_raw
#     contour_x_corrected = segmented_contour_x*window_size_x_raw + x_top_raw
    
#     # Non-dimensionalize coordinates by pipe diameter
#     contour_r_corrected /= 1 #D_in
#     contour_x_corrected /= 1 #D_in
    
#     contour_r_corrected_array = np.array(contour_r_corrected)
#     contour_x_corrected_array = np.array(contour_x_corrected)
    
#     # Combine the x and y coordinates into a single array of shape (n_coords, 2)
#     contour_corrected_coords = np.array([contour_r_corrected_array, contour_x_corrected_array]).T

#     # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
#     contour_corrected = np.zeros((len(contour_r_corrected_array), 1, 2))
#     contour_corrected[:, 0, :] = contour_corrected_coords
    
#     return contour_corrected    

# def process_df(df, D_in, offset_to_wall_center, offset):
#     """
#     Process the DataFrame by shifting and normalizing coordinates.

#     :param df: DataFrame to process.
#     :param D_in: Diameter for normalization.
#     :param offset_to_wall_center: Offset for x-coordinate shifting.
#     :param offset: Offset for y-coordinate shifting.
#     :return: Processed DataFrame.
#     """
#     df['x_shift [mm]'] = df['x [mm]'] - (D_in/2 - offset_to_wall_center)
#     df['y_shift [mm]'] = df['y [mm]'] + offset

#     df['x_shift_norm'] = df['x_shift [mm]']/D_in
#     df['y_shift_norm'] = df['y_shift [mm]']/D_in

#     df['x_shift [m]'] = df['x_shift [mm]']*1e-3
#     df['y_shift [m]'] = df['y_shift [mm]']*1e-3

#     return df
