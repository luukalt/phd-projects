# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:25:24 2023

@author: laaltenburg

[A] Conservation of mass
A1: mass conservation in axial direction
A2: mass conservation in radial direction

[B] Conservation of momentum [Axial]
B1: Advection of U_x in axial direction
B2: Advection of U_x in radial direction
B3: Viscous term
B4: Reynolds normal stress term
B5: Reynolds shear stress term

[C] Conservation of momentum [Radial]  
C1: Advection of U_r in axial direction
C2: Advection of U_r in radial direction
C3: Viscous term
C4: Reynolds shear stress term
C5: Reynolds normal stress term


File columns:
0: x [mm]
1: y [mm]
2: Velocity u [m/s]
3: Velocity v [m/s]
4: Velocity |V| [m/s]
5: du/dx [1/s]
6: du/dy [1/s]
7: dv/dx [1/s]
8: dv/dy [1/s]
9: Vorticity w_z (dv/dx - du/dy) [1/s]
10: |Vorticity| [1/s]
11: Divergence 2D (du/dx + dv/dy) [1/s]
12: Swirling strength 2D (L_ci) [1/s^2]
13: Average kinetic energy [(m/s)^2]
14: Number of vectors [n]
15: Reynolds stress Rxx [(m/s)^2]
16: Reynolds stress Rxy [(m/s)^2]
17: Reynolds stress Ryy [(m/s)^2]
18: Standard deviation Vx [m/s]
19: Standard deviation Vy [m/s]
20: Turbulent kinetic energy [(m/s)^2]
21: Turbulent shear stress [(m/s)^2]
22:	x_shift [mm]
23:	y_shift [mm]
24:	x_shift_norm
25:	y_shift_norm
26:	x_shift [m]
27:	y_shift [m]

# axial direction: axis=0
# radial direction: axis=1
"""

#% IMPORT PACKAGES
# import os
import pickle
import numpy as np
import pandas as pd
# import matplotlib as mpl
# from matplotlib import pyplot as plt
# from scipy.interpolate import griddata
# from custom_colormaps import parula
# # from parameters import set_mpl_params
# from parameters_ls import r_range_left, r_range_right, poly_left_fit, poly_right_fit

def jacobi_for_rho(u, v, flame, max_iterations=1000, tol=1e-6):
    
    # Create x-y meshgrid
    x_array = u.columns
    y_array = u.index
    x_norm, y_norm = np.meshgrid(x_array, y_array)
    
    # dx = np.mean(np.diff(x_array))
    # dy = np.mean(np.diff(y_array))
    
    r_wall = 0.5
    tube_left_wall_index = u.columns.to_series().sub(-r_wall).abs().values.argmin() + 1
    tube_right_wall_index = u.columns.to_series().sub(r_wall).abs().values.argmin()
    tube_center_index = u.columns.to_series().sub(0).abs().values.argmin()
    
    u = u.values
    v = v.values
    
    # dudx = np.gradient(u, x_array, axis=1)
    # dvdy = np.gradient(v, y_array, axis=0)
    
    # Grid parameters
    ny, nx = u.shape
    
    # Initialize rho
    rho = np.ones((ny, nx)) * 1.2   # Starting with an initial guess
    
    # rho[0, :] = 1.2 # Bottom boundary
    # rho[0, tube_left_wall_index:tube_right_wall_index] = flame.properties.rho_u  # Bottom boundary
    
    rho = apply_bc(rho, flame, tube_left_wall_index, tube_right_wall_index)
    
    rho = upwind(rho, u, v, x_norm, y_norm)
    
    # Jacobi iteration
    for _ in range(max_iterations):
        
        rho_prev = rho.copy()
        
        rho = upwind(rho, u, v, x_norm, y_norm)
        rho = apply_bc(rho, flame, tube_left_wall_index, tube_right_wall_index)   
        
        error = np.linalg.norm(rho - rho_prev)
        
        print(error)
        
        # Convergence check
        if error < tol:
           
            rho_df = pd.DataFrame(data=rho, index=y_array, columns=x_array)
            break
        
    return rho_df

# Upwind scheme function
def upwind(rho, u, v, x_norm, y_norm):
    
    # Grid parameters
    ny, nx = u.shape
    
    rho_new = rho.copy()
    
    for j in range(1, ny-1):  # Excluding top and bottom boundaries
        for i in range(1, nx-1):  # Excluding left and right boundaries
            
            p = 1 if u[j, i] > 0 else -1
            q = 1 if v[j, i] > 0 else -1
            
            dx = -p*(x_norm[j, i-p] - x_norm[j, i])
            dy = -q*(y_norm[j-q, i] - y_norm[j, i])
            
            # Cylindrical coordinate system
            Ar = p*u[j, i]/dx
            Ax = q*v[j, i]/dy
            B = -p*rho_new[j, i-p]*u[j, i-p]*(x_norm[j, i-p]/x_norm[j, i])/dx
            C = -q*rho_new[j-q, i]*v[j-q, i]/dy
            rho_new[j, i] = -(B + C)/(Ar + Ax)
            
    return rho_new

# Deferred Correction function
def deferred_correction(rho_initial, u, v, x_norm, y_norm):
    
    # Grid parameters
    ny, nx = u.shape
    
    rho_corrected = rho_initial.copy() #np.copy(rho_initial)
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            
            dx = (x_norm[j, i+1] - x_norm[j, i-1])/2
            dy = (y_norm[j+1, i] - y_norm[j-1, i])/2
            
            # print(dx, dy)
            
            drho_udx = (1/x_norm[j, i])*(rho_initial[j, i+1]*x_norm[j, i+1]*u[j, i+1] - rho_initial[j, i-1]*x_norm[j, i-1]*u[j, i-1]) / (2*dx)
            drho_vdy = (rho_initial[j+1, i]*v[j+1, i] - rho_initial[j-1, i]*v[j-1, i]) / (2*dy)
            
            residual = drho_udx + drho_vdy
            rho_corrected[j, i] = rho_initial[j, i] - residual  # Using residual to correct the density
            
    return rho_corrected

def apply_bc(rho, flame, tube_left_wall_index, tube_right_wall_index):
    
    # Left boundary (Neumann)
    rho[:, 0] = rho[:, 1]  
    
    # Right boundary (Neumann)
    rho[:, -1] = rho[:, -2]  # Right boundary (Neumann)
    
    # Top boundary condition (Neumann)
    rho[-1, :] = rho[-2, :]
    
    # Bottom boundary condition
    rho[0, :] = 1.2 
    rho[0, tube_left_wall_index:tube_right_wall_index] = flame.properties.rho_u  # Bottom boundary
    
    return rho

def ns_comp_terms(df, D, U, Re_D, flame):
    
    """
    Parameters:
    df_piv (pandas.DataFrame): DataFrame containing velocity data.
    D (float): Diameter of the pipe [mm].
    U (float): Bulk velocity of the fluid [m/s].
    Re_D (float): Reynolds number (Re) based on the diameter D.
    """
    # Kinematic viscosity
    D = D*1e-3
    nu = U*D/Re_D
    
    u_r_col_index = 2
    u_x_col_index = 3

    R_xx_col_index = 15 # RADIAL DIRECTION
    R_xy_col_index = 16
    R_yy_col_index = 17 # AXIAL DIRECTION
    
    # Get the column headers
    headers = df.columns
    
    var_index = headers[25]
    var_column = headers[24]
    u_r_norm = pd.pivot_table(df, values=headers[u_r_col_index], index=var_index, columns=var_column)/U
    
    # Create x-y meshgrid
    r_norm_array = u_r_norm.columns
    x_norm_array = u_r_norm.index
    r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    
    # Calculate density field
    bottom_limit = 0
    top_limit = 2.2
    left_limit = -0.6
    right_limit = 0.6
    index_name = 'y_shift_norm'
    column_name = 'x_shift_norm'
    filtered_df = df[(df[index_name] > bottom_limit) & (df[index_name] < top_limit) & (df[column_name] > left_limit) & (df[column_name] < right_limit)]
    
    u_r_norm_filter = pd.pivot_table(filtered_df, values=headers[u_r_col_index], index=index_name, columns=column_name)/U
    u_x_norm_filter = pd.pivot_table(filtered_df, values=headers[u_x_col_index], index=index_name, columns=column_name)/U

    rho_df = jacobi_for_rho(u_r_norm_filter, u_x_norm_filter, flame)
    rho = rho_df.values
    
    # Create x-y meshgrid filtered
    r_norm_filter_array = u_r_norm_filter.columns
    x_norm_filter_array = u_r_norm_filter.index
    r_norm_filter, x_norm_filter = np.meshgrid(r_norm_filter_array, x_norm_filter_array)
    
    var_index = 'y_shift [m]'
    var_column = 'x_shift [m]'
    u_r = pd.pivot_table(filtered_df, values=headers[u_r_col_index], index=var_index, columns=var_column)
    u_x = pd.pivot_table(filtered_df, values=headers[u_x_col_index], index=var_index, columns=var_column)

    R_rr = pd.pivot_table(filtered_df, values=headers[R_xx_col_index], index=var_index, columns=var_column)
    R_rx = pd.pivot_table(filtered_df, values=headers[R_xy_col_index], index=var_index, columns=var_column)
    R_xx = pd.pivot_table(filtered_df, values=headers[R_yy_col_index], index=var_index, columns=var_column)
    
    # Create x-y meshgrid
    r_array = u_r.columns
    x_array = u_r.index
    r, x = np.meshgrid(r_array, x_array)
    
    dr = np.mean(np.diff(r_array))
    dx = np.mean(np.diff(x_array))
    
    # [A] Conservation of mass
    du_xdx = np.gradient(u_x, x_array, axis=0)
    
    drho_u_xdx = np.gradient(rho*u_x, x_array, axis=0)
    A1 = drho_u_xdx #np.gradient(u_x, x_array, axis=0)
    A2 = (1/r)*np.gradient(r*rho*u_r, r_array, axis=1)
    
    # [B] Conservation of momentum [Axial]
    du_xdr = np.gradient(u_x, r_array, axis=1)
    drho_u_x_u_xdx = np.gradient(rho*u_x*u_x, x_array, axis=0)
    B1 = drho_u_x_u_xdx
    
    drho_u_x_u_rdr = np.gradient(rho*u_x*u_r, r_array, axis=1)
    B2 = drho_u_x_u_rdr
    B3 = nu*(np.gradient(du_xdx, x_array, axis=0) + (1/r)*np.gradient(r*du_xdr, r_array, axis=1))
    B4 = np.gradient(rho*R_xx, x_array, axis=0)
    B5 = (1/r)*np.gradient(rho*r*R_rx, r_array, axis=1)
    # B5 = np.gradient(rho*R_rx, r_array, axis=1)
    g = 9.81 # Gravitational constant g = 9.81 m.s^-2
    B6 = (rho - 1.2)*g
    
    # [C] Conservation of momentum [Radial]
    du_rdx = np.gradient(u_r, x_array, axis=0)
    du_rdr = np.gradient(u_r, r_array, axis=1)
    drho_u_x_u_rdx = np.gradient(rho*u_x*u_r, x_array, axis=0)
    drho_u_r_u_rdr = np.gradient(rho*u_r*u_r, r_array, axis=1)
    C1 = drho_u_x_u_rdx #u_x*du_rdx
    C2 = drho_u_r_u_rdr #u_r*du_rdr
    C3 = nu*(np.gradient(du_rdx, x_array, axis=0) + (1/r)*np.gradient(r*du_rdr, r_array, axis=1) - u_r/(r**2))
    C4 = np.gradient(rho*R_rx, x_array, axis=0)
    C5 = (1/r)*np.gradient(rho*r*R_rr, r_array, axis=1)
    # C5 = np.gradient(rho*R_rr, r_array, axis=1)
    
    dpdx = -B1 - B2 - B4 - B5 #- B6
    dpdr = -C1 - C2 - C4 - C5
    
    # Integrate along x and y separately:
    p_r = np.zeros(dpdx.shape)
    p_x = np.zeros(dpdx.shape)
    
    # Integrate dp_dx across rows (i.e., along x direction)
    for i in range(1, dpdx.shape[1]):
        p_r[:, i] = p_r[:, i-1] + 0.5 * (dpdr[:, i] + dpdr[:, i-1]) * dr
    
    # Integrate dp_dy across columns (i.e., along y direction)
    for j in range(1, dpdx.shape[0]):
        p_x[j, :] = p_x[j-1, :] + 0.5 * (dpdx[j, :] + dpdx[j-1, :]) * dx
    
    # Combine them to get the pressure matrix:
    p = p_r + p_x
    
    # # If you have a reference pressure at some (i, j):
    # p_ref = 0  # Put your reference value here
    # i_ref, j_ref = 0, 0  # Adjust to your known reference location
    # p -= (p[i_ref, j_ref] - p_ref)  # Adjust entire field so that p at (i_ref, j_ref) is p_ref
    
    p_field = p
    
    norm_mass = 1 #U/D
    norm_mom = 1 #U**2/D
    
    mass_cons = [A1+A2, A1, A2]
    mom_x = [B1, B2, B3, B4, B5, dpdx, B6]
    mom_r = [C1, C2, C3, C4, C5, dpdr]
    
    mass_cons = [pd.DataFrame(data=term, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mass if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mass for term in mass_cons]
    mom_x = [pd.DataFrame(data=term, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mom for term in mom_x]
    mom_r = [pd.DataFrame(data=term, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_filter_array, columns=r_norm_filter_array)/norm_mom for term in mom_r]
    
    return mass_cons, mom_x, mom_r, p_field, dr, dx 

if __name__ == '__main__':
    
    import os
    D_in = 25.16
    U = 2.46
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    name = 'react_h0_s4000_hs_record1'
    frame_nr = 0
    segment_length_mm = 1 # units: mm
    window_size = 31 # units: pixels
    piv_folder = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

    spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
    # spydata_dir = os.path.join(main_dir, 'spydata')
    
    fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'
    
    with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
        flame = pickle.load(f)
    

    Avg_Stdev_file = os.path.join('U:\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\',
      'session_003',
      'Recording_Date=230622_Time=102848_01',
      'PIV_MP(3x16x16_75%ov_ImgCorr)',
      'Avg_Stdev',
      'Export',
      'B0001.csv')
    
    df_piv = pd.read_csv(Avg_Stdev_file)
    
    # Read variable field
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    df_piv['x_shift [mm]'] = df_piv['x [mm]'] - (D_in/2 - offset_to_wall_center)
    df_piv['y_shift [mm]'] = df_piv['y [mm]'] + offset
    
    df_piv['x_shift_norm'] = df_piv['x_shift [mm]']/D_in
    df_piv['y_shift_norm'] = df_piv['y_shift [mm]']/D_in
    
    df_piv['x_shift [m]'] = df_piv['x_shift [mm]']*1e-3
    df_piv['y_shift [m]'] = df_piv['y_shift [mm]']*1e-3
    
    Re_D = 4000
    # mass_cons, mom_x, mom_r = ns_incomp_terms(df_piv, D_in, U, Re_D)
    
    mass_cons, mom_x, mom_r = ns_comp_terms(df_piv, D_in, U, Re_D, flame)
    
    A1, A2, A12 = mass_cons
    B1, B2, B3, B4, B5 = mom_x
    C1, C2, C3, C4, C5 = mom_r 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    