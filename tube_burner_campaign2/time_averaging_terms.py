# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:22:07 2024

@author: luuka
"""
import numpy as np
import pandas as pd

def ns_incomp_terms(df, D, U, Re_D):
    
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

    # du_rdr_col_index = 5
    # du_rdx_col_index = 6
    # du_xdr_col_index = 7
    # du_xdx_col_index = 8
    
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
    
    var_index = 'y_shift [m]'
    var_column = 'x_shift [m]'
    
    u_r = pd.pivot_table(df, values=headers[u_r_col_index], index=var_index, columns=var_column)
    u_x = pd.pivot_table(df, values=headers[u_x_col_index], index=var_index, columns=var_column)

    # du_rdr = pd.pivot_table(df, values=headers[du_rdr_col_index], index=var_index, columns=var_column)
    # du_rdx = pd.pivot_table(df, values=headers[du_rdx_col_index], index=var_index, columns=var_column)
    # du_xdr = pd.pivot_table(df, values=headers[du_xdr_col_index], index=var_index, columns=var_column)
    # du_xdx = pd.pivot_table(df, values=headers[du_xdx_col_index], index=var_index, columns=var_column)
    
    R_rr = pd.pivot_table(df, values=headers[R_xx_col_index], index=var_index, columns=var_column)
    R_rx = pd.pivot_table(df, values=headers[R_xy_col_index], index=var_index, columns=var_column)
    R_xx = pd.pivot_table(df, values=headers[R_yy_col_index], index=var_index, columns=var_column)
    
    # Create x-y meshgrid
    r_array = u_r.columns
    x_array = u_r.index
    r, x = np.meshgrid(r_array, x_array)
    
    # dr = np.mean(np.diff(r, axis=1))
    # dx = np.mean(np.diff(x, axis=0))
    
    # r_array = dr
    # x_array = dx
    
    # [A] Conservation of mass
    du_xdx = np.gradient(u_x, x_array, axis=0)
    A1 = du_xdx
    A2 = (1/r)*np.gradient(r*u_r, r_array, axis=1)
    # A2 = np.gradient(u_r, r_array, axis=1)
    # A2 = u_r/r 
    
    
    # [B] Conservation of momentum [Axial]
    du_xdr = np.gradient(u_x, r_array, axis=1)
    B1 = u_x*du_xdx
    B2 = u_r*du_xdr
    B3 = nu*(np.gradient(du_xdx, x_array, axis=0) + (1/r)*np.gradient(r*du_xdr, r_array, axis=1))
    B4 = np.gradient(R_xx, x_array, axis=0)
    B5 = (1/r)*np.gradient(r*R_rx, r_array, axis=1)
    # B5 = np.gradient(R_rx, r_array, axis=1)
    # B5 = R_rx/r
    
    # [C] Conservation of momentum [Radial]
    du_rdx = np.gradient(u_r, x_array, axis=0)
    du_rdr = np.gradient(u_r, r_array, axis=1)
    C1 = u_x*du_rdx
    C2 = u_r*du_rdr
    C3 = nu*(np.gradient(du_rdx, x_array, axis=0) + (1/r)*np.gradient(r*du_rdr, r_array, axis=1) - u_r/(r**2))
    C4 = np.gradient(R_rx, x_array, axis=0)
    C5 = (1/r)*np.gradient(r*R_rr, r_array, axis=1)
    # C5 = np.gradient(F_rr, r_array, axis=1)
    # C5 = F_rr/r
    
    dpdx = -B1 - B2 - -B3 - B4 - B5 
    dpdr = -C1 - C2 - C3 - C4 - C5
    
    norm_mass = U/D
    norm_mom = (U**2)/D
    
    mass_cons = [A1+A2] # [A1+A2, A1, A2 ]
    mom_x = [B1, B2, dpdx, B3, B4, B5]
    mom_r = [C1, C2, dpdr, C3, C4, C5]
    
    mass_cons = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mass if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mass for term in mass_cons]
    mom_x = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_x]
    mom_r = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_r]
    
    return mass_cons, mom_x, mom_r, # A0, A1#, r, x

def fans_terms(df, flame):
    
    """
    Parameters:
    df_piv (pandas.DataFrame): DataFrame containing velocity data.
    D (float): Diameter of the pipe [mm].
    U (float): Bulk velocity of the fluid [m/s].
    Re_D (float): Reynolds number (Re) based on the diameter D.
    """
    
    # Kinematic viscosity
    D = flame.D_in*1e-3
    U = flame.u_bulk_measured
    Re_D = flame.Re_D
    rho_u = flame.properties.rho_u
    
    nu = U*D/Re_D
    
    # Get the column headers
    headers = df.columns
    u_r_norm = pd.pivot_table(df, values='Velocity u [m/s]', index='y_shift_norm', columns='x_shift_norm')/U
    
    # Create x-y meshgrid
    r_norm_array = u_r_norm.columns
    x_norm_array = u_r_norm.index
    r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    
    var_index = 'y_shift [m]'
    var_column = 'x_shift [m]'
    
    # rho = pd.pivot_table(df, values='rho [kg/m^3]', index=var_index, columns=var_column)
    rho = pd.pivot_table(df, values='rho [kg/m^3]', index=var_index, columns=var_column)
    
    u_r = pd.pivot_table(df, values='Velocity u [m/s]', index=var_index, columns=var_column)
    u_x = pd.pivot_table(df, values='Velocity v [m/s]', index=var_index, columns=var_column)
    
    u_r_tilde = pd.pivot_table(df, values='u_favre [m/s]', index=var_index, columns=var_column)
    u_x_tilde = pd.pivot_table(df, values='v_favre [m/s]', index=var_index, columns=var_column)

    F_rr = pd.pivot_table(df, values='rho*u_fluc_favre*u_fluc_favre', index=var_index, columns=var_column)
    F_rx = pd.pivot_table(df, values='rho*u_fluc_favre*v_fluc_favre', index=var_index, columns=var_column)
    F_xx = pd.pivot_table(df, values='rho*v_fluc_favre*v_fluc_favre', index=var_index, columns=var_column)
    
    # Create x-y meshgrid
    r_array = u_r_tilde.columns
    x_array = u_r_tilde.index
    r, x = np.meshgrid(r_array, x_array)
    
    dr = np.mean(np.diff(r_array))
    dx = np.mean(np.diff(x_array))
    
    # [A] Conservation of mass
    drho_u_xdx = np.gradient(rho*u_x_tilde, x_array, axis=0)
    A1 = drho_u_xdx #np.gradient(u_x, x_array, axis=0)
    A2 = (1/r)*np.gradient(r*rho*u_r_tilde, r_array, axis=1)
    
    # [B] Conservation of momentum [Axial]
    drho_u_x_u_xdx = np.gradient(rho*u_x_tilde*u_x_tilde, x_array, axis=0)
    drho_u_x_u_rdr = (1/r)*np.gradient(r*rho*u_x_tilde*u_r_tilde, r_array, axis=1)
    B1 = drho_u_x_u_xdx
    B2 = drho_u_x_u_rdr
    
    # Assumption of continuity equation = 0
    rho_u_x_du_xdx = rho*u_x_tilde*np.gradient(u_x_tilde, x_array, axis=0)
    rho_u_r_du_xdr = rho*u_r_tilde*np.gradient(u_x_tilde, r_array, axis=1)
    B1 = rho_u_x_du_xdx
    B2 = rho_u_r_du_xdr
    
    # B3 = np.zeros(A1.shape)
    du_xdx = np.gradient(u_x_tilde, x_array, axis=0)
    du_xdr = np.gradient(u_x_tilde, r_array, axis=1)
    B3 = nu*(np.gradient(du_xdx, x_array, axis=0) + (1/r)*np.gradient(r*du_xdr, r_array, axis=1))
    B4 = np.gradient(F_xx, x_array, axis=0)
    B5 = (1/r)*np.gradient(r*F_rx, r_array, axis=1)
    
    
    # [C] Conservation of momentum [Radial]
    drho_u_x_u_rdx = np.gradient(rho*u_x_tilde*u_r_tilde, x_array, axis=0)
    drho_u_r_u_rdr = (1/r)*np.gradient(r*rho*u_r_tilde*u_r_tilde, r_array, axis=1)
    C1 = drho_u_x_u_rdx
    C2 = drho_u_r_u_rdr
    
    # Assumption of continuity equation = 0
    rho_u_x_du_rdx = rho*u_x_tilde*np.gradient(u_r_tilde, x_array, axis=0)
    rho_u_r_du_rdr = rho*u_r_tilde*np.gradient(u_r_tilde, r_array, axis=1)
    C1 = rho_u_x_du_rdx
    C2 = rho_u_r_du_rdr
    
    # C3 = np.zeros(A1.shape)
    du_rdx = np.gradient(u_r_tilde, x_array, axis=0)
    du_rdr = np.gradient(u_r_tilde, r_array, axis=1)
    C3 = nu*(np.gradient(du_rdx, x_array, axis=0) + (1/r)*np.gradient(r*du_rdr, r_array, axis=1) - u_r/(r**2))
    C4 = np.gradient(F_rx, x_array, axis=0)
    C5 = (1/r)*np.gradient(r*F_rr, r_array, axis=1)
    
    dpdx = -B1 - B2 - B3 - B4 - B5 
    dpdr = -C1 - C2 - C3 - C4 - C5
    
    norm_mass = rho_u*U/D
    norm_mom = rho_u*(U**2)/D
    
    # mass_cons = [A1+A2, A1, A2]
    mass_cons = [A1+A2]
    
    mom_x = [B1, B2, dpdx, B4, B5]
    mom_r = [C1, C2, dpdr, C4, C5]
    
    mass_cons = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mass if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mass for term in mass_cons]
    mom_x = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_x]
    mom_r = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_r]
    
    return mass_cons, mom_x, mom_r