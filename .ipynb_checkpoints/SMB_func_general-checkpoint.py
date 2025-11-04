#%%
# -*- coding: utf-8 -*-
# # %%
                        

#%%

# import pandas as pd

# --------------------------------------------------- Functions

# ----- smb

# UNITS:
# All units must conform to:
# Time - s
# Lengths - cm^2
# Volumes - cm^3
# Masses - g
# Concentrations - g
# Volumetric flowrates - cm^3/s
# import numpy as np
# from scipy.optimize import minimize
# from scipy.integrate import solve_ivp
# from scipy import integrate
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
# from scipy.optimize import differential_evolution
# from scipy.optimize import minimize, NonlinearConstraint
# import json
# from scipy.stats import norm
# from scipy.integrate import solve_ivp
# from scipy import integrate
# import warnings
# import time


def SMB(SMB_inputs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.integrate import solve_ivp
    from scipy import integrate
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
    from scipy.optimize import differential_evolution
    from scipy.optimize import minimize, NonlinearConstraint
    import json
    from scipy.stats import norm
    from scipy.integrate import solve_ivp
    from scipy import integrate
    import warnings
    import time

    iso_type, Names, color, num_comp, nx_per_col, e, D_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all ,grouping_type, t_simulation_end = SMB_inputs[0:]

    ###################### (CALCUALTED) SECONDARY INPUTS #########################
    # print(f'parameter_sets[0][C_feed]: {parameter_sets[0]['C_feed']}')
    # Column Dimensions:
    ################################################################
    F = (1-e)/e     # Phase ratio
    t=0
    t_sets = 0

    Ncol_num = np.sum(zone_config) # Total number of columns
    
    L_total = L*Ncol_num # Total Lenght of all columns
    A_col = np.pi*0.25*d_col**2 # cm^2
    V_col = A_col * L # cm^3
    V_col_total = Ncol_num * V_col # cm^3
    A_in = np.pi * (d_in/2)**2 # cm^2
    alpha = A_in / A_col
    Z1 = zone_config[0]
    Z2 = zone_config[1]
    Z3 = zone_config[2]
    Z4 = zone_config[3]


       # Time Specs:
    ################################################################

    t_index = t_index_min*60 # s #

    # Notes:
    # - Cyclic Steady state typically happens only after 10 cycles (ref: https://doi.org/10.1205/026387603765444500)
    # - The system is not currently designed to account for periods of no external flow

    if n_num_cycles != None and t_simulation_end == None:
        
        
        n_1_cycle = t_index * Ncol_num  # s How long a single cycle takes

        total_cycle_time = n_1_cycle*n_num_cycles # s

        tend = total_cycle_time # s # Final time point in ODE solver


    elif n_num_cycles == None and t_simulation_end != None: # we specified tend instead
        
        
        n_1_cycle = t_index * Ncol_num # s

        total_cycle_time = t_simulation_end * 60 * 60  # hrs => s

        n_num_cycles = np.round(total_cycle_time/n_1_cycle)

        tend = t_simulation_end* 60 * 60 
    
    
    tend_min = tend/60
    t_span = (0, tend) # +dt)  # from t=0 to t=n
    num_of_injections = int(np.round(tend/t_index)) # number of switching periods

    # 't_start_inject_all' is a vecoter containing the times when port swithes occur for each port
    # Rows --> Different Ports
    # Cols --> Different time points
    t_start_inject_all = [[] for _ in range(Ncol_num)]  # One list for each node (including the main list)

    # Calculate start times for injections
    for k in range(num_of_injections):
        t_start_inject = k * t_index
        t_start_inject_all[0].append(t_start_inject)  # Main list
        for node in range(1, Ncol_num):
            t_start_inject_all[node].append(t_start_inject + node * 0)  # all rows in t_start_inject_all are identical

    t_schedule = t_start_inject_all[0]

    # REQUIRED FUNCTIONS:
    ################################################################

    # 1.
    # Func to Generate Indices for the columns


# Func to divide the column into nodes
    
    def cusotom_isotherm_func(cusotom_isotherm_params, c):
        """
        c => liquid concentration of ci
        q_star => solid concentration of ci @ equilibrium
        cusotom_isotherm_params = cusotom_isotherm_params_all[i] => given parameter set of component, i
        """

        # Uncomment as necessary

        #------------------- 1. Single Parameters Models
        ## Linear
        K1 = cusotom_isotherm_params[0]
        # print(f'H = {K1}')
        H = K1 # Henry's Constant
        q_star_1 = H*c

        # #------------------- 2. Two-Parameter Models
        # K1 = cusotom_isotherm_params[0]
        # K2 = cusotom_isotherm_params[1]

        # # #  Langmuir  
        # Q_max = K1
        # b = K2
        # #-------------------------------
        # q_star_2 = Q_max*b*c/(1 + b*c)
        # #-------------------------------

        #------------------- 3. Three-Parameter models 
        # K1 = cusotom_isotherm_params[0]
        # K2 = cusotom_isotherm_params[1]
        # K3 = cusotom_isotherm_params[2]

        # # Linear + Langmuir
        # H = K1
        # Q_max = K2
        # b = K3
        # ##-------------------------------
        # q_star_3 = H*c + Q_max*b*c/(1 + b*c)
        # ##-------------------------------

        return q_star_1 # [qA, ...]

    def cusotom_CUP_isotherm_func(cusotom_isotherm_params_all, c, IDX, comp_idx):
        """
        Returns  solid concentration, q_star vector for given comp_idx
        *****
        Variables:
        cusotom_isotherm_params => parameters for each component [[A's parameters], [B's parameters]]
        NOTE: This function is (currently) structured to assume A and B have 1 parameter each. 
        c => liquid concentration of c all compenents
        IDX => the first row-index in c for respective components
        comp_idx => which of the components we are currently retreiving the solid concentration, q_star for
        q_star => solid concentration of ci @ equilibrium

        """
        # Unpack the component vectors (currently just considers binary case of A and B however, could be more)
        cA = c[IDX[0]: IDX[0] + nx]
        cB = c[IDX[1]: IDX[1] + nx]
        c_i = [cA, cB]
        # Now, different isotherm Models can be built using c_i
        
        # (Uncomment as necessary)

        #------------------- 1. Linear Models

        # cusotom_isotherm_params_all has linear constants for each comp
        # # Unpack respective parameters
        K1 = cusotom_isotherm_params_all[comp_idx][0] # 1st (and only) parameter of HA or HB
        # print(f'H = {K1}')
        q_star_1 = K1*c_i[comp_idx]


        #------------------- 2. Coupled Langmuir Models
        # The parameter in the numerator is dynamic, depends on comp_idx:
        # K =  cusotom_isotherm_params_all[comp_idx][0]
        
        # # Fix the sum of parameters in the demoninator:
        # K1 = cusotom_isotherm_params_all[0][0] # 1st (and only) parameter of HA 
        # K2 = cusotom_isotherm_params_all[1][0] # 1st (and only) parameter of HB
        
        # q_star_2 = K*c_i[comp_idx]/(1+ K1*c_i[0] + K2*c_i[1])

        #------------------- 3. Combined Coupled Models
        # The parameter in the numerator is dynamic, depends on comp_idx:
        # K_lin =  cusotom_isotherm_params_all[comp_idx][0]
        
        # # Fix the sum of parameters in the demoninator:
        # K1 = cusotom_isotherm_params_all[0][0] # 1st (and only) parameter of HA 
        # K2 = cusotom_isotherm_params_all[1][0] # 1st (and only) parameter of HB
        
        # c_sum = K1 + K2
        # linear_part = K_lin*c_i[comp_idx]
        # langmuir_part = K*c_i[comp_idx]/(1+ K1*c_i[0] + K2*c_i[1])

        # q_star_3 =  linear_part + langmuir_part


        return q_star_1 # [qA, ...]

    # DOES NOT INCLUDE THE C0 NODE (BY DEFAULT)
    def set_x(L, Ncol_num,nx_col,dx):
        if nx_col == None:
            x = np.arange(0, L+dx, dx)
            nnx = len(x)
            nnx_col = int(np.round(nnx/Ncol_num))
            nx_BC = Ncol_num - 1 # Number of Nodes (mixing points/boundary conditions) in between columns

            # Indecies belonging to the mixing points between columns are stored in 'start'
            # These can be thought of as the locations of the nx_BC nodes.
            return x, dx, nnx_col,  nnx, nx_BC

        elif dx == None:
            nx = Ncol_num * nx_col
            nx_BC = Ncol_num - 1 # Number of Nodes in between columns
            x = np.linspace(0,L_total,nx)
            ddx = x[1] - x[0]

            # Indecies belonging to the mixing points between columns are stored in 'start'
            # These can be thought of as the locations of the nx_BC nodes.

            return x, ddx, nx_col, nx, nx_BC

    # 4. A func that:
    # (i) Calcualtes the internal flowrates given the external OR (ii) Visa-versa
    def set_flowrate_values(set_Q_int, set_Q_ext, Q_rec):
        if set_Q_ext is None and Q_rec is None:  # Chosen to specify internal/zone flowrates
            Q_I = set_Q_int[0]
            Q_II = set_Q_int[1]
            Q_III = set_Q_int[2]
            Q_IV = set_Q_int[3]

            QX = -(Q_I - Q_II)
            QF = Q_III - Q_II
            QR = -(Q_III - Q_IV)
            QD = -(QF + QX + QR) # OR: Q_I - Q_IV

            Q_ext = np.array([QF, QR, QD, QX]) # cm^3/s

            return Q_ext

        elif set_Q_int is None and Q_rec is not None:  # Chosen to specify external flowrates
            QF = set_Q_ext[0]
            QR = set_Q_ext[1]
            QD = set_Q_ext[2]
            QX = set_Q_ext[3]

            Q_I = Q_rec  # m^3/s
            Q_III = (QX + QF) + Q_I
            Q_IV = (QD - QX) + Q_I  # Fixed Q_II to Q_I as the variable was not defined yet
            Q_II = (QR - QX) + Q_IV
            Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])
            return Q_internal


    # 5. Function to Build Port Schedules:

    # This is done in two functions: (i) repeat_array and (ii) build_matrix_from_vector
    # (i) repeat_array
    # Summary: Creates the schedule for the 1st port, port 0, only. This is the port boadering Z2 & Z3 and always starts as a Feed port at t=0
    # (i) build_matrix_from_vector
    # Summary: Takes the output from "repeat_array" and creates schedules for all other ports.
    # The "trick" is that the states of each of the, n, ports at t=0, is equal to the first, n, states of port 0.
    # Once we know the states for each port at t=0, we form a loop that adds the next state.

    # 5.1
    def position_maker(schedule_quantity_name, F, R, D, X, Z_config):

        """

        Function that initializes the starting schedueles for a given quantitiy at all positions

        F, R, D and X are the values of the quantiity at the respective feed ports

        """
        # Initialize:
        X_j = np.zeros(Ncol_num)


        # We set each port in the appropriate position, depending on the nuber of col b/n Zones:
        # By default, Position i = 0 (entrance to col,0) is reserved for the feed node.

        # Initialize Positions:
        # Q_position is a vector whose len is = number of mixing points (ports) b/n columns

        X_j[0] = F        # FEED
        X_j[Z_config[2]] = R     # RAFFINATE
        X_j[Z_config[2] + Z_config[3]] = D    # DESORBENT
        X_j[Z_config[2] + Z_config[3]+  Z_config[0]] = X   # EXTRACT

        return X_j

    # 5.2
    def repeat_array(vector, start_time_num):
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        # start_time_num = The number of times the state changes == num of port switches == num_injections
        repeated_array = np.tile(vector, (start_time_num // len(vector) + 1))
        return repeated_array[:start_time_num]

    def initial_u_col(Zconfig, Qint):
        """
        Fun that returns the the inital state at t=0 of the volumetric
        flows in all the columns.

        """
        # First row is for col0, which is the feed to zone 3
        Zconfig_roll = np.roll(Zconfig, -2)
        Qint_roll = np.roll(Qint, -2)

        # print(Qint)
        X = np.array([])

        for i in range(len(Qint_roll)):
            X_add = np.ones(Zconfig_roll[i])*Qint_roll[i]

            # print('X_add:\n', X_add)


            X = np.append(X, X_add)
        # X = np.concatenate(X)
        # print('X:\n', X)
        return X


    def build_matrix_from_vector(vector, t_schedule):
        """
        Fun that returns the schedeule given the inital state at t=0
        vector: inital state of given quantity at t=0 at all nodes
        t_schedule: times at which port changes happen

        """
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        start_time_num = int(len(t_schedule))
        vector = np.array(vector)  # Convert the vector to a NumPy array
        n = len(vector) # number of ports/columns

        # Initialize the matrix for repeated elements, ''ALL''
        ALL = np.zeros((n, start_time_num), dtype=vector.dtype)  # Shape is (n, start_time_num)

        for i in range(start_time_num):
            # print('i:',i)
            ALL[:, i] = np.roll(vector, i)
        return ALL



    # # Uncomment as necessary depending on specification of either:
    # # (1) Internal OR (2) External flowrates :
    # # (1)
    # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])
    Q_external = set_flowrate_values(Q_internal, None, None) # Order: QF, QR, QD, QX
    QF, QR, QD, QX = Q_external[0], Q_external[1], Q_external[2], Q_external[3]
    # print('Q_external:', Q_external)

    # (2)
    # QX, QF, QR = -0.277, 0.315, -0.231  # cm^3/s
    # QD = - (QF + QX + QR)
    # Q_external = np.array([QF, QR, QD, QX])
    # Q_rec = 33.69 # cm^3/s
    # Q_internal = set_flowrate_values(None, Q_external, Q_rec) # Order: QF, QR, QD, Q

    ################################################################################################


    # Make concentration schedules for each component

    Cj_pulse_all = [[] for _ in range(num_comp)]
    for i in range(num_comp):
        Cj_position = []
        Cj_position = position_maker('Feed Conc Schedule:', parameter_sets[i]['C_feed'], 0, 0, 0, zone_config)
        Cj_pulse = build_matrix_from_vector(Cj_position,  t_schedule)
        Cj_pulse_all[i] = Cj_pulse


    Q_position = position_maker('Vol Flow Schedule:', Q_external[0], Q_external[1], Q_external[2], Q_external[3], zone_config)
    Q_pulse_all = build_matrix_from_vector(Q_position,  t_schedule)

    # Spacial Discretization:
    # Info:
    # nx --> Total Number of Nodes (EXCLUDING mixing points b/n nodes)
    # nx_col --> Number of nodes in 1 column
    # nx_BC --> Number of mixing points b/n nodes
    x, dx, nx_col, nx, nx_BC = set_x(L=L_total, Ncol_num = Ncol_num, nx_col = nx_per_col, dx = None)
    start = [i*nx_col for i in range(0,Ncol_num)] # Locations of the BC indecies
    Q_col_at_t0 = initial_u_col(zone_config, Q_internal)
    Q_col_all = build_matrix_from_vector(Q_col_at_t0, t_schedule)

    Bay_matrix = build_matrix_from_vector(np.arange(1,Ncol_num+1), t_schedule)


    # DISPLAYING INPUT INFORMATION:
    # print('---------------------------------------------------')
    # print('Number of Components:', num_comp)
    # print('---------------------------------------------------')
    # print('\nTime Specs:\n')
    # print('---------------------------------------------------')
    # print('Number of Cycles:', n_num_cycles)
    # print('Time Per Cycle:', n_1_cycle/60, "min")
    # print('Simulation Time:', tend_min/60, 'hrs')
    # print('Index Time:', t_index, 's OR', t_index/60, 'min' )
    # print('Number of Port Switches:', num_of_injections)
    # print('Injections happen at t(s) = :', t_schedule, 'seconds')
    # print('---------------------------------------------------')
    # print('\nColumn Specs:\n')
    # print('---------------------------------------------------')
    # print('Configuration:', zone_config, '[Z1,Z2,Z3,Z4]')
    # print(f"Number of Columns: {Ncol_num}")
    # print('Column Length:', L, 'cm')
    # print('Column Diameter:', d_col, 'cm')
    # print('Column Volume:', V_col, 'cm^3')

    # print("alpha:", alpha, '(alpha = A_in / A_col)')
    # print("Nodes per Column:",nx_col)
    # print("Boundary Nodes locations,x[i], i =", start)
    # print("Total Number of Nodes (nx):",nx)
    # print('---------------------------------------------------')
    # print('\nFlowrate Specs:\n')
    # print('---------------------------------------------------')
    # print("External Flowrates =", Q_external*3.6, '[F,R,D,X] L/h')
    # print("Ineternal Flowrates =", Q_internal*3.6, 'L/h')
    # print('---------------------------------------------------')
    # print('\nPort Schedules:')
    # for i in range(num_comp):
    #     print(f"Concentration Schedule:\nShape:\n {Names[i]}:\n",np.shape(Cj_pulse_all[i]),'\n', Cj_pulse_all[i], "\n")
    # print("Injection Flowrate Schedule:\nShape:",np.shape(Q_pulse_all),'\n', Q_pulse_all, "\n")
    # print("Respective Column Flowrate Schedule:\nShape:",np.shape(Q_col_all),'\n', Q_col_all, "\n")
    # print("Bay Schedule:\nShape:",np.shape(Bay_matrix),'\n', Bay_matrix, "\n")


    ###########################################################################################

    ###########################################################################################

    # Mass Transfer (MT) Models:

    def mass_transfer(kav_params, q_star, q ): # already for specific comp
        # kav_params: [kA, kB]

        # 1. Single Parameter Model
        # Unpack Parameters
        kav =  kav_params
        # MT = kav * Bm/(5 + Bm) * (q_star - q)
        MT = kav * (q_star - q)

        # 2. Two Parameter Model
        # Unpack Parameters
        # kav1 =  kav_params[0]
        # kav2 =  kav_params[1]
    
        # MT = kav1* (q_star - q) + kav2* (q_star - q)**2

        return MT

    # MT PARAMETERS
    ###########################################################################################
    # print('np.shape(parameter_sets[:]["kh"]):', np.shape(parameter_sets[3]))
    
    # print('kav_params:', kav_params)
    # print('----------------------------------------------------------------')
    ###########################################################################################

    # # FORMING THE ODES


    # Form the remaining schedule matrices that are to be searched by the funcs

    # Column velocity schedule:
    # 1. Veclocity Scheudle if there were no groups
    # u_col_all = -Q_col_all/A_col/e
    # print(f'u_col_all: {np.shape(u_col_all)}')
    # print(f'u_col_all: {u_col_all}')

    # 2. Veclocity Scheudle if there were groups
    def get_u_col_at_t0_adj(u_col_at_t0,grouping_type):
        """
        Adjusts the linear velocities of columns based on group configurations.
        
        If a column is located in a bay that is part of a group's downstream bays 
        (i.e., group[1]), then its velocity is divided by the number of such bays 
        to reflect the shared flow among them.

        Parameters:
            u_col_at_t0 (ndarray): 1D array of linear velocities for each bay at time t0.
           grouping_type (list): List of groups. Each group is a pair of lists: [upstream_bays, downstream_bays].

        Returns:
            u_col_adj_at_t0 (ndarray): Adjusted linear velocity array.
        """
        u_col_adj_at_t0 = np.copy(u_col_at_t0)

        for i in range(len(u_col_at_t0)):
            bay = i + 1  # Bay numbers start from 1
            adjusted = False

            for group in grouping_type:
                downstream_bays = group[1]

                if bay in downstream_bays:
                    divid_by = len(downstream_bays)
                    u_col_adj_at_t0[i] = u_col_at_t0[i] / divid_by
                    adjusted = True
                    # print(f"Bay {bay}: {u_col_at_t0[i]} divided by {divid_by} -> {u_col_adj_at_t0[i]}")
                    break  # Stop after finding the first matching group

            if not adjusted:
                u_col_adj_at_t0[i] = u_col_at_t0[i]
                # print(f"Bay {bay}: not in group, remains {u_col_at_t0[i]}")

        return u_col_adj_at_t0

    

    u_col_at_t0 = initial_u_col(zone_config, -Q_internal/A_col/e)
    Q_col_at_t0 = initial_u_col(zone_config, Q_internal)


    u_col_at_t0_new = get_u_col_at_t0_adj(u_col_at_t0,grouping_type)
    Q_col_at_t0_new = get_u_col_at_t0_adj(Q_col_at_t0,grouping_type)

    # print(f'u_col_at_t0:{u_col_at_t0}')
    # print(f'u_col_at_t0_new:{u_col_at_t0_new}\n\n')

    u_col_all = build_matrix_from_vector(u_col_at_t0_new, t_schedule)
    Q_col_all = build_matrix_from_vector(Q_col_at_t0_new, t_schedule)


    # print(f'u_col_all_adj: {u_col_all_adj}')
    # print(f'u_col_all: {u_col_all}')



    # Column Dispersion schedule:
    # Different matrices for each comp, because diff Pe's for each comp
    D_col_all = []
    for i in range(num_comp): # for each comp
        # D_col = -(u_col_all*L)/Pe_all[i] # constant dispersion coeff
        D_col = np.ones_like(u_col_all)*D_all[i]
        D_col_all.append(D_col)


    # print(f'Shape of u_col_all: {np.shape(u_col_all)}')
    # print(f'Shape of D_col_all: {np.shape(D_col_all)}')
    # print(f'u_col_all: {u_col_all}')
    # print(f'\nD_col_all: {D_col_all}')
    # Storage Spaces:

    coef_0 = np.zeros_like(u_col_all)
    coef_1 = np.zeros_like(u_col_all)
    coef_2 = np.zeros_like(u_col_all)

    # coef_0, coef_1, & coef_2 correspond to the coefficents of ci-1, ci & ci+1 respectively
    # These depend on u and so change with time, thus have a schedule

    # From descritization:
    coef_0_all = [[] for _ in range(num_comp)]
    coef_1_all = [[] for _ in range(num_comp)]
    coef_2_all = [[] for _ in range(num_comp)]

    coef_0_all = []
    coef_1_all = []
    coef_2_all = []

    for j in range(num_comp): # for each comp
        for i  in range(Ncol_num): # coefficients for each col
            coef_0[i,:] = ( D_col_all[j][i,:]/dx**2 ) - ( u_col_all[i,:]/dx ) # coefficeint of i-1
            coef_1[i,:] = ( u_col_all[i,:]/dx ) - (2*D_col_all[j][i,:]/(dx**2))# coefficeint of i
            coef_2[i,:] = (D_col_all[j][i,:]/(dx**2))    # coefficeint of i+1

        coef_0_all.append(coef_0)
        coef_1_all.append(coef_1)
        coef_2_all.append(coef_2)

    # print(f'coef_0_all: {np.shape(coef_0_all)}')
    # All shedules:
    # For each shceudle, rows => col idx, columns => Time idx
    # :
    # - Q_pulse_all: Injection flowrates
    # - C_pulse_all: Injection concentrations for each component
    # - Q_col_all:  Flowrates WITHIN each col
    # - u_col_all: Linear velocities WITHIN each col
    # - D_col_all: Dispersion coefficeints WITHIN each col
    # - coef_0, 1 and 2: ci, ci-1 & ci+1 ceofficients

    # print('coef_0:\n',coef_0)
    # print('coef_1:\n',coef_1)
    # print('coef_2:\n',coef_2)
    # print('\nD_col_all:\n',D_col_all)
    # print('Q_col_all:\n',Q_col_all)
    # print('A_col:\n',A_col)
    # print('u_col_all:\n',u_col_all)
    def get_column_idx_for_bae(t, bay, t_schedule, Bay_matrix):
        """
        Returns the column ID (i.e. row index of Bay_matrix) where 'bae' is located at time 't'.

        Parameters:
        - t : float
            Current time
        - bae : int
            Bay number (1-based)
        - t_schedule : list of float
            Time slice boundaries; length should be one more than number of time steps
        - Bay_matrix : 2D array-like
            Shape: (n_columns, n_time_slices)
            Rows = column IDs (0-based)
            Columns = time slices
            Entries = bay numbers (1-based)

        Returns:
        - column_idx : int
            The column ID (row index) where the bay is found at time t.
            Returns None if not found.
        """
        import numpy as np
        # Bay_matrix = np.array(Bay_matrix)

        # Find time slice index
        time_idx = None
        for j in range(len(t_schedule) - 1):
            if t_schedule[j] <= t < t_schedule[j+1]:
                time_idx = j
                # print(f'j: {j}')

                break
        else:
            if t >= t_schedule[-1]:
                time_idx = len(t_schedule) - 1

        if time_idx is not None:
            # Scan the time_idx column across all rows
            for row_idx in range(Bay_matrix.shape[0]):
                if Bay_matrix[row_idx, time_idx] == bay:
                    # print(f'row_idx: {row_idx}')
                    return row_idx  # This is the column ID

        return None





    def coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx, Bay_matrix,grouping_type): # note that c_length must include nx_BC

        # Define the functions that call the appropriate schedule matrices:
        # Because all scheudels are of the same from, only one function is required
        # Calling volumetric flows:
        get_X = lambda t, X_schedule, col_idx: next((X_schedule[col_idx][j] for j in range(len(X_schedule[col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)
        get_C = lambda t, C_schedule, col_idx, comp_idx: next((C_schedule[comp_idx][col_idx][j] for j in range(len(C_schedule[comp_idx][col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)

        # tHE MAIN DIFFERENCE BETWEEEN get_X and get_C is that get_C considers teh component level 

        def small_col_matix(nx_col, col_idx):
        # Initialize small_col_coeff ('small' = for 1 col)

            small_col_coeff = np.zeros((int(nx_col),int(nx_col))) #(5,5)

            # Where the 1st (0th) row and col are for c1
            # get_C(t, coef_0_all, k, comp_idx)
            # small_col_coeff[0,0], small_col_coeff[0,1] = get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[0,0], small_col_coeff[0,1] = get_C(t, coef_1_all,col_idx, comp_idx), get_C(t,coef_2_all,col_idx, comp_idx)
            # for c2:
            # small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_X(t,coef_0,col_idx), get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_C(t,coef_0_all,col_idx, comp_idx), get_C(t, coef_1_all, col_idx, comp_idx), get_C(t, coef_2_all,col_idx, comp_idx)

            for i in range(2,nx_col): # from row i=2 onwards
                # np.roll the row entries from the previous row, for all the next rows
                new_row = np.roll(small_col_coeff[i-1,:],1)
                small_col_coeff[i:] = new_row

            small_col_coeff[-1,0] = 0
            small_col_coeff[-1,-1] = small_col_coeff[-1,-1] +  get_C(t,coef_2_all,col_idx, comp_idx) # coef_1 + coef_2 account for rolling boundary


            return small_col_coeff

        # Initialize:
        component_coeff_matrix = np.zeros((nx,nx)) # ''large'' = for all cols # (20, 20)

        # Add the cols
        for col_idx in range(Ncol_num):

            srt = col_idx*nx_col
            end = (col_idx+1)*nx_col

            component_coeff_matrix[srt:end, srt:end] = small_col_matix(nx_col,col_idx)
        # print('np.shape(larger_coeff_matrix)\n',np.shape(larger_coeff_matrix))

        # vector_add: vector that applies the boundary conditions to each boundary node
        def vector_add(nx, c, start, comp_idx, Bay_matrix,grouping_type):
            vec_add = np.zeros(nx)
            c_BC = np.zeros(Ncol_num)
            # print(f'start: {start}')
            # Indeceis for the boundary nodes are stored in "start"
            # Each boundary node is affected by the form:
            # c_BC = V1 * C_IN - V2 * c[i] + V3 * c[i+1]

            # R1 = ((beta * alpha) / gamma)
            # R2 = ((2 * Da / (u * dx)) / gamma)
            # R3 = ((Da / (2 * u * dx)) / gamma)

            # Where:
            # C_IN is the weighted conc exiting the port facing the column entrance.
            # alpha , bata and gamma depend on the column vecolity and are thus time dependant
            # Instead of forming schedules for alpha , bata and gamma, we calculate them in-line
            # for i in range(len(start)):

            for i, strt in enumerate(start):
                bay = get_X(t, Bay_matrix, i)
                found_group = False
                # print(f'time: {t/60} min')
                # print(f'col: {i}')
                # print(f'bay: {bay}\n\n')


                for sz, group in enumerate(grouping_type):
                    if bay in group[1]:
                        found_group = True
                        feed_col_bays = group[0]

                        # example for single bay case
                        if len(feed_col_bays) == 1:
                            # print(f'better to come in here!')
                            # print(f'time: {t/60} min')
                            # print(f'feed_col_bays[0]: {feed_col_bays[0]}')
                            col_idx_feed = get_column_idx_for_bae(t, feed_col_bays[0], t_schedule, Bay_matrix)
                            # print(f'col_idx_feed: {col_idx_feed}\n\n')
                            # print(f'time: {t/60} min')
                            # print(f'bay: {bay}')
                            # print(f'col_idx_feed: {col_idx_feed}\n\n')


                            Q_1 = get_X(t, Q_col_all, col_idx_feed)
                            c_1 = c[((col_idx_feed + 1) * nx_per_col) - 1]

                            Q_inj = get_X(t, Q_pulse_all, (col_idx_feed + 1)%Ncol_num)
                            c_inj = get_C(t, Cj_pulse_all, (col_idx_feed + 1)%Ncol_num, comp_idx)

                            Q_previous = [Q_1, Q_inj]
                            c_sum = [c_1, c_inj]

                            
                            Q_sum = sum(Q_previous)

                            if Q_inj > 0:
                                c_sum = [(q / Q_sum) * c for q, c in zip(Q_previous, c_sum)]
                                C_IN = sum(c_sum)
                            else:
                                C_IN = c_1
                            
                           

                            u = get_X(t, u_col_all, i)
                            
                            # n_current_sub_zone = int(len(group[1]))

                        
                            # print(f'time: {t/60} min')
                            # print(f'coming from: col: {col_idx_feed} in bay: {feed_col_bays[0]}')
                            # print(f'going to: col: {i} in bay: {bay}')
                            # print(f'Q_previous: {Q_previous} col {col_idx_feed} [flowrate, desorbent_flow]')
                            # print(f'c_sum: {c_sum}\n\n')
                            # print(f'------------------')
                            # print(f'n_current_sub_zone: {len(group[1])}')
                            # print(f'Q before split: {Q_sum} cm3/s')
                            # print(f'C_IN: {C_IN}')
                            # print(f'Q (in col {i}) after split: {-1*u*A_col*0.56} cm3/s\n\n')

                        elif len(feed_col_bays) > 1:  # if that group has multiple feed bays
                            # print(f'please dont come in here!')
                            c_sum = []
                            Q_previous = []
                            # print(f'feed_col_bays: {feed_col_bays}')
                            for fb, feed_bay in enumerate(feed_col_bays):  # for each feed bay
                                #What is the number (ID) of the feed column of interest?:
                                col_idx_feed = get_column_idx_for_bae(t, feed_bay, t_schedule, Bay_matrix)  # get the column index using bae
                                # Use the column label to get the flowrate of the col
                                # print(f'time: {t/60} min')
                                # print(f'bay:{bay}')
                                # print(f'col_idx_feed: {col_idx_feed}\n\n')
                                Q_1 = get_X(t, Q_col_all, col_idx_feed)
                                # Again use the col_idx_feed; to get the concentration out of that column
                                c_1 = c[(col_idx_feed + 1) * nx_per_col - 1]

                                # Store this information
                                Q_previous.append(Q_1)
                                c_sum.append(c_1)

                                # Is there any flow event (F,R,X,D) happening after the 3 columns in the preceding feed bay
                                # if so, that information will be in the schedule matrix at the row corresponding to
                                # the col that is (one) position ahead of the last in the feed_bay set

                                if fb == len(feed_col_bays)-1:
                                    col_idx_ahead = (col_idx_feed + 1) % Ncol_num 
                                    # print(f'inj schedule from: {col_idx_ahead}')
                                    Q_inj = get_X(t, Q_pulse_all, col_idx_ahead)
                                    c_inj = get_C(t, Cj_pulse_all, col_idx_ahead, comp_idx)
                                    
                                    # Store
                                    if Q_inj > 0:
                                        Q_previous.append(Q_inj)
                                        c_sum.append(c_inj)

                            # print(f'time: {t/60} min')
                            # print(f'coming from: col: {col_idx_feed} in bay:{feed_bay}')
                            # print(f'going to: col: {i} in bay: {bay}')
                            # print(f'Q_previous: {Q_previous} [col {col_idx_feed} flowrate, desorbent]')
                            # print(f'c_sum: {c_sum} [conc out upstream cols, conc from #flow_event]')
                            # print(f'--------------------')

                            # print(f'Q_previous: {Q_previous}')

                            Q_sum = np.sum(Q_previous)

                             
                            weights = Q_previous / Q_sum
                            c_add_them_up = weights * c_sum
                            # print(f'Q_sum: {Q_sum} cm3/s')
                            # print(f'weights: {weights}')
                            # print(f'c_sum: {c_sum}, [perv cols, cinj]')   
                            # print(f'c_add_them_up: {c_add_them_up}')
                            # print(f'--------------------\n\n')

                            C_IN = np.sum(c_add_them_up)
                            


                            n_current_sub_zone = len(group[1])

                            u = get_X(t, u_col_all, i)

                            # print(f'Q before manifold: {Q_sum} cm3/s')
                            # print(f'Q (in col {i}) after manifold: {-1*u*A_col*0.56} cm3/s\n\n')

                        break  # done once group match is found

                if found_group ==  False:
      
                    Q_1 = get_X(t, Q_col_all, i-1)
                    Q_2 = get_X(t, Q_pulse_all, i)
                    W1 = Q_1 / (Q_1 + Q_2)
                    W2 = Q_2 / (Q_1 + Q_2)

                    u = get_X(t, u_col_all, i)
                    c_injection = get_C(t, Cj_pulse_all, i, comp_idx)


                    if Q_2 > 0:
                        C_IN = W1 * c[strt - 1] + W2 * c_injection
                    else:
                        C_IN = c[strt - 1]

                    # print(f'time: {t/60} min')
                    # print(f'From column: {i-1} in bay: {get_X(t, Bay_matrix, i-1)}')
                    # print(f'To column: {i}, in bay {bay}')
                    # print(f'[{Q_1}, {Q_2}]: [col {i} flowrate, feed/raff]\n\n') # get the column index using bae


                # Calcualte alpha, bata and gamma:
                
                Da = get_C(t, D_col_all, i, comp_idx)
                beta = 1 / alpha
                gamma = 1 - 3 * Da / (2 * u * dx)

                ##
                # R1 = ((beta * alpha) / gamma)
                R1 = ((beta *alpha) / gamma)
                R2 = ((2 * Da / (u * dx)) / gamma)
                R3 = ((Da / (2 * u * dx)) / gamma)
                ##

                # Calcualte the BC effects:
                
                c_BC[i] = R1 * C_IN - R2 * c[strt] + R3 * c[strt+1] # the boundary concentration for that node
                
            # print('c_BC:\n', c_BC)

            for k in range(len(c_BC)):
                # vec_add[start[k]]  = get_X(t,coef_0,k)*c_BC[k]
                # print(vec_add)
                vec_add[start[k]]  = get_C(t, coef_0_all, k, comp_idx)*c_BC[k]
                # print(f'vec_add: {vec_add}')

            return vec_add
            # print('np.shape(vect_add)\n',np.shape(vec_add(nx, c, start)))

        return component_coeff_matrix, vector_add(nx, c, start, comp_idx, Bay_matrix,grouping_type)


    
    def matrix_builder(M):
        """
        M => Set of matricies describing the dynamics in all columns of each comp [M_A, M_B]
        -------------------------------------
        This func takes M and adds it to M0.
        """
        # M = Matrix to add (small)

        n = len(M) # number of components
        rx = len(M[0][0,:])  # all members of M are square and equal in size, we just want the col num
        nn = int(n * rx)
        # print(f'nn: {nn}')
        
        M0 = np.zeros((nn, nn))
        # M0 = Initial state of the matrix to be added to


        positon_1 = 0
        positon_2 = rx
        positon_3 = 2*rx

        M0[positon_1:positon_2, positon_1:positon_2] = M[0]

        M0[positon_2:positon_3, positon_2:positon_3] = M[1]

        return M0


    # ###########################################################################################

    # # mod1: UNCOUPLED ISOTHERM:
    # # Profiles for each component can be solved independently
    
    # ###########################################################################################
    def mod1(t, v, comp_idx, Q_pulse_all):
        # call.append("call")
        # print(len(call))
        c = v[:nx]
        q = v[nx:]

        # Initialize the derivatives
        dc_dt = np.zeros_like(c)
        dq_dt = np.zeros_like(q)
        # print('v size\n',np.shape(v))

        # Isotherm:
        #########################################################################
        isotherm = cusotom_isotherm_func(cusotom_isotherm_params_all[comp_idx],c)
        # isotherm = iso_lin(theta_lin[comp_idx], c)
        #isotherm = iso_langmuir(theta_lang[comp_idx], c, comp_idx)
        #isotherm = iso_freundlich(theta_fre, c)


        # Mass Transfer:
        #########################################################################
        # print('isotherm size\n',np.shape(isotherm))
        MT = mass_transfer(kav_params_all[comp_idx], isotherm, q)
        #print('MT:\n', MT)

        coeff_matrix, vec_add = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx, Bay_matrix,grouping_type)
        # print('coeff_matrix:\n',coeff_matrix)
        # print('vec_add:\n',vec_add)
        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT

        return np.concatenate([dc_dt, dq_dt])

    # ##################################################################################


    def mod2(t, v):

        # where, v = [c, q]
        c = v[:num_comp*nx] # c = [cA, cB] | cA = c[:nx], cB = c[nx:]
        q = v[num_comp*nx:] # q = [qA, qB]| qA = q[:nx], qB = q[nx:]

        # Craate Lables so that we know the component assignement in the c vecotor:
        A, B = 0*nx, 1*nx # Assume Binary 2*nx, 3*nx, 4*nx, 5*nx
        IDX = [A, B]

        # Thus to refer to the liquid concentration of the i = nth row of component B: c[C + n]
        # Or the the solid concentration 10th row of component B: q[B + 10]
        # Or to refer to all A's OR B's liquid concentrations: c[A + 0: A + nx] OR c[B + 0: B + nx]


        # Initialize the derivatives
        dc_dt = np.zeros_like(c)
        dq_dt = np.zeros_like(q)


        # coeff_matrix, vec_add = coeff_matrix_builder_CUP(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, IDX)
        # print('coeff_matrix:\n',coeff_matrix)
        # print('vec_add:\n',vec_add)
        coeff_matrix_A, vec_add_A = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[0:nx], nx_col, 0, Bay_matrix,grouping_type)
        coeff_matrix_B, vec_add_B = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[nx:2*nx], nx_col, 1, Bay_matrix,grouping_type)

        coeff_matrix  = matrix_builder([coeff_matrix_A, coeff_matrix_B])
        vec_add = np.concatenate([vec_add_A, vec_add_B])

        ####################### Building MT Terms ####################################################################

        # Initialize

        MT = np.zeros(len(c)) # column vector: MT kinetcis for each comp: MT = [MT_A MT_B]

        for comp_idx in range(num_comp): # for each component
            
            

            ######################(ii) Isotherm ####################################################################

            # Comment as necessary for required isotherm:
            # isotherm = iso_bi_langmuir(theta_blang[comp_idx], c, IDX, comp_idx)
            # isotherm = iso_cup_langmuir(theta_cup_lang, c, IDX, comp_idx)
            isotherm = cusotom_CUP_isotherm_func(cusotom_isotherm_params_all, c, IDX, comp_idx)
            # print('qstar:\n', isotherm.shape)
            ################### (ii) MT ##########################################################
            MT_comp = mass_transfer(kav_params_all[comp_idx], isotherm, q[IDX[comp_idx]: IDX[comp_idx] + nx ])
            MT[IDX[comp_idx]: IDX[comp_idx] + nx ] = MT_comp

            # [MT_A, MT_B, . . . ] KINETICS FOR EACH COMP

        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT

        return np.concatenate([dc_dt, dq_dt])

    # ##################################################################################

    # SOLVING THE ODES
    # creat storage spaces:
    y_matrices = []

    t_sets = []
    t_lengths = []

    c_IN_values_all = []
    F_in_values_all = []
    call = []

    # print('----------------------------------------------------------------')
    # print("\n\nSolving the ODEs. . . .")



    if iso_type == "UNC": # UNCOUPLED - solve 1 comp at a time
        for comp_idx in range(num_comp): # for each component
            print(f'Solving comp {comp_idx}. . . .')
            v0 = np.zeros(Ncol_num* (nx_col + nx_col)) #  for both c and q
            solution = solve_ivp(mod1, t_span, v0, args=(comp_idx , Q_pulse_all))
            y_solution, t = solution.y, solution.t
            y_matrices.append(y_solution)
            t_sets.append(t)
            t_lengths.append(len(t))
            # print(f'y_matrices[{i}]', y_matrices[i].shape)


    # Assuming only a binary coupled system
    if iso_type == "CUP": # COUPLED - solve
            # nx = nx_col*num_comp
            v0 = np.zeros(num_comp*(nx)*2) # for c and 2, for each comp
            solution = solve_ivp(mod2, t_span, v0)
            y_solution, t = solution.y, solution.t
            # Convert y_solution from: [cA, cB, qA, qB] ,  TO: [[cA, qA ], [cB, qB]]
            # Write a function to do that

            def reshape_ysol(x, nx, num_comp):
                # Initialize a list to store the reshaped components
                reshaped_list = []

                # Iterate over the number of components
                for i in range(num_comp):
                    # Extract cX and qX submatrices for the i-th component
                    cX = x[i*nx:(i+1)*nx, :]      # Extract cX submatrix
                    qX = x[i*nx + num_comp*nx : (i+1)*nx + num_comp*nx, :]       # Extract qX submatrix
                    concat = np.concatenate([cX, qX])
                    # print('i:', i)
                    # print('cX:\n',cX)
                    # print('qX:\n',qX)
                    # Append the reshaped pair [cX, qX] to the list
                    reshaped_list.append(concat)

                # Convert the list to a NumPy array
                result = np.array(reshaped_list)

                return result

            y_matrices = reshape_ysol(y_solution, nx, num_comp)
            # print('len(t_sets) = ', len(t_sets[0]))
            # print('len(t) = ', len(t))

    # print('----------------------------------------------------------------')
    # print('\nSolution Size:')
    # for i in range(num_comp):
    #     print(f'y_matrices[{i}]', y_matrices[i].shape)
    # print('----------------------------------------------------------------')
    # print('----------------------------------------------------------------')




    # ###########################################################################################

    # VISUALIZATION

    ###########################################################################################




    # MASS BALANCE AND PURITY CURVES
    ###########################################################################################

    def find_indices(t_ode_times, t_schedule):
        """
        t_schedule -> vector of times when (events) port switches happen e.g. at [0,5,10] seconds
        t_ode_times -> vector of times from ODE

        We want to know where in t_ode_times, t_schedule occures
        These iwll be stored as indecies in t_idx
        Returns:np.ndarray: An array of indices in t_ode_times corresponding to each value in t_schedule.
        """
        t_idx = np.searchsorted(t_ode_times, t_schedule)
        t_idx = np.append(t_idx, len(t_ode_times))

        return t_idx

    # Fucntion to find the values of scheduled quantities
    # at all t_ode_times points

    def get_all_values(X, t_ode_times, t_schedule_times, Name):

        """
        X -> Matrix of Quantity at each schedule time. e.g:
        At t_schedule_times = [0,5,10] seconds feed:
        a concentraction of, X = [1,2,3] g/m^3

        """
        # Get index times
        t_idx = find_indices(t_ode_times, t_schedule_times)
        # print('t_idx:\n', t_idx)

        # Initialize:
        nrows = np.shape(X)[0]
        # print('nrows', nrows)

        values = np.zeros((nrows, len(t_ode_times))) # same num of rows, we just extend the times
        # print('np.shape(values):\n',np.shape(values))

        # Modify:
        k = 0

        for i in range(len(t_idx)-1): # during each schedule interval
            j = i%nrows

            # # k is a counter that pushes the row index to the RHS every time it loops back up
            # if j == 0 and i == 0:
            #     pass
            # elif j == 0:
            #     k += 1

            # print('j',j)

            X_new = np.tile(X[:,j], (len(t_ode_times[t_idx[i]:t_idx[i+1]]), 1))

            values[:, t_idx[i]:t_idx[i+1]] = X_new.T # apply appropriate quantity value at approprite time intrval

        # Visualize:
        # # Table
        # print(Name," Values.shape:\n", np.shape(values))
        # print(Name," Values:\n", values)
        # # Plot
        # plt.plot(t_ode_times, values)
        # plt.xlabel('Time (s)')
        # plt.ylabel('X')
        # plt.show()

        return values, t_idx


    # Function that adds row slices from a matrix M into one vector
    def get_X_row(M, row_start, jump, width):

        """
        M  => Matrix whos rows are to be searched and sliced
        row_start => Starting row - the row that the 1st slice comes from
        jump => How far the row index jumps to caputre the next slice
        width => the widths of each slice e.g. slice 1 is M[row, width[0]:width[1]]

        """
        # Quick look at the inpiuts
        # print('M.shape:\n', M.shape)
        # print('width:', width)

        # Initialize
        values = []
        values_avg = []
        nrows = M.shape[0]

        for i in range(len(width)-1):


            j = i%nrows
            # print('i', i)
            # print('j', j)
            t_start = int(width[i])
            tend = int(width[i+1])

            kk = (row_start+j*jump)%nrows

            MM = M[kk, t_start:tend]

            if i == 0:
                values_avg.append(MM[0])
                # print(f'values:\n{values} ')
                # print(f'values_avg:\n{values_avg} ')
            
            MM_avg = np.average(MM)

            values.extend(MM)
            values_avg.append(MM_avg)

        return [values, values_avg]



    #  MASS INTO SYSMEM

    # - Only the feed port allows material to FLOW IN
    ###########################################################################################

    # Convert the Feed concentration schedule to show feed conc for all time
    # Do this for each component
    # C_feed_all = [[] for _ in range(num_comp)]

    row_start = 0 # iniital feed port row in schedule matrix

    row_start_matrix_raff = nx_col*Z3
    row_start_matrix_ext = (nx_col*(Z3 + Z4 + Z1))

    row_start_schedule_raff = row_start+Z3
    row_start_schedule_ext = row_start+Z3+Z4+Z1

    jump_schedule = 1
    jump_matrix = nx_col


    def feed_profile(t_odes, Cj_pulse_all, t_schedule, row_start, jump):

        """"
        Function that returns :
        (i) The total mass fed of each component
        (ii) Vector of feed conc profiles of each component
        """

        # Storage Locations:
        C_feed_all = []
        t_idx_all = []
        m_feed = []

        C_feed = [[] for _ in range(num_comp)]
        C_feed_avg = []

        for i in range(num_comp):

            if iso_type == 'UNC':

                C, t_idx = get_all_values(Cj_pulse_all[i], t_odes[i], t_schedule, 'Concentration')
                t_idx_all.append(t_idx)

            elif iso_type == 'CUP':
                C, t_idx_all = get_all_values(Cj_pulse_all[i], t_odes, t_schedule, 'Concentration')

            C_feed_all.append(C)

            # print('t_idx_all:\n', t_idx_all )

        for i in range(num_comp):
            if iso_type == 'UNC':
                C_feed[i] = get_X_row( C_feed_all[i], row_start, jump, t_idx_all[i])[0] # g/cm^3
            elif iso_type == 'CUP':
                C_feed[i] = get_X_row( C_feed_all[i], row_start, jump, t_idx_all)[0] # g/cm^3
                # C_feed_avg[i] = get_X_row( C_feed_all[i], row_start, jump, t_idx_all)[1] # g/cm^3
        # print('C_feed[0]:',C_feed[0])

        for i in range(num_comp):
            F_feed = np.array([C_feed[i]]) * QF # (g/cm^3 * cm^3/s)  =>  g/s | mass flow into col (for comp, i)
            F_feed = np.array([F_feed]) # g/s

            if iso_type == 'UNC':
                m_feed_add = integrate.simpson(F_feed, x=t_odes[i]) # g
            if iso_type == 'CUP':
                m_feed_add = integrate.simpson(F_feed, x=t_odes) # g

            m_feed.append(m_feed_add)

        m_feed = np.concatenate(m_feed) # g
        # print(f'm_feed: {m_feed} g')

        return C_feed, m_feed, t_idx_all

    if iso_type == 'UNC':
        C_feed, m_feed, t_idx_all = feed_profile(t_sets, Cj_pulse_all, t_schedule, row_start, jump_schedule)
    elif iso_type == 'CUP':
        C_feed, m_feed, t_idx_all = feed_profile(t, Cj_pulse_all, t_schedule, row_start, jump_schedule)




    def clean_matrix(X, X_feed):
        """
        Replace values in X with 0 if X[i,j]/X_feed < 0.001.

        Parameters:
        - X: np.ndarray of shape (m, n)
        - X_feed: float

        Returns:
        - np.ndarray: cleaned matrix
        """
        ratio = X / X_feed
        cleaned_X = np.where(ratio < 0.001, 0, X)
        return cleaned_X



    def prod_profile(t_odes, y_odes, t_schedule, row_start_matrix, jump_matrix, t_idx_all, row_start_schedule):

        """"
        Function that can be used to return:

        (i) The total mass exited at the Raffinate or Extract ports of each component
        (ii) Vector of Raffinate or Extract mass flow profiles of each component
        (iii) Vector of Raffinate or Extract vol flow profiles of each component

        P = Product either raff or ext
        """
        ######## Storages for the Raffinate #########


        C_P1 = []
        C_P2 = []
        # Storage
        P_mprofile_AVG = []
        P_cprofile_AVG = []
        C_P1_AVG = []
        C_P2_AVG = []

        Q_all_flows = [] # Flowrates expirenced by each component
        m_out_P = np.zeros(num_comp)

        P_vflows_1 = []
        P_mflows_1 = []
        m_P_1 = []

        P_vflows_2 = []
        P_mflows_2 = []
        m_P_2 = []
        t_idx_all_Q = []

        P_mprofile = []
        P_cprofile = []        
        P_mprofile_smooth = []
        P_cprofile_smooth = []

        P_vflow = [[] for _ in range(num_comp)]


        # First get the values of the volumetric flowrates in each column @ all times from 0 to t_ode[-1]
        # Because UNC has different time intervals for each component, we need to get the time vecoters for each component
        if iso_type == 'UNC':
            for i in range(num_comp): # for each component
                Q_all_flows_add, b = get_all_values(Q_col_all, t_odes[i], t_schedule, 'Column Flowrates')
                # print('Q_all_flows_add:\n', Q_all_flows_add)
                Q_all_flows.append(Q_all_flows_add) # cm^3/s
                t_idx_all_Q.append(b)



        elif iso_type == 'CUP':
            Q_all_flows, t_idx_all_Q = get_all_values(Q_col_all, t_odes, t_schedule, 'Column Flowrates')
            # print('Q_all_flows:\n', Q_all_flows)
            # print('Q_all_flows:\n', np.shape(Q_all_flows))
            # print(f't_idx_all_Q: {np.shape(t_idx_all_Q)}')


        for i in range(num_comp):# for each component

            if iso_type == 'UNC':
                # Search the ODE matrix
                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all_Q[i])[0]) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all_Q[i])[0])
                
                C_R1_add_AVG = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all_Q[i])[1]) # exclude q
                C_R2_add_AVG = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all_Q[i])[1])
                
                
                # # Search the Flowrate Schedule
                # P_vflows_1_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule-1, jump_schedule, t_idx_all_Q[i]))
                # P_vflows_2_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule, jump_schedule, t_idx_all_Q[i]))

            elif iso_type == 'CUP':

                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all)[0]) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all)[0])
                
                C_R1_add_AVG = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all)[1]) # exclude q
                C_R2_add_AVG = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all)[1])


                #Remove funcky concentration vals
                C_R1_add_AVG = clean_matrix(C_R1_add_AVG, parameter_sets[i]['C_feed'])
                # print(f'C_R1_add_AVG:\n{C_R1_add_AVG}')
                # print(f'C_R1_add_AVG negative: {np.any(C_R1_add_AVG < 0)}')
                # print(f'C_R1_add_AVG: {C_R1_add_AVG}\n\n')
             
                
                # P_vflows_1_add = np.array(get_X_row(Q_all_flows, row_start_schedule-1, jump_schedule, t_idx_all_Q))
                # P_vflows_2_add = np.array(get_X_row(Q_all_flows, row_start_schedule, jump_schedule, t_idx_all_Q))


            # Raffinate Massflow Curves
            # print('C_R1_add.type():\n',type(C_R1_add))
            # print('np.shape(C_R1_add):\n', np.shape(C_R1_add))

            # print('P_vflows_1_add.type():\n',type(P_vflows_1_add))
            # print('np.shape(P_vflows_1_add):\n', np.shape(P_vflows_1_add))

            # Assuming only conc change accross port when (i) adding feed or (ii) desorbent
            C_R1_add = C_R2_add # ??
            # P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s
            # P_mflows_2_add = C_R2_add * P_vflows_2_add  # g/s
            
            if row_start_matrix == row_start_matrix_raff:
                P_vflows_1_add = -QR*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s
                P_mflows_1_add_AVG = C_R1_add_AVG * -QR  # (g/cm^3 * cm^3/s)  =>  g/s

            elif row_start_matrix == row_start_matrix_ext:
                print(f'QR:{QR}')
                P_vflows_1_add = -QX*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s
                P_mflows_1_add_AVG = C_R1_add_AVG * -QX




            # Flow profiles:

            # Volumetric cm^3/s
            P_vflow[i] = P_vflows_1_add #- P_vflows_2_add # cm^3

            # Integrate


            if iso_type == 'UNC':
                m_P_add_1 = integrate.simpson(P_mflows_1, x=t_odes[i]) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes[i]) # g

            if iso_type == 'CUP':
                m_P_add_1 = integrate.simpson(P_mflows_1_add, x=t_odes) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes) # g

            
            
            #     
            # Concentration
            P_cprofile.append(C_R1_add) # g/mL
            # Average Concentration Profile
            # print(f'Before adding comp: {i}')
            # print(f'np.shape(P_cprofile_AVG): {np.shape(P_cprofile_AVG)}')
            P_cprofile_AVG.append(C_R1_add_AVG) # g/mL

            # print(f'AFTER adding comp: {i}')
            # print(f'np.shape(P_cprofile_AVG): {np.shape(P_cprofile_AVG)}')

            # Mass g/s
            P_mprofile.append(P_mflows_1_add) #- P_mflows_2_add) # g/s
            P_mprofile_AVG.append(P_mflows_1_add_AVG)


            C_P1.append(C_R1_add)  # Concentration Profiles
            C_P2.append(C_R2_add)

                    

            P_vflows_1.append(P_vflows_1_add)
            # P_vflows_2.append(P_vflows_2_add)

            P_mflows_1.append(P_mflows_1_add)
            # P_mflows_2.append(P_mflows_2_add)

            m_P_1.append(m_P_add_1) # masses of each component
            # m_P_2.append(m_P_add_2) # masses of each component

        # Final Mass Exited
        # Mass out from P and ext
        for i in range(num_comp):
            m_out_P_add = m_P_1[i] #- m_P_2[i]
            # print(f'i:{i}')
            # print(f'm_out_P_add = m_P_1[i] - m_P_2[i]: { m_P_1[i]} - {m_P_2[i]}')
            m_out_P[i] = m_out_P_add # [A, B] g
        
        # print(f'np.shape(P_cprofile_AVG): {np.shape(P_cprofile_AVG)}')
        # print(f'np.shape(P_cprofile): {np.shape(P_cprofile)}')


        return P_cprofile, P_cprofile_AVG, P_mprofile, P_mprofile_AVG, m_out_P, P_vflow



    # Evaluating the product flowrates
    #######################################################
    # raff_mprofile, m_out_raff, raff_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_R1, row_start_R2, jump_matrix, t_idx_all, row_start+Z3)
    # ext_mprofile, m_out_ext, ext_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_X1, row_start_X2, jump_matrix, t_idx_all, row_start+Z3+Z4+Z1)
    if iso_type == 'UNC':
        raff_cprofile, raff_avg_cprofile, raff_mprofile, raff_avg_mprofile, m_out_raff, raff_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_matrix_raff, jump_matrix, t_idx_all, row_start_schedule_raff)
        ext_cprofile, ext_avg_cprofile, ext_mprofile, ext_avg_mprofile, m_out_ext, ext_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_matrix_ext, jump_matrix, t_idx_all, row_start_schedule_ext)
    elif iso_type == 'CUP':
        raff_cprofile, raff_avg_cprofile, raff_mprofile, raff_avg_mprofile, m_out_raff, raff_vflow = prod_profile(t, y_matrices, t_schedule, row_start_matrix_raff, jump_matrix, t_idx_all, row_start_schedule_raff)
        ext_cprofile, ext_avg_cprofile, ext_mprofile, ext_avg_mprofile, m_out_ext, ext_vflow = prod_profile(t, y_matrices, t_schedule, row_start_matrix_ext, jump_matrix, t_idx_all, row_start_schedule_ext)
    #######################################################
    # print(f'raff_vflow: {raff_vflow}')
    # print(f'np.shape(raff_vflow): {np.shape(raff_vflow[0])}')
    # print(f'ext_vflow: {ext_vflow}')
    # print(f'np.shape(ext_vflow): {np.shape(ext_vflow[0])}')









    # MASS BALANCE:
    #######################################################

    # Error = Expected Accumulation - Model Accumulation

    #######################################################

    # Expected Accumulation = Mass In - Mass Out
    # Model Accumulation = Integral in all col at tend (how much is left in col at end of sim)


    # Calculate Expected Accumulation
    #######################################################
    m_out = np.array([m_out_raff]) + np.array([m_out_ext]) # g
    m_out = np.concatenate(m_out)
    m_in = np.concatenate(m_feed) # g
    # ------------------------------------------
    Expected_Acc = m_in - m_out # g
    # ------------------------------------------


    # Calculate Model Accumulation
    #######################################################
    def model_acc(y_ode, V_col_total, e, num_comp):
        """
        Func to integrate the concentration profiles at tend and estimate the amount
        of solute left on the solid and liquid phases
        """
        mass_l = np.zeros(num_comp)
        mass_r = np.zeros(num_comp)

        for i in range(num_comp): # for each component

            V_l = e * V_col_total # Liquid Volume cm^3
            V_r = (1-e)* V_col_total # resin Volume cm^3

            # conc => g/cm^3
            # V => cm^3
            # integrate to get => g

            # # METHOD 1:
            # V_l = np.linspace(0,V_l,nx) # cm^3
            # V_r = np.linspace(0,V_r,nx) # cm^3
            # mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], x=x)*A_col*e # mass in liq at t=tend
            # mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], x=x)*A_col*(1-e) # mass in resin at t=tend

            # METHOD 2:
            V_l = np.linspace(0,V_l,nx) # cm^3
            V_r = np.linspace(0,V_r,nx) # cm^3

            mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], x=V_l) # mass in liq at t=tend
            mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], x=V_r) # mass in resin at t=tend

            # METHOD 3:
            # c_avg[i] = np.average(y_ode[i][:nx,-1]) # Average conc at t=tend
            # q_avg[i] = np.average(y_ode[i][:nx,-1])

            # mass_l = c_avg * V_l
            # mass_r = q_avg * V_r


        Model_Acc = mass_l + mass_r # g

        return Model_Acc

    Model_Acc = model_acc(y_matrices, V_col_total, e, num_comp)

    # ------------------------------------------
    Error = abs(Model_Acc) - abs(Expected_Acc)

    Error_percent = (sum(Error)/sum(Expected_Acc))*100
    # ------------------------------------------

    # Calculate KEY PERORMANCE PARAMETERS:
    #######################################################
    # 1. Inegral Purity
    # 2. Inegral Recovery
    # 3. Output Recovery
    # 3. Productivity




    import numpy as np



    def replace_nan_with_zero(arr: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with zeros, leaving all other values unchanged.

        Parameters
        ----------
        arr : np.ndarray
            Input matrix of shape (m, n).

        Returns
        -------
        np.ndarray
            Copy of the matrix with NaNs replaced by zeros.
        """
        return np.nan_to_num(arr, nan=0.0)




    # 1. Purity
    #######################################################
    # 1.1 Instantanoues:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        raff_inst_purity = raff_avg_cprofile/sum(raff_avg_cprofile)
        ext_inst_purity = ext_avg_cprofile/sum(ext_avg_cprofile)

    # CLEAN
    raff_inst_purity = replace_nan_with_zero(raff_inst_purity)
    ext_inst_purity = replace_nan_with_zero(ext_inst_purity)

    # 1.2 Integral:
    raff_intgral_purity = m_out_raff/sum(m_out_raff)
    ext_intgral_purity = m_out_ext/sum(m_out_ext)

    # Final Attained Purity in the Stream
    # raff_stream_final_purity = np.zeros(num_comp)
    # ext_stream_final_purity = np.zeros(num_comp)

    # for i in range(num_comp):
    #     raff_stream_final_purity[i] = raff_cprofile[i][-1]
    #     ext_stream_final_purity[i] = ext_cprofile[i][-1]

    

    # 2. Recovery
    #######################################################

    # Initialize:

    # 2.1 Instantanoues Feed Recovery:
    raff_inst_feed_recovery= np.zeros_like(raff_avg_cprofile)
    ext_inst_feed_recovery = np.zeros_like(ext_avg_cprofile)
    # 2.1 Instantanoues Output Recovery:
    raff_inst_output_recovery = np.zeros_like(raff_avg_cprofile)
    ext_inst_output_recovery = np.zeros_like(ext_avg_cprofile)

    # Populate
    for i in range(num_comp):
        # print(f'\nraff_avg_mprofile negatives\n: {raff_avg_mprofile[i]}')    
        # print(f'ext_avg_mprofile negatives\n: {ext_avg_mprofile[i]}')   


        raff_inst_feed_recovery[i] = raff_avg_mprofile[i] / C_feed[i][0]*QF
        ext_inst_feed_recovery[i] = ext_avg_mprofile[i] / C_feed[i][0]*QF

        # OUTLET RECOVERY  
        denom = raff_avg_mprofile[i] + ext_avg_mprofile[i]

        # print(f'denom\n: {denom}\n')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            raff_inst_output_recovery[i, :] = raff_avg_mprofile[i] / denom
            ext_inst_output_recovery[i, :] = ext_avg_mprofile[i] / denom

    # CLEAN;
    raff_inst_output_recovery = replace_nan_with_zero(raff_inst_output_recovery)
    ext_inst_output_recovery = replace_nan_with_zero(ext_inst_output_recovery)
    # 2.2 Integral Recovery:
    # Relaive to Feed
    raff_feed_recov = m_out_raff/m_in
    ext_feed_recov = m_out_ext/m_in
    # Relaive to Outputs
    raff_output_recov = np.zeros(num_comp)
    ext_output_recov = np.zeros(num_comp)   
    for i in range(num_comp):
        # print(f'{Names[i]} mass out @ [Raff, Ext]: [{m_out_raff[i]}, {m_out_ext[i]} ] g')
        total_mass_out_of_i = m_out_raff[i] +  m_out_ext[i]
        # print(f'\ntotal_mass_out_of_{Names[i]}: {total_mass_out_of_i} g')

        raff_output_recov[i] = m_out_raff[i]/total_mass_out_of_i
        ext_output_recov[i] = m_out_ext[i]/total_mass_out_of_i

    # print(f'\nraff_output_recov: {raff_output_recov}')
    # print(f'ext_output_recov: {ext_output_recov}')


    # 3. Productivity
    #######################################################



    # Visuliization of PERORMANCE PARAMETERS:
    #######################################################

    ############## TABLES ##################
    # Purities
    # raff_den = np.sum(raff_mprofile, axis=0, keepdims=True)   # shape (1, n_time)
    # ext_den  = np.sum(ext_mprofile, axis=0, keepdims=True)    # shape (1, n_time)

    # raff_inst_purity = np.divide(
    #     raff_mprofile, 
    #     raff_den, 
    #     out=np.zeros_like(raff_mprofile, dtype=float),
    #     where=raff_den != 0
    # )  # shape (num_comp, n_time)

    # ext_inst_purity = np.divide(
    #     ext_mprofile, 
    #     ext_den, 
    #     out=np.zeros_like(ext_mprofile, dtype=float),
    #     where=ext_den != 0
    # )  # shape (num_comp, n_time)


    # # Recoveries
    # denom = raff_mprofile + ext_mprofile   # shape (num_comp, n_time)

    # raff_inst_output_recovery = np.divide(
    #     raff_mprofile, denom,
    #     out=np.zeros_like(raff_mprofile, dtype=float),
    #     where=denom != 0
    # )

    # ext_inst_output_recovery = np.divide(
    #     ext_mprofile, denom,
    #     out=np.zeros_like(ext_mprofile, dtype=float),
    #     where=denom != 0
    # )

    print(f'{raff_output_recov}, {ext_output_recov}')
    # Define the data for the table
    data = {
        'Metric': [
            'Total Mass IN',
            'Total Mass OUT',
            'Total Expected Acc (IN-OUT)',
            'Total Model Acc (r+l)',
            'Total Error (Mod-Exp)',
            'Total Error Percent (relative to Exp_Acc)',
            '',
            f'Raffinate Purity {Names}',
            f'Extract Purity {Names}',
            # 'Final Raffinate Dimensionless Stream Concentration  [A, B,. . ]',
            # 'Final Extract Dimensionless Stream Concentration  [A, B,. . ]',
            f'Raffinate Feed Recovery {Names}',
            f'Extract Feed Recovery {Names}',
            '',
            f'Raffinate Exit Recovery {Names}',
            f'Extract Exit Recovery {Names}'
        ],
        'Value': [
            f"{m_in} g",
            f"{m_out} g",
            f'{sum(Expected_Acc)} g',
            f'{sum(Model_Acc)} g',
            f'{sum(Error)} g',
            f'{Error_percent} %\n',
            '',
            f'{raff_intgral_purity} %',
            f'{ext_intgral_purity} %',
            # f'{raff_stream_final_purity} g/cm^3',
            # f'{ext_stream_final_purity}',
            f'{raff_feed_recov} %',
            f'{ext_feed_recov} %',
            '',
            f'{raff_output_recov} %',
            f'{ext_output_recov} %'
        ]
    }

    # Create a DataFrame
    # import pandas as pd
    # df = pd.DataFrame(data)

    # # Display the DataFrame
    # print(df)
    results = [y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_feed_recov, ext_intgral_purity, ext_feed_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent, 
                raff_inst_purity, ext_inst_purity, raff_inst_feed_recovery, ext_inst_feed_recovery, raff_inst_output_recovery, ext_inst_output_recovery, raff_avg_cprofile, ext_avg_cprofile, raff_avg_mprofile, ext_avg_mprofile, t_schedule, raff_output_recov, ext_output_recov
                 ]
    return results




# Plotting Fucntions - if need be
###########################################################################################
# Loading the Plotting Libraries
# from matplotlib.pyplot import subplots
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# # from PIL import Image
# from scipy import integrate
# import plotly.graph_objects as go
# ###########################################
# # IMPORTING MY OWN FUNCTIONS
# ###########################################
# # Loading the Plotting Libraries
# from matplotlib.pyplot import subplots
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# # from PIL import Image
# from scipy import integrate
# import plotly.graph_objects as go
# ###########################################
# # IMPORTING MY OWN FUNCTIONS
# ###########################################
# def see_prod_curves(t_odes, Y, t_index):
#     """
#     Plot feed, raffinate, and extract concentration curves (zigzag + averages).

#     Parameters
#     ----------
#     t_odes : np.ndarray
#         Time vector in seconds.
#     Y : list
#         [C_feed, raff_cprofile, ext_cprofile, raff_vflow, ext_vflow,
#          raff_avg_cprofile, ext_avg_cprofile, t_schedule]
#     t_index : float
#         Indexing time (s).
#     """

#     (C_feed, raff_cprofile, ext_cprofile, _, _, 
#      raff_avg_cprofile, ext_avg_cprofile, t_schedule) = Y
#     colour_set_1 = ['lightcoral','lightblue']
#     fig, ax = plt.subplots(1, 2, figsize=(25, 5))
#     t_indexing  = t_schedule[1] - t_schedule[0] # s

#     for i, t in enumerate(t_schedule):
#         t_schedule[i] = t + t_indexing
    
#     where_to_insert = 0
#     t_schedule.insert(where_to_insert, 0)

#     t_odes_hr   = np.array(t_odes) / 3600         # convert to hours


#     t_sched_hr  = np.array(t_schedule) / 3600

#     print(f'raff_avg_cprofile : {raff_avg_cprofile}')

#     # Plot each component
#     for i in range(num_comp): 
#         label = f"{Names[i]}"

#         if iso_type == "UNC":
#             # Feed
#             ax[0].plot(t_odes_hr[i], C_feed[i], color=colors[i], label=label)

#             # Raffinate zigzag (light grey)
#             ax[1].plot(t_odes_hr[i], raff_cprofile[i], color=colour_set_1[i], linewidth=1.0, alpha=0.7)
#             # Raffinate average (colored dotted)
#             ax[1].plot(t_sched_hr, raff_avg_cprofile[i], linestyle="-", color=colors[i], linewidth=2, label=label)

#             # Extract zigzag
#             ax[2].plot(t_odes_hr[i], ext_cprofile[i], color=colour_set_1[i], linewidth=1.0, alpha=0.7)
#             # Extract average
#             ax[2].plot(t_sched_hr, ext_avg_cprofile[i], linestyle="-", color=colors[i], linewidth=2, label=label)

#         elif iso_type == "CUP":
#             # Feed
#             # ax[0].plot(t_odes_hr, C_feed[i], color=colors[i], label=label)

#             # Raffinate zigzag
#             ax[0].plot(t_odes_hr, raff_cprofile[i], color=colour_set_1[i], linewidth=1.0, alpha=0.7)
#             ax[0].plot(t_sched_hr, raff_avg_cprofile[i], linestyle="-", color=colors[i], linewidth = 1.5)

#             # Extract zigzag
#             ax[1].plot(t_odes_hr, ext_cprofile[i], color=colour_set_1[i], linewidth=1.0, alpha=0.7)
#             ax[1].plot(t_sched_hr, ext_avg_cprofile[i], linestyle="-", color=colors[i], linewidth=1.5, label=label)

#     # Axis labels and titles
#     # ax[0].set_xlabel('Time (h)', fontsize=16)
#     # # ax[0].set_ylabel('g/mL'
#     # ax[0].set_title(f'Feed Concentration Curves in g/mL') #\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
#     # # ax[0].legend()
#     # ax[0].tick_params(axis='both', labelsize=14)


#     ax[0].set_xlabel('Time (h)', fontsize=16)
#     ax[0].set_ylabel('g/mL', fontsize=16)
#     ax[0].set_title(f'Raffinate Elution Curves in g/mL', fontsize=16) #\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
#     ax[0].legend()
#     ax[0].tick_params(axis='both', labelsize=14)


#     ax[1].set_xlabel('Time (h)', fontsize=16)
#     ax[1].set_ylabel('g/mL', fontsize=16)
#     ax[1].set_title(f'Extract Elution Curves in g/mL', fontsize=16) #\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
#     ax[1].legend()
#     ax[1].tick_params(axis='both', labelsize=14)
#     plt.show()


# def col_liquid_profile(t, y, Axis_title, c_in, Ncol_num, L_total):
#     y_plot = np.copy(y)
#     # # Removeing the BC nodes
#     # for del_row in start:
#     #     y_plot = np.delete(y_plot, del_row, axis=0)
        
#     # print('y_plot:', y_plot.shape)
    
#     x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
#     dt = t[1] - t[0]
    

    
#     # Start vs End Snapshot
#     fig, ax = plt.subplots(1, 2, figsize=(25, 5))

#     ax[0].plot(x, y_plot[:, 0], label="t_start")
#     ax[0].plot(x, y_plot[:, -1], label="t_end")

#     # Add vertical black lines at positions where i % nx_col == 0
#     for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
#         x_pos = col_idx #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
#         #x_pos = dx * x_pos
#         ax[0].axvline(x=x_pos, color='k', linestyle='-')
#         ax[1].axvline(x=x_pos, color='k', linestyle='-')

#     ax[0].set_xlabel('Column Length, m')
#     ax[0].set_ylabel('($\mathregular{g/l}$)')
#     ax[0].axhline(y=c_in, color='g', linestyle= '--', linewidth=1, label="Inlet concentration")  # Inlet concentration
#     # ax[0].legend()

#     # Progressive Change at all ts:
#     for j in range(np.shape(y_plot)[1]):
#         ax[1].plot(x, y_plot[:, j])
#         ax[1].set_xlabel('Column Length, m')
#         ax[1].set_ylabel('($\mathregular{g/l}$)')
#     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter   # for smooth mean

# def col_liquid_profile(t, y, Axis_title, c_in, Ncol_num, L_total, smooth=True, window=11, poly=3):
#     """
#     Plot column concentration profiles with optional smoothing.

#     Parameters
#     ----------
#     t : array
#         Time points
#     y : 2D array
#         Concentration profiles [space, time]
#     Axis_title : str
#         Title for y-axis label
#     c_in : float
#         Inlet concentration
#     Ncol_num : int
#         Number of columns
#     L_total : float
#         Column length
#     smooth : bool
#         Whether to apply Savitzky-Golay smoothing
#     window : int
#         Window length for smoothing (must be odd)
#     poly : int
#         Polynomial order for smoothing
#     """
#     y_plot = np.copy(y)
#     nx = y_plot.shape[0]  # number of spatial nodes

#     x = np.linspace(0, L_total, nx)

#     fig, ax = plt.subplots(1, 2, figsize=(25, 5))

#     # --- Start vs End Snapshot ---
#     y_start, y_end = y_plot[:, 0], y_plot[:, -1]

#     if smooth:
#         y_start_s = savgol_filter(y_start, window, poly)
#         y_end_s   = savgol_filter(y_end, window, poly)
#     else:
#         y_start_s, y_end_s = y_start, y_end

#     ax[0].plot(x, y_start, color="gray", alpha=0.4, label="t_start (raw)")
#     ax[0].plot(x, y_start_s, "r-", label="t_start (smoothed)")
#     ax[0].plot(x, y_end, color="gray", alpha=0.4, label="t_end (raw)")
#     ax[0].plot(x, y_end_s, "b-", label="t_end (smoothed)")

#     for col_idx in range(Ncol_num + 1):
#         ax[0].axvline(x=col_idx, color='k', linestyle='-')
#         ax[1].axvline(x=col_idx, color='k', linestyle='-')

#     ax[0].set_xlabel('Column Length, m')
#     ax[0].set_ylabel(Axis_title)
#     ax[0].axhline(y=c_in, color='g', linestyle='--', linewidth=1, label="Inlet concentration")
#     ax[0].legend()

#     # --- Progressive Profiles ---
#     for j in range(y_plot.shape[1]):
#         yj = y_plot[:, j]
#         if smooth:
#             yj_s = savgol_filter(yj, window, poly)
#             ax[1].plot(x, yj_s, alpha=0.8)
#         else:
#             ax[1].plot(x, yj, alpha=0.8)

#     ax[1].set_xlabel('Column Length, m')
#     ax[1].set_ylabel(Axis_title)
#     ax[1].set_title("Progressive evolution (smoothed)" if smooth else "Progressive evolution")

#     plt.show()



# def col_solid_profile(t, y, Axis_title, Ncol_num, start, L_total):
    
#     # Removeing the BC nodes
#     y_plot = np.copy(y)
#     # Removeing the BC nodes
#     for del_row in start:
#         y_plot = np.delete(y_plot, del_row, axis=0)
        
#     # print('y_plot:', y_plot.shape)
    
#     x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
#     dt = t[1] - t[0]
    
#     # Start vs End Snapshot
#     fig, ax = plt.subplots(1, 2, figsize=(25, 5))

#     ax[0].plot(x, y_plot[:, 0], label="t_start")
#     ax[0].plot(x, y_plot[:, -1], label="t_end")
#     # ax[0].plot(x, y_plot[:, len(t) // 2], label="t_middle")

#     # Add vertical black lines at positions where i % nx_col == 0
#     for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
#         x_pos = col_idx*L #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
#         #x_pos = dx * x_pos
#         ax[0].axvline(x=x_pos, color='k', linestyle='-')
#         ax[1].axvline(x=x_pos, color='k', linestyle='-')

#     ax[0].set_xlabel('Column Length, m')
#     ax[0].set_ylabel('($\mathregular{g/l}$)')
#     ax[0].set_title(f'{Axis_title}')
#     ax[0].legend()

#     # Progressive Change at all ts:
#     for j in range(np.shape(y_plot)[1]):
#         ax[1].plot(x, y_plot[:, j])
#         ax[1].set_xlabel('Column Length, m')
#         ax[1].set_ylabel('($\mathregular{g/l}$)')
#         ax[1].set_title(f'{Axis_title}')
#     plt.show()  # Display all the figures 




# def see_prod_curves_with_data(t_odes, Y, t_index, exp_data_raff=None, exp_data_ext=None, show_exp=True):
#     """
#     Plot feed, raffinate, and extract concentration curves (zigzag + averages).
#     Also computes and displays R² values for raffinate and extract fits.

#     Parameters
#     ----------
#     t_odes : np.ndarray
#         Time vector in seconds.
#     Y : list
#         [C_feed, raff_cprofile, ext_cprofile, raff_vflow, ext_vflow,
#          raff_avg_cprofile, ext_avg_cprofile, t_schedule]
#     t_index : float
#         Indexing time (s).
#     exp_data_raff : dict, optional
#         Experimental raffinate data {i: (t_exp, C_exp)} for component i.
#     exp_data_ext : dict, optional
#         Experimental extract data {i: (t_exp, C_exp)} for component i.
#     show_exp : bool, optional
#         Whether to plot experimental data.
#     """

#     C_feed, raff_cprofile, ext_cprofile, _, _, raff_avg_cprofile, ext_avg_cprofile, t_schedule = Y

#     fig, ax = plt.subplots(1, 3, figsize=(25, 5), constrained_layout=True)

#     t_odes_hr = np.array(t_odes) / 3600       # time in hours
#     t_sched_hr = np.array(t_schedule) / 3600  # schedule in hours

#     r2_raff_all = []
#     r2_ext_all = []

#     for i in range(num_comp):
#         label_base = f"{Names[i]}"

#         # --- Feed (zigzag only) ---
#         ax[0].plot(t_odes_hr, C_feed[i], color=colors[i], label=label_base)

#         # --- Raffinate ---
#         ax[1].plot(t_odes_hr, raff_cprofile[i], color="grey", linewidth=1.0, alpha=0.7)
#         ax[1].plot(t_sched_hr, raff_avg_cprofile[i], linestyle=":", color=colors[i], linewidth=2, label=label_base)

#         # --- Extract ---
#         ax[2].plot(t_odes_hr, ext_cprofile[i], color="grey", linewidth=1.0, alpha=0.7)
#         ax[2].plot(t_sched_hr, ext_avg_cprofile[i], linestyle=":", color=colors[i], linewidth=2, label=label_base)

#         # --- Experimental data + R² computation ---
#         if show_exp:
#             if exp_data_raff and i in exp_data_raff:
#                 t_exp_r, C_exp_r = exp_data_raff[i]
#                 ax[1].scatter(t_exp_r / 3600, C_exp_r, color=colors[i], marker='x', s=40, alpha=0.6, label="Exp Raff")

#                 # Interpolate simulated raffinate average to experimental times
#                 sim_raff_interp = np.interp(t_exp_r / 3600, t_sched_hr, raff_avg_cprofile[i])
#                 ss_res = np.sum((C_exp_r - sim_raff_interp) ** 2)
#                 ss_tot = np.sum((C_exp_r - np.mean(C_exp_r)) ** 2)
#                 r2_raff = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
#                 r2_raff_all.append(r2_raff)

#             if exp_data_ext and i in exp_data_ext:
#                 t_exp_e, C_exp_e = exp_data_ext[i]
#                 ax[2].scatter(t_exp_e / 3600, C_exp_e, color=colors[i], marker='o', s=40, alpha=0.6, label="Exp Ext")

#                 # Interpolate simulated extract average to experimental times
#                 sim_ext_interp = np.interp(t_exp_e / 3600, t_sched_hr, ext_avg_cprofile[i])
#                 ss_res = np.sum((C_exp_e - sim_ext_interp) ** 2)
#                 ss_tot = np.sum((C_exp_e - np.mean(C_exp_e)) ** 2)
#                 r2_ext = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
#                 r2_ext_all.append(r2_ext)

#     # Titles and labels
#     ax[0].set_xlabel("Time (h)")
#     ax[0].set_title(f"Feed Concentration Curves (g/mL)\nConfig: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min")

#     if r2_raff_all:
#         ax[1].set_title(
#             f"Raffinate Elution Curves (g/mL)\nMean R² = {np.nanmean(r2_raff_all):.3f}\n"
#             f"Config: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min"
#         )
#     else:
#         ax[1].set_title(f"Raffinate Elution Curves (g/mL)\nConfig: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min")

#     if r2_ext_all:
#         ax[2].set_title(
#             f"Extract Elution Curves (g/mL)\nMean R² = {np.nanmean(r2_ext_all):.3f}\n"
#             f"Config: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min"
#         )
#     else:
#         ax[2].set_title(f"Extract Elution Curves (g/mL)\nConfig: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min")

#     for a in ax:
#         a.set_ylabel("Concentration (g/mL)")
#         a.legend()

#     plt.show()




# def see_instantane_outputs(t_odes, Y2, t_index, Y_labels):
#     """
#     Y2 = [raff_inst_purity, 
#           ext_inst_purity, 
#           raff_inst_output_recovery, 
#           ext_inst_output_recovery]
#     """

#     fig, ax = plt.subplots(2, 2, figsize=(25, 5), constrained_layout=True)

#     t_odes_hr = t_odes / 3600  # convert to hours

#     for i in range(num_comp):
#         label_base = f"{Names[i]}" #,{cusotom_isotherm_params_all[i]}, kh:{kav_params_all[i]}"

#         # Determine clipping bounds
#         t_min = 0
#         t_max = np.inf

#         # Manual clipping using index-based slicing (not masks)
#         if iso_type == "UNC":
#             t_i = t_odes[i]
#             start_idx = np.searchsorted(t_i, t_min, side='left')
#             end_idx   = np.searchsorted(t_i, t_max, side='right')
#             t_plot = t_i[start_idx:end_idx] / 3600

#             ax[0].plot(t_plot, Y2[0][i][start_idx:end_idx], color=colors[i], label=label_base)
#             ax[1].plot(t_plot, Y2[1][i][start_idx:end_idx], color=colors[i], label=label_base)
#             ax[2].plot(t_plot, Y2[2][i][start_idx:end_idx], color=colors[i], label=label_base)
#             ax[3].plot(t_plot, Y2[3][i][start_idx:end_idx], color=colors[i], label=label_base)

#         elif iso_type == "CUP":
#             start_idx = np.searchsorted(t_odes, t_min, side='left')
#             end_idx   = np.searchsorted(t_odes, t_max, side='right')
#             t_plot = t_odes[start_idx:end_idx] / 3600

#             ax[0].plot(t_plot, Y2[0][i][start_idx:end_idx]*100, color=colors[i], label=label_base)



#             ax[1].plot(t_plot, Y2[1][i][start_idx:end_idx]*100, color= "#1b9e77", alpha = 0.1,label=label_base)
#             # Plot the average
#             ax[1].plot(t_plot, Y2[-2][i][start_idx:end_idx]*100, color=colors[i], label=label_base)

#             ax[2].plot(t_plot, Y2[2][i][start_idx:end_idx]*100, color=colors[i], label=label_base)
#             # Plot the average
#             ax[2].plot(t_plot, Y2[-1][i][start_idx:end_idx]*100, color=colors[i], label=label_base)

#             ax[3].plot(t_plot, Y2[3][i][start_idx:end_idx]*100, color=colors[i], label=label_base)


    
# # #     # Titles and labels


#     ax[0].set_xlabel('Time, hrs')
#     ax[0].set_title(f'{Y_labels[0]} (%) \nConfig: {Z1}:{Z2}:{Z3}:{Z4}\nIndex Time: {t_index/60}min')

#     ax[1].set_xlabel('Time, hrs')
#     ax[1].set_title(f'{Y_labels[1]} (%) \nConfig: {Z1}:{Z2}:{Z3}:{Z4}\nIndex Time: {t_index/60}min')

#     ax[2].set_xlabel('Time, hrs')
#     ax[2].set_title(f'{Y_labels[2]} (%) \nConfig: {Z1}:{Z2}:{Z3}:{Z4}\nIndex Time: {t_index/60}min')

#     ax[3].set_xlabel('Time, hrs')
#     ax[3].set_title(f'{Y_labels[3]} (%) \nConfig: {Z1}:{Z2}:{Z3}:{Z4}\nIndex Time: {t_index/60}min')

#     for a in ax:
#         a.legend()
#     # plt.tight_layout()
#     plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_instantane_outputs(t_odes, Y2, titles, t_index, y_limit = None, iso_type="CUP"):
#     """
#     Visualizes instantaneous purities and recoveries for raffinate & extract.

#     Parameters
#     ----------
#     t_odes : array-like
#         Time vector (s).
#     Y2 : list of lists
#         Structure: [
#           raff_inst_purity, 
#           ext_inst_purity, 
#           raff_inst_output_recovery, 
#           ext_inst_output_recovery
#         ]
#         Each entry is [num_comp x time_array].
#     t_index : float
#         Indexing time (s).
#     iso_type : str
#         "UNC" (unconverted time array per component)
#         or "CUP" (common time array).
#     """

#     # --- Setup figure ---
#     fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
#     axs = axs.ravel()  # flatten to 1D for easy looping

#     # --- Titles for subplots ---
#     # titles = [
#     #     "Raffinate Instantaneous Purity (%)",
#     #     "Extract Instantaneous Purity (%)",
#     #     "Raffinate Instantaneous Outlet-Recovery (%)",
#     #     "Extract Instantaneous Outlet-Recovery (%)",
#     # ]

#     # --- Time conversion ---
#     def get_time_slice(t_data, t_min=0, t_max=np.inf):
#         start_idx = np.searchsorted(t_data, t_min, side='left')
#         end_idx   = np.searchsorted(t_data, t_max, side='right')
#         return t_data[start_idx:end_idx] / 3600, start_idx, end_idx  # in hours

#     # --- Plotting loop ---
#     for i in range(num_comp):
#         label_base = f"{Names[i]}"
#         scale = 1.0 if iso_type == "UNC" else y_limit[1]

#         for j in range(4):
#             if iso_type == "UNC":
#                 t_i = t_odes[i]
#             else:
#                 t_i = t_odes

#             t_plot, start_idx, end_idx = get_time_slice(t_i)
#             axs[j].scatter(
#                 t_plot,
#                 Y2[j][i][start_idx:end_idx] * scale,
#                 color=colors[i],
#                 label=label_base
#             )

#     # --- Formatting ---
#     for idx, ax in enumerate(axs):
#         ax.set_xlabel("Time (hrs)")
#         ax.set_title(
#             f"{titles[idx%len(titles)]}\nConfig: {Z1}:{Z2}:{Z3}:{Z4}, Index Time: {t_index/60:.2f} min"
#         )
#         if y_limit != None:
#             ax.set_ylim(y_limit)
#         ax.grid(True)
    
#     # Move legend outside
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=num_comp, bbox_to_anchor=(0.5, 1.05))

#     plt.show()


# def mj_to_Qj(mj, t_index_min):
#     '''
#     Converts flowrate ratios to internal flowrates - flowrates within columns
#     '''
#     r_col = d_col / 2
#     # Calculate the area of the base
#     A_col = np.pi * (r_col ** 2) # cm^2
#     V_col = A_col*L # cm^3
#     Qj = (mj*V_col*(1-e) + V_col*e)/(t_index_min*60) # cm^3/s
#     return Qj

# import json
# import numpy as np
# import os
# import time
# import json
# import numpy as np
# import os



# def make_serializable(obj):
#     """Convert numpy objects to Python scalars/lists for JSON saving."""
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (np.floating, np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.integer, np.int32, np.int64)):
#         return int(obj)
#     elif isinstance(obj, dict):
#         return {k: make_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [make_serializable(v) for v in obj]
#     else:
#         return obj  # leave as-is for plain scalars/strings


# def save_output_to_json(output, description_text="run", save_path=None):
#     """
#     Save SMB output dictionary to a JSON file.

#     Parameters
#     ----------
#     output : dict
#         Results dictionary from SMB function.
#     description_text : str
#         Descriptor to include in filename.
#     save_path : str or None
#         If None, saves to current directory.
#     """
#     serializable_output = make_serializable(output)
#     from datetime import datetime
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"sim_commissioning_results_{Names[0]}_{Names[1]}_{timestamp}.json"

#     if save_path is None:
#         save_path = os.path.join(os.getcwd(), filename)
#     else:
#         save_path = os.path.join(save_path, filename)

#     with open(save_path, "w") as f:
#         json.dump(serializable_output, f, indent=4)

#     print(f"? Results saved to {save_path}")
#     return save_path





# %%


# sub_zone information - EASIER TO FILL IN IF YOU DRAW THE SYSTEM
# -----------------------------------------------------
# sub_zone_j = [[feed_bays], [reciveinig_bays]]
# -----------------------------------------------------
# feed_bay = the bay(s) that feed the set of reciveing bays in "reciveinig_bays" e.g. [2] or [2,3,4] 
# reciveinig_bays = the set of bayes that recieve material from the feed bay

# """
# sub-zones are counted from the feed onwards i.e. sub_zone_1 is the first group "seen" by the feed stream. 
# Bays are counted in the same way, starting from 1 rather than 0
# """
# # Borate-HCL
# sub_zone_1 = [[22, 23, 24], [1]] # ---> in group 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""

# sub_zone_2 = [[3], [4,5,6]] 
# sub_zone_3 = [[4,5,6], [7]] 

# sub_zone_4 = [[9], [10, 11, 12]] 
# sub_zone_5 = [[10,11,12], [13]]

# sub_zone_6 = [[15],[16, 17, 18]]
# sub_zone_7 = [[16, 17, 18],[19]]

# sub_zone_8 = [[21],[22, 23, 24]]

# # PACK:
# grouping_type_BH_illovo = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6,sub_zone_7,sub_zone_8]

# # Glucose Fructose
# sub_zone_1 = [[3], [4,5,6]] # ---> in group 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""
# sub_zone_2 = [[4,5,6], [7,8,9]] 

# sub_zone_3 = [[7,8,9], [10,11,12]] 
# sub_zone_4 = [[10,11,12], [13]]

# sub_zone_5 = [[15],[16, 17, 18]]
# sub_zone_6 = [[16, 17, 18],[19, 20, 21]]
# sub_zone_7 = [[19, 20, 21], [22, 23, 24]]
# sub_zone_8 = [[22, 23, 24], [1]]

# # PACK:
# grouping_type_GF_commissiong = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6,sub_zone_7,sub_zone_8]



# def mj_to_Qj(mj, t_index_min, V_col, e):
#     '''
#     Converts flowrate ratios to internal flowrates - flowrates within columns
#     '''
#     Qj = (mj*V_col*(1-e) + V_col*e)/(t_index_min*60) # cm^3/s
#     return Qj

# def Qj_to_mj(Qj, t_index_min, V_col, e):
#     '''
#     Converts flowrate ratios to internal flowrates - flowrates within columns
#     '''
#     mj = (Qj * t_index_min*60 - V_col*e)/(V_col*(1-e)) # cm^3/s
#     return mj

# # # ---------------------------------------------------
# # Numerical discretization
# nx_per_col = 15  # Number of spatial nodes per column
# iso_type = "CUP"  # "CUP" or "UNC"
# Bm = 300
# # ---------------------------------------------------

# # # # # # ------- PLEASE SELECT THE SYSTEM YOU WANT TO SIMULATE ---------


# # # # # # # ILLOVO UBK
# # # # # # # parameter_sets = [ {"C_feed": 0.003190078*1.6}, {"C_feed": 0.012222}] 
# # # # # # # cusotom_isotherm_params_all = np.array([[5.567], [3.28]]) # [ [H_borate], [H_hcl] ]
# # # # # # # kav_params_all = np.array([[0.691], [0.461]])
# # # # # # # Da_all = np.array([5.80e-5, 1.90e-5]) 
# # # # # # # triangle_guess = [6.5, 3.28, 5.567, 3] # m1, m2, m3, m4

# # # # # # # ILLOVO PCR
# # # # # # # parameter_sets = [ {"C_feed": 0.003190078*1.8}, {"C_feed": 0.012222}] 
# # # # # # parameter_sets = [ {"C_feed": 0.003190078*1.8}, {"C_feed": 0.003190078*1.8}] 
# # # # # # cusotom_isotherm_params_all = np.array([[3.28], [3.16]]) # [ [H_borate], [H_hcl] ]
# # # # # # kav_params_all = np.array([[0.880], [0.486]])
# # # # # # Da_all = np.array([3.93e-5, 9.0e-5]) 
# # # # # # triangle_guess = [5.5, 3.16, 3.28, 2] # m1, m2, m3, m4


# # # # # # # # General SMB parameters for all systems
# # # # # # description_text = ['SMB_General_ON_PCR_lower_indexing_time'] # Just a general SMB simulation
# # # # # # Names =  ["Borate", "HCl"]  # ['Borate', 'HCl'] , ["Glucose", "Fructose"]
# # # # # # colors = ['red', 'blue']  #, ['red, 'blue'], ["green", "orange"]  
# # # # # # num_comp = len(Names) # Number of components

# # # # # # t_index_min = 11 # min # Index time # How long the pulse holds before swtiching
# # # # # # n_num_cycles = 7  # Number of Cycles you want the SMB to run for
# # # # # # t_simulation_end = None # HRS
# # # # # # Z1, Z2, Z3, Z4 = 4, 4, 4, 4 # *3 for smb config
# # # # # # zone_config = np.array([Z1, Z2, Z3, Z4])
# # # # # # L = 71 # cm # Length of one column
# # # # # # d_col = 5 # cm # column internal diameter
# # # # # # d_in = 0.2 * d_col # cm
# # # # # # e = 0.56 # bed voidage
# # # # # #grouping_type = [] # no subzoning

# # # # # # m1, m2, m3, m4 = triangle_guess[0],triangle_guess[1],triangle_guess[2],triangle_guess[3]
# # # # # # Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min)
# # # # # # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s





# # # # # # # 1. Subramani et al (2003) - Glucose Fructose

# # # # Subramani et al (2003) - Glucose Fructose
# # # description_text = ['SMB_Subramani_Glucose_Fructose']
# # # Names =  ["Fructose", "Glucose"]  # ['Borate', 'HCl'] , ["Glucose", "Fructose"]
# # # colors = ['red', 'blue']  #, ['red, 'blue'], ["green", "orange"]  
# # # num_comp = len(Names) # Number of components
# # # # Subramani et al (2003) - Glucose Fructose
# # # # cusotom_isotherm_params_all = np.array([[0.27],[0.53]])
# # # # kav_params_all = [[0.031], [0.0218]] # [[A], [B]]
# # # # Da_all = np.array([6.41e-6, 6.41e-6]) # cm^2/s

# # # cusotom_isotherm_params_all = np.array([[0.53],[0.27]])
# # # kav_params_all = [[0.0218], [0.031]] # [[A], [B]]
# # # Da_all = np.array([6.41e-7, 6.41e-7]) # cm^2/s

# # # parameter_sets = [{"C_feed": 0.2222}, {"C_feed": 0.2222}] #, # Fructose
# # # # Subramani et al (2003) - Glucose Fructose
# # # Q_I, Q_II, Q_III, Q_IV = 2.0412, 1.674, 1.8756, 1.44 # L/h
# # # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])/3.6 # L/h -> mL/s
# # # t_index_min = 3.3 # min # Index time # How long the pulse holds before swtiching
# # # n_num_cycles = 12  # Number of Cycles you want the SMB to run for
# # # t_simulation_end = None # HRS
# # # Z1, Z2, Z3, Z4 = 3,3,3,3 # *3 for smb config
# # # zone_config = np.array([Z1, Z2, Z3, Z4])
# # # L = 30 # cm # Length of one column
# # # e= 0.4 # bed voidage
# # # d_col = 2.6 # cm # column internal diameter
# # # d_in = 0.2 * d_col # cm
# # #grouping_type = [] # no subzoning


# # # # # # # 2. SMB Commissioning - Glucose Fructose

# # # SMB Commissioning - Glucose Fructose
# description_text = ['SMB_COMMISSIONING_Glucose_Fructose. Comparing to Pilot Plant data']
# Names =  ["Glucose", "Fructose"]  # ['Borate', 'HCl'] , ["Glucose", "Fructose"]
# colors = ['red', 'blue']  #, ['red, 'blue'], ["green", "orange"]  
# num_comp = len(Names) # Number of components
# parameter_sets =  [{"C_feed": 0.0420}, {"C_feed": 0.0430}]   # Fructose
# cusotom_isotherm_params_all = np.array([[4.20],[4.55]])
# kav_params_all = np.array([[0.169], [0.799]])
# Da_all = np.array([5.69e-5, 4.60e-5 ]) 
# # parameter_sets = [{"C_feed": 0.03 * 4.5}, {"C_feed": 0.02 * 4.5}] #, # Fructose

# # Q_I, Q_II, Q_III, Q_IV = 10, 7.96 , 8.96, 7 # L/h # Timothy  - Pilot Plant data
# # 
# # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])/3.6 # L/h -> mL/s
# t_index_min = 6 # 6.67 # min # Index time # How long the pulse holds before swtiching
# n_num_cycles = None  # Number of Cycles you want the SMB to run for
# t_simulation_end = 10 # HRS
# # Simulation Took: 15.2 min
# Z1, Z2, Z3, Z4 = 3,3,3,3 # *3 for smb config
# zone_config = np.array([Z1, Z2, Z3, Z4])
# L = 71 # cm # Length of one column
# d_col = 5 # cm # column internal diameter
# # L = 30 # cm # Length of one column
# # d_col = 2.6 # cm # column internal diameter

# d_in = 0.2 * d_col # cm
# e = 0.56 # bed voidage
# # # Calculate the radius
# r_col = d_col / 2
# # Calculate the area of the base
# A_col = np.pi * (r_col ** 2) # cm^2
# V_col = A_col*L # cm^3
#grouping_type = grouping_type_GF_commissiong
#grouping_type = [] # no subzoning

# triangle_guess = np.array([5.5, 4.20, 4.71, 3, t_index_min])
# m1, m2, m3, m4 = triangle_guess[0],triangle_guess[1],triangle_guess[2],triangle_guess[3]
# Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min, V_col, e), mj_to_Qj(m2, t_index_min, V_col, e), mj_to_Qj(m3, t_index_min, V_col, e), mj_to_Qj(m4, t_index_min, V_col, e)
# Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])

# # # # # # # # 3. SMB Illovo - Borate-HCL

# # # # # # # ## ILLOVO Waste Water ON PCR - Borate-HCL
# # # description_text = ['SMB_ILLOVO wastewater on PCR. Comparing to Pilot Plant data']
# # # Names =  ['Borate', 'HCl']   
# # # colors = ['red', 'blue']  #, ['red, 'blue'], ["green", "orange"]  
# # # num_comp = len(Names) # Number of components
# # # cusotom_isotherm_params_all = np.array([[3.907],[4.50]])
# # # kav_params_all = [[0.1027], [0.8003]] # [[A], [B]]
# # # Da_all = np.array([3.29e-5, 1.00e-4]) # cm^2/s
# # # parameter_sets = [{"C_feed": 0.012222*0.8},  {"C_feed": 0.003190078*1.4}] 
# # # Q_I, Q_II, Q_III, Q_IV = 16, 12, 15, 10 # L/h
# # # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])/3.6 # L/h -> mL/s
# # # t_index_min = 4.00 # min # Index time # How long the pulse holds before swtiching
# # # n_num_cycles = None  # Number of Cycles you want the SMB to run for
# # # t_simulation_end = 16 # HRS 
# # # Z1, Z2, Z3, Z4 = 6, 6, 6, 6 # *3 for smb config
# # # zone_config = np.array([Z1, Z2, Z3, Z4])
# # # L = 71 # cm # Length of one column
# # # d_col = 5 # cm # column internal diameter
# # # d_in = 0.2 * d_col # cm
# # # e = 0.56 # bed voidage
# # #grouping_type = grouping_type_BH_illovo # even grouping


# # # # # # ## ILLOVO Waste Water ON UBK - Borate-HCL
# description_text = ['SMB_ILLOVO wastewater on UBK. Comparing to Pilot Plant data']
# Names =  ['Borate', 'HCl']   
# colors = ['red', 'blue']  #, ['red, 'blue'], ["green", "orange"]  
# num_comp = len(Names) # Number of components
# cusotom_isotherm_params_all = np.array([[3.907],[4.50]])

# kav_params_all = [[0.1027], [0.8003]] # [[A], [B]]
# Da_all = np.array([3.29e-5, 1.00e-4]) # cm^2/s



# parameter_sets = [{"C_feed": 0.012222*0.8},  {"C_feed": 0.003190078*1.4}] 

# # Q_I, Q_II, Q_III, Q_IV = 16, 12, 15, 10 # L/h
# # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])/3.6 # L/h -> mL/s
# t_index_min = 15.00 # min # Index time # How long the pulse holds before swtiching
# n_num_cycles = 12  # Number of Cycles you want the SMB to run for
# t_simulation_end = None # HRS 
# Z1, Z2, Z3, Z4 = 3,3,3,3 # *3 for smb config
# zone_config = np.array([Z1, Z2, Z3, Z4])
# L = 71 # cm # Length of one column
# d_col = 5 # cm # column internal diameter

# # L = 30 # cm # Length of one column
# # d_col = 2.6 # cm # column internal diameter

# d_in = 0.2 * d_col # cm
# e = 0.56 # bed voidage
# # # # Calculate the radius
# r_col = d_col / 2
# # Calculate the area of the base
# A_col = np.pi * (r_col ** 2) # cm^2
# V_col = A_col*L # cm^3
# grouping_type = grouping_type_BH_illovo # even grouping
# grouping_type = [] # no subzoning
# triangle_guess = np.array([6.5, 3.26, 5.42, 2, t_index_min])
# m1, m2, m3, m4 = triangle_guess[0],triangle_guess[1],triangle_guess[2],triangle_guess[3]
# Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min, V_col, e), mj_to_Qj(m2, t_index_min, V_col, e), mj_to_Qj(m3, t_index_min, V_col, e), mj_to_Qj(m4, t_index_min, V_col, e)
# Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])


# # # # STORE/INITALIZE SMB VAIRABLES
# SMB_inputs = [iso_type, Names, colors, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all,grouping_type, t_simulation_end]
# #%% ---------- SAMPLE RUN IF NECESSARY
# start_test = time.time()
# results = SMB(SMB_inputs)
                 
# y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_feed_recov, ext_intgral_purity, ext_feed_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = results[0:19]
# raff_inst_purity, ext_inst_purity, raff_inst_feed_recovery, ext_inst_feed_recovery, raff_inst_output_recovery, ext_inst_output_recovery, raff_avg_cprofile, ext_avg_cprofile, raff_avg_mprofile, ext_avg_mprofile, t_schedule, raff_output_recov, ext_output_recov = results[19:]
# end_test = time.time()
# sim_time = end_test - start_test
# print(f'Simulation Took: {sim_time/60} min')
# # SAVE THE RESUTLS
# output = {'results': []}

# output['results'].append({      # Appended scalars:
#                                 'y_matrices': y_matrices,
#                                 'nx': nx,
#                                 't': t,
#                                 't_sets': t_sets,
#                                 't_schedule': t_schedule,
#                                 'C_feed': C_feed,
#                                 'm_in': m_in,
#                                 'm_out': m_out,

#                                 'raff_cprofile': raff_cprofile,
#                                 'ext_cprofile': ext_cprofile,
#                                 'raff_intgral_purity': raff_intgral_purity,
#                                 'raff_feed_recov': raff_feed_recov,
#                                 'ext_intgral_purity': ext_intgral_purity,
#                                 'ext_feed_recov': ext_feed_recov,

#                                 'raff_vflow': raff_vflow,
#                                 'ext_vflow': ext_vflow,

#                                 'Model_Acc': Model_Acc,
#                                 'Expected_Acc': Expected_Acc,
#                                 'Error_percent': Error_percent,

#                                 'raff_inst_purity': raff_inst_purity,
#                                 'ext_inst_purity': ext_inst_purity,
#                                 'raff_inst_feed_recovery': raff_inst_feed_recovery,
#                                 'ext_inst_feed_recovery': ext_inst_feed_recovery,
#                                 'raff_inst_output_recovery': raff_inst_output_recovery,
#                                 'ext_inst_output_recovery': ext_inst_output_recovery,

#                                 'raff_avg_cprofile': raff_avg_cprofile,
#                                 'ext_avg_cprofile': ext_avg_cprofile,
#                                 'raff_avg_mprofile': raff_avg_mprofile,
#                                 'ext_avg_mprofile': ext_avg_mprofile,

#                                 'raff_output_recov': raff_output_recov,
#                                 'ext_output_recov': ext_output_recov,

#                                 'Simulation_time': sim_time
#                                 })

# # # # Save
# save_path = save_output_to_json(output, description_text=description_text)

# # %% Plotting

# print(f'Simulation Took: {sim_time/60} min')
# # print(f'ext_cprofile: {ext_cprofile}')
# # print(f'raff_cprofile: {raff_cprofile}')
# print("-----------------------------------------------------------")

# # Extract for plotting
# #%% 

# # Y1
# Y1 = [C_feed, raff_cprofile, ext_cprofile, raff_vflow, ext_vflow, raff_avg_cprofile, ext_avg_cprofile, t_schedule]
# # Y2
# Y2 = [raff_avg_cprofile, ext_avg_cprofile, raff_avg_mprofile, ext_avg_mprofile]
# Y2_lables = ['Raffinate Concentration Profile', 'Extract Concentration Profile','Raffinate Mass Flow Profile', 'Extract Mass Flow Profile' ]

# # Y3
# Y3 = [raff_inst_purity, ext_inst_purity, raff_inst_output_recovery, ext_inst_output_recovery] # RAFF AND EXT PURITY AND RECOVERY
# Y3_lables = ['Raffinate Instantaneous Purity', 'Extract Instantaneous Purity','Raffinate Instantaneous Recovery', 'Extract  Instantaneous Recovery' ]

# # print(f'raff_avg_cprofile.shape: {np.shape(raff_avg_cprofile)}') # np.shape(raff_avg_cprofile)
# # print(f'raff_avg_cprofile: {raff_avg_cprofile}') # np.shape(raff_avg_cprofile)'

# # plt.scatter(t_schedule, ext_avg_cprofile)
# # plt.show()

# # # g/L

# t_exp_raff = np.array([120, 140, 160, 180, 210, 240, 300, 320, 340, 360, 380, 400, 420, 440, 460, 470, 480, 490, 500, 520, 530 ])*60 # seconds
# t_exp_ext = np.array([80, 100, 120, 140, 160, 180, 210, 240, 300, 320, 340, 360, 380, 400, 420,440, 460, 470, 480, 490, 500, 520, 530 ])*60


# # 0 => Glu, 1 => Fru
# exp_data_raff = {
#     0: (t_exp_raff, np.array([0.56, 2.67, 6.69, 7.01, 3.94, 5.53, 5.74, 7.11, 7.01, 7.54, 9.44, 8.81, 10.18, 9.97, 9.55, 9.12, 10.07, 9.12, 9.12, 8.81, 8.38 ])/1000),  # (time in hrs, conc)
#     1: (t_exp_raff, np.array([2.94, 4.03, 3.81, 4.19, 8.76, 4.37, 4.46, 4.79, 5.59, 6.76, 6.76, 6.19, 6.72, 6.73, 6.55, 6.78, 6.43, 7.08, 6.68, 6.99, 9.82])/1000),
# }

# exp_data_ext = {
#     0: (t_exp_ext, np.array([ 2.57,5.21,6.16,6.69,10.92,6.69,12.93, 13.25, 11.45, 15.26, 11.77, 14.20, 13.78, 14.09, 12.72, 12.61, 10.07, 14.20,12.82, 15.05, 13.88, 14.52, 15.57])/1000),
#     1: (t_exp_ext, np.array([8.73,6.39, 7.34, 7.01, 6.58, 6.91, 2.97, 2.65, 6.75, 5.64, 8.43, 6.40, 8.02, 7.31, 7.98, 8.59, 9.13, 5.80, 9.38, 6.15, 8.22, 6.28, 6.43])/1000),
# }

# if iso_type == "UNC":
#     see_prod_curves(t, Y1, t_index_min*60)
#     plot_instantane_outputs(t_schedule, Y2, t_index_min*60)
#     # see_prod_curves_with_data(t_sets, Y, t_index_min*60, exp_data_raff, exp_data_ext, show_exp=True)
# elif iso_type == "CUP":
#     see_prod_curves(t, Y1, t_index_min*60)
#     see_prod_curves_with_data(t, Y1, t_index_min*60, exp_data_raff, exp_data_ext, show_exp=True)
#     # plot_instantane_outputs(np.array(t_schedule), Y3, Y3_lables, t_index_min*60, (-1,102))
#     # plot_instantane_outputs(t, Y1, t_index_min*60, exp_data_raff, exp_data_ext, show_exp=True)


# # Define the data for the table

# sigfig = 3
# data = {
#     'Metric': [
#         'Total Expected Acc (IN-OUT)', 
#         'Total Model Acc (r+l)', 
#         'Total Error Percent (relative to Exp_Acc)', 

#         # f'Mass In {Names}',
#         # f'Mass Out {Names}',
        
#         f'Raffinate Integral Purity {Names}', 
#         f'Extract Integral Purity {Names}',

#         f'Raffinate Integral OUTPUT Recovery {Names}', 
#         f'Extract Integral OUTPUT Recovery {Names}',

#         f'Raffinate Integral Feed Recovery {Names}', 
#         f'Extract Integral Feed Recovery {Names}'
#         ],

#     'Value': [
#         f'{np.round(sum(Expected_Acc), sigfig)} g', 
#         f'{np.round(sum(Model_Acc), sigfig)} g', 
#         f'{np.round(Error_percent, sigfig + 2)} %',

#         # f'{m_in} g',
#         # f'{m_out} g', 

#         f'[{np.round(raff_intgral_purity[0], sigfig)*100}, {np.round(raff_intgral_purity[1], sigfig)*100}] %', 
#         f'[{np.round(ext_intgral_purity[0],sigfig)*100}, {np.round(ext_intgral_purity[1], sigfig)*100}] %', 

#         f'[{np.round(raff_output_recov[0],sigfig)*100}, {np.round(raff_output_recov[1],sigfig)*100}] %', 
#         f'[{np.round(ext_output_recov[0],sigfig)*100}, {np.round(ext_output_recov[1],sigfig)*100}] %',

#         f'[{np.round(raff_feed_recov[0], sigfig)*100}, {np.round(raff_feed_recov[1], sigfig)*100}] %', 
#         f'[{np.round(ext_feed_recov[0], sigfig)*100}, {np.round(ext_feed_recov[1], sigfig)*100}] %'

#     ]
# }

# import pandas as pd
# # Create a DataFrame
# df = pd.DataFrame(data)

# # Display the DataFrame
# print(df)


# # Plot the table as a figure
# fig, ax = plt.subplots(figsize=(8, 4)) # Adjust figure size as needed
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# # Format the table's appearance
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.5, 1.5)  # Adjust scaling of the table

# # Display the table
# plt.show()


# #%%
# print(f'shape(y): {np.shape(y_matrices)}')

# #%%


# # ANIMATION
# ###########################################################################################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def animate_smb_concentration_profiles(y, t, labels, colors, nx_per_col, cols_per_zone, L_col,
#                                        t_index, parameter_sets, filename="smb_profiles.mp4"):
#     """
#     Create an animated visualization of SMB liquid-phase concentration profiles across zones.

#     Parameters:
#     - y: (n_components, nx_total, time_points) concentration data
#     - t: time vector (in seconds)
#     - labels: list of component names
#     - colors: list of colors per component
#     - nx_per_col: number of spatial points per column
#     - cols_per_zone: list with number of columns per zone
#     - L_col: length of each column (m)
#     - t_index: time (s) for 1 indexing shift
#     - parameter_sets: list of dicts per component
#     - filename: output video file name
#     """
#     n_components, nx_total, nt = y.shape
#     n_zones = len(cols_per_zone)
#     n_cols_total = sum(cols_per_zone)
#     L_total = n_cols_total * L_col

#     # Determine frame indices for animation (= 90s total shown if t > 120s)
#     if t[-1] > 120:
#         t_segment = 30  # seconds
#         frames_per_segment = int(t_segment / (t[1] - t[0]))
#         first_idx = np.arange(0, frames_per_segment)
#         middle_idx = np.arange(nt // 2 - frames_per_segment // 2, nt // 2 + frames_per_segment // 2)
#         last_idx = np.arange(nt - frames_per_segment, nt)
#         selected_frames = np.concatenate([first_idx, middle_idx, last_idx])
#     else:
#         selected_frames = np.arange(nt)

#     # Calculate column junction positions
#     col_boundaries = [i * nx_per_col for i in range(n_cols_total + 1)]
#     x_full = np.linspace(0, L_total, nx_total)

#     # Initial port positions (index in spatial array), assuming inlet at col 0 (zone 3)
#     stream_order = ["Feed", "Extract", "Raffinate", "Desorbent"]
#     stream_colors = ["red", "blue", "orange", "purple"]
#     stream_zone = [2, 1, 3, 0]  # zone indices: Feed at zone 3 (index 2), etc.

#     # Compute starting port positions in terms of column number
#     start_ports = np.cumsum([0] + cols_per_zone[:-1])  # column index per zone start

#     # Pre-compute port positions over time (indexed every t_index)
#     port_positions = {stream: [] for stream in stream_order}
#     for time_val in t:
#         idx_shift = int(time_val // t_index)
#         for i, stream in enumerate(stream_order):
#             base_col = start_ports[stream_zone[i]]
#             pos = (base_col + idx_shift) % n_cols_total
#             port_positions[stream].append(pos * nx_per_col * L_col)  # convert to length

#     # Set up figure and axes (4 stacked panels)
#     fig, axes = plt.subplots(n_zones, 1, figsize=(8, 10), sharex=True)
#     lines = [[] for _ in range(n_zones)]

#     for zone_id, ax in enumerate(axes):
#         ax.set_xlim(0, L_total)
#         ax.set_ylim(0, np.max(y))
#         ax.set_ylabel("C (g/L)")
#         ax.set_title(f"Zone {zone_id + 1}")

#         # Vertical black lines for column boundaries
#         col_start = sum(cols_per_zone[:zone_id]) * nx_per_col * L_col
#         for i in range(cols_per_zone[zone_id] + 1):
#             ax.axvline(x=col_start + i * L_col, color='black', linewidth=0.5)

#         # Plot initialization for each component
#         for comp_idx in range(n_components):
#             # Spatial slice for this zone
#             start = sum(cols_per_zone[:zone_id]) * nx_per_col
#             end = start + cols_per_zone[zone_id] * nx_per_col
#             x = x_full[start:end]
#             line, = ax.plot(x, y[comp_idx, start:end, 0], color=colors[comp_idx], label=labels[comp_idx])
#             lines[zone_id].append(line)

#     # Add time and legend box
#     time_text = axes[0].text(0.95, 0.9, '', transform=axes[0].transAxes,
#                              ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
#     axes[-1].set_xlabel("Column Length (m)")
#     axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

#     # Initialize stream vertical lines
#     stream_lines = [axes[0].axvline(0, color=color, linestyle='--', linewidth=1.5) for color in stream_colors]

#     # Update function
#     def update(frame_idx):
#         t_hr = t[frame_idx] / 3600  # convert to hours
#         time_text.set_text(f"Time: {t_hr:.2f} h")

#         for zone_id, ax in enumerate(axes):
#             start = sum(cols_per_zone[:zone_id]) * nx_per_col
#             end = start + cols_per_zone[zone_id] * nx_per_col
#             for comp_idx in range(n_components):
#                 lines[zone_id][comp_idx].set_ydata(y[comp_idx, start:end, frame_idx])

#         # Update stream lines (positioned in top axis only)
#         for i, stream in enumerate(stream_order):
#             x_pos = port_positions[stream][frame_idx]
#             stream_lines[i].set_xdata(x_pos)

#         return [l for sublist in lines for l in sublist] + stream_lines + [time_text]

#     # Create animation
#     ani = animation.FuncAnimation(fig, update, frames=selected_frames, interval=100, blit=True)

#     # Save animation
#     writer = animation.FFMpegWriter(fps=15, bitrate=1800)
#     ani.save(filename, writer=writer)
#     plt.close()
#     return filename

# # Run it with the simulated data
# sample_data_bundle = {
#     "y": y_matrices,
#     "t": t,
#     "labels": Names,
#     "colors": colors,
#     "nx_per_col": nx_per_col,
#     "cols_per_zone": zone_config,
#     "L_col": L,
#     "t_index": t_index_min,
#     "parameter_sets": parameter_sets
# }

# def plot_all_columns_single_axes(y_matrices, t,indxing_period, time_index,
#                                   nx_per_col, L_col, zone_config, 
#                                   labels=None, colors=None,
#                                   title="Concentration Across Entire Column"):
#     """
#     Plot all columns together on one continuous axis for each component at a given time index.

#     Parameters:
#     - y_matrices: array of shape (n_components, nx_total, n_timepoints)
#     - time_index: int, index along the time axis
#     - nx_per_col: int, spatial points per column
#     - L_col: float, physical length of each column
#     - labels: list of component names (optional)
#     - colors: list of line colors per component (optional)
#     - title: overall figure title
#     """

#     n_components, nx_total, n_timepoints = y_matrices.shape

#     n_columns = np.sum(zone_config)
#     nx_total = nx_per_col*n_columns # just for the liquid phase

#     total_length = L_col * n_columns # cm
#     x = np.linspace(0, total_length, nx_total)
#     # Rotate so that Z3 is first
#     zone_config_rot = np.roll(zone_config, -2)  # => [Z3, Z4, Z1, Z2]

#     # Labels accordingly
#     zone_labels = ['Z3', 'Z4', 'Z1', 'Z2']

#     plt.figure(figsize=(10, 6))
#     for i in range(n_components):
#         y_vals = y_matrices[i][ 0:nx_total, time_index]
#         label = labels[i] if labels else f"Comp {i+1}"
#         color = colors[i] if colors else None
#         plt.plot(x, y_vals, label=label, color=color)
#     x_plot = 0
#     for i, x_zone in enumerate(zone_config_rot):
#         x_plot += x_zone
#         plt.axvline(x=x_plot*L, color='k', linestyle='--', linewidth=2)
#         plt.text(x_plot*L, plt.ylim()[1]*0.995, zone_labels[i], ha='right', va='top',
#                 fontsize=8, color='k')
#     for i in range(0,n_columns+1):
#         x_plot = i*L
#         plt.axvline(x=x_plot, color='grey', linestyle='--', linewidth=1)

#     plt.title(f"{title} (Time Index {time_index}) (Time Stamp: {t[time_index]/60} min)\nIndxing_period: {indxing_period} min")
#     plt.xlabel("Position along full unit (cm)")
#     plt.ylabel("Concentration (g/L)")
#     # plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# #%%
# # animate_smb_concentration_profiles(**sample_data_bundle)
# # plot_all_columns_single_axes(
# #     y_matrices=y_matrices,
# #     t = t,
# #     indxing_period = t_index_min,
# #     # time_index= int(np.round(np.shape(y_matrices)[2]*0.01)),
# #     time_index= 700, # 27, 31, 35. 70, 90
# #     nx_per_col=nx_per_col,
# #     L_col = L,
# #     zone_config = zone_config,
# #     labels=Names,
# #     colors= colors
# # )




# # %%



# # %%
