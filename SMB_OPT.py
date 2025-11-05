# -*- coding: utf-8 -*-
# # %%


#%%

# -*- coding: utf-8 -*-
# # %%


#%%


print(f'Hello from SMB_OPT.py')

def constrained_MOBO_func(batch, SMB):
    # UNPACK "batch":
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

    # Import the custom SMB model
    from SMB_func_general import SMB
    
    opt_inputs = batch[0]
    SMB_inputs = batch[1]
    
    # UNPACK RESPECTIVE INPUTS
    Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, optimization_budget, constraint_threshold, PF_weight, bounds, triangle_guess, x_i, similarity_threshold = opt_inputs[0:]
    iso_type, Names, color, num_comp, nx_per_col, e, Da_all, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, grouping_type, t_simulation_end = SMB_inputs[0:]

    
    # SECONDARY VARIABLES

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
    


    # - - - - -
    m_diff_max = Q_max/(V_col*(1-e))
    m_diff_min = Q_min/(V_col*(1-e))
    
    # - - - - -

    def lhq_sample_mj(m_min, m_max, n_samples, diff=0.1):
        """
        Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
        Note that for all mjs: (m_min < m_j < m_max)
        And that:
            (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
            (ii)  m2 < m3 - (diff*m3)
            (iii) m3 > m4 + (diff*m4)
        Final result is an np.array of size: (n_samples, 4)
        """

        # Initialize the array to store the samples
        samples = np.zeros((n_samples, 4))

        for i in range(n_samples):
            # Sample m1, m2, m3, m4 within bounds and respecting constraints
            m1 = np.random.uniform(m_min, m_max)
            m4 = np.random.uniform(m_min, m1-diff*m1)

            # Sample m2 such that it respects the constraint: m2 < m1 - (diff*m1)
            m2 = np.random.uniform(m_min, m_max)
            while m2 >= m1 - (diff * m1):  # Ensuring m2 < m1 - (diff*m1)
                m2 = np.random.uniform(m_min, m_max)

            # Sample m3 such that it respects the constraint: m3 > m2 + (diff*m2)
            m3 = np.random.uniform(m_min, m_max)
            while m3 <= m2 + (diff * m2):  # Ensuring m3 > m2 + (diff*m2)
                m3 = np.random.uniform(m_min, m_max)

            # Ensure the constraint: m3 > m4 + (diff * m4)
            while m3 <= m4 + (diff * m4):  # Ensuring m3 > m4 + (diff*m4)
                m3 = np.random.uniform(m_min, m_max)

            # Store the sample in the array
            samples[i] = [m1, m2, m3, m4]

        return samples

    def fixed_feed_lhq_sample_mj(t_index_min, Q_fixed_feed, m_min, m_max, n_samples, diff=0.1):
        """
        Since the feed is fixed, m3 is caluclated

        Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
        Note that for all mjs: (m_min < m_j < m_max)
        And that:
            (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
            (ii)  m2 < m3 - (diff*m3)
            (iii) m3 > m4 + (diff*m4)
        Final result is an np.array of size: (n_samples, 4)
        """
        # Initialize the array to store the samples
        samples = np.zeros((n_samples, 4))

        for i in range(n_samples):
            # Sample m1, m2, m3, m4 within bounds and respecting constraints
            m1 = np.random.uniform(m_min, m_max)
            m4 = np.random.uniform(m_min, m1-diff*m1)

            # Sample m2 such that it respects the constraint: m2 < m1 - (diff*m1)
            m2 = np.random.uniform(m_min, m_max)
            while m2 >= m1 - (diff * m1):  # Ensuring m2 < m1 - (diff*m1)
                m2 = np.random.uniform(m_min, m_max)

            # Sample m3 such that it respects the constraint: m3 > m2 + (diff*m2)
            m3 = np.random.uniform(m_min, m_max)
            while m3 <= m2 + (diff * m2):  # Ensuring m3 > m2 + (diff*m2)
                m3 = np.random.uniform(m_min, m_max)

            # Ensure the constraint: m3 > m4 + (diff * m4)
            while m3 <= m4 + (diff * m4):  # Ensuring m3 > m4 + (diff*m4)
                m3 = np.random.uniform(m_min, m_max)

            # Store the sample in the array
            samples[i] = [m1, m2, m3, m4]

        return samples

    def fixed_m1_and_m4_lhq_sample_mj(m1, m4, m_min, m_max, n_samples, n_m2_div, diff=0.1):
        """
        - Since the feed is fixed, m3 is caluclated AND
        - Since the desorbant is fixed, m1 is caluclated

        Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
        Note that for all mjs: (m_min < m_j < m_max)
        And that:
            (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
            (ii)  m2 < m3 - (diff*m3)
            (iii) m3 > m4 + (diff*m4)
        Final result is an np.array of size: (n_samples, 4)

        [1.78902051 1.10163238 1.75875405 0.20421105], 7.438074624877448

        """
        
        # Initialize the array to store the samples
        samples = np.zeros((n_samples*n_m2_div, 5))
        samples[:,0] = np.ones(n_samples*n_m2_div)*m_max
        
        samples[:,-2] = np.ones(n_samples*n_m2_div)*2
        # print(f'samples: {samples}')
        nn = int(np.round(n_samples/2))
        num_of_m3_per_m2 = n_samples

        m2_set = np.linspace(m_min, m_max*0.9, n_m2_div)
        # print(f'm2_set: {m2_set}')

        i = np.arange(len(m2_set))
        k = np.repeat(i,num_of_m3_per_m2)

        print(f'k:{k}')
        #Sample from the separation triangle:
        for i in range(len(k)): # for each vertical line
            # print(f'k: {k[i]}')
            m2 = m2_set[k[i]]

            samples[i, 1] = m2

            if i == 0:
                m2 = 0.8
                m3 = m2 + 0.1
                samples[i, 1] = m2
                samples[i, 2] = m3  # apex of trianlge
            else:
                m3 = np.random.uniform(m2, m_max)   

            samples[i, 2] = m3
            samples[i, -2] = m2 - 0.3
            samples[i,-1] = 0.6
        return samples


    # ---------- Objective Function

    def mj_to_Qj(mj, t_index_min):
        '''
        Converts flowrate ratios to internal flowrates - flowrates within columns
        '''
        Qj = (mj*V_col*(1-e) + V_col*e)/(t_index_min*60) # cm^3/s
        return Qj

    # Define the obj and constraint functions
    # All parameteres
    def obj_con(X):
        """Feasibility weighted objective; zero if not feasible.

            X = [m1, m2, m3, m4, t_index];
            Objective: WAR = Weighted Average Recovery
            Constraint: WAP = Weighted Average Purity

            Use WAP to calculate the feasibility weights. Which
            will scale teh EI output.

        """
        X = np.array(X)


        # print(f'np.shape(x_new)[0]: {np.shape(X)}')
        if X.ndim == 1:

            Pur = np.zeros(2)
            Rec = np.zeros(2)
            # Unpack and convert to float and np.arrays from torch.tensors:
            m1, m2, m3, m4, t_index_min = float(X[0]), float(X[1]), float(X[2]), float(X[3]), float(X[4])

            # print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')

            Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min)
            Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
            Qfeed = Q_III - Q_II
            Qraffinate = Q_III - Q_IV
            Qdesorbent = Q_I - Q_IV
            Qextract = Q_I - Q_II
            Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
        
            print(f'----------------------------------')
            print(f'Q_internal: {Q_internal*3.6} L/h [Q1, Q2, Q3, Q4]')
            print(f'Q_external: {Q_external*3.6} L/h [QF, QR, QD, QE]')
            print(f'----------------------------------')
      

            SMB_inputs[11] = t_index_min  # Update t_index
            SMB_inputs[13] = Q_internal # Update Q_internal

            results = SMB(SMB_inputs)

          

            raff_purity_both_comp = results[10]  # [Glu, Fru]
            ext_purity_both_comp = results[12]  # [Glu, Fru]

            # Recovery
            # i=11,13 (raff, ext) for  relative to feed
            # i=-2,-1 (raff, ext) for  relative to exit
            raff_recovery_both_comp = results[11]  # [Glu, Fru]
            ext_recovery_both_comp = results[13]  # [Glu, Fru]

            # 1. At the raffinate, we will always want high purity of the less retained component (Glu)
            # 2. At the extract, we will always want high purity of the more retained component (Fru)
            min_index = np.argmin(cusotom_isotherm_params_all)
            max_index = np.argmax(cusotom_isotherm_params_all)

            # RAFFINATE:
            # 1. At the raffinate, we will always want high purity of the less retained component (Glu)
            pur1 = raff_purity_both_comp[min_index]
            rec1 = raff_recovery_both_comp[min_index]
            # print(f'Optimizing {Names[min_index]} at` the Raffinate')
            # EXTRACT:
            # 2. At the extract, we will always want high purity of the more retained component (Fru)
            pur2 = ext_purity_both_comp[max_index]
            rec2 = ext_recovery_both_comp[max_index]
            # print(f'Optimizing {Names[max_index]} at` the Extract')



            # Pack
            # WAP[i] = WAP_add
            # WAR[i] = WAR_add
            Pur[:] = [pur1, pur2] # [raff, ext]
            Rec[:] = [rec1, rec2] # [raff, ext]

        elif X.ndim > 1:
            Pur = np.zeros((len(X[:,0]), 2))
            Rec = np.zeros((len(X[:,0]), 2))

            for i in range(len(X[:,0])):

                # Unpack and convert to float and np.arrays from torch.tensors:
                m1, m2, m3, m4, t_index_min = float(X[i,0]), float(X[i,1]), float(X[i,2]), float(X[i,3]), float(X[i,4])

                # print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')
                Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min) 
                Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
                # Calculate and display external flowrates too
                Qfeed = Q_III - Q_II
                Qraffinate = Q_III - Q_IV
                Qdesorbent = Q_I - Q_IV
                Qextract = Q_I - Q_II

                Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
                # print(f'Q_internal: {Q_internal} cm^s/s')
                # print(f'Sampled Inputs')
                print(f'----------------------------------')
                print(f'Q_internal: {Q_internal*3.6} L/h [Q1, Q2, Q3, Q4]')
                # print(f'Q_external: {Q_external} cm^s/s')
                print(f'Q_external: {Q_external*3.6} L/h [QF, QR, QD, QE]')
                print(f'----------------------------------')

                # print(f'Q_internal type: {type(Q_internal)}')
                # Update SMB_inputs:
                SMB_inputs[11] = t_index_min  # Update t_index
                SMB_inputs[13] = Q_internal # Update Q_internal

                results = SMB(SMB_inputs)

                # print(f'done solving sample {i+1}')

                raff_purity = results[10]  # [Glu, Fru]
                ext_purity = results[12]  # [Glu, Fru]

                # Recovery
                # i=11,13 (raff, ext) for recovery relative tp feed
                # i=-2,-1 (raff, ext) for recovery relative to exit

                raff_recovery = results[11]  # [Glu, Fru]
                ext_recovery = results[13]  # [Glu, Fru]

                # 1. At the raffinate, we will always want high purity of the less retained component (Glu)
                # 2. At the extract, we will always want high purity of the more retained component (Fru)
                # Pack
                min_index = np.argmin(cusotom_isotherm_params_all)
                max_index = np.argmax(cusotom_isotherm_params_all)

                # RAFFINATE:
                # 1. At the raffinate, we will always want high purity of the less retained component (Glu)
                pur1 = raff_purity[min_index]
                rec1 = raff_recovery[min_index]
                # print(f'Optimizing {Names[min_index]} at` the Raffinate')
                # EXTRACT:
                # 2. At the extract, we will always want high purity of the more retained component (Fru)
                pur2 = ext_purity[max_index]
                rec2 = ext_recovery[max_index]
                # print(f'Optimizing {Names[max_index]} at` the Extract')
                # print(f'Optimizing {Names[min_index]} at` the Raffinate')
                Pur[i,:] = [pur1, pur2]
                Rec[i,:] = [rec1, rec2]
                # print(f'Pur: {pur1}, {pur2}')
                # print(f'Rec: {rec1}, {rec2}\n\n')

        return  Rec, Pur, np.array([m1, m2, m3, m4, t_index_min])


    # ------ Generate Initial Data
    def generate_initial_data(triangle_guess, sampling_budget=1):

        # generate training data
        triangle_guess = np.array([triangle_guess]) # add additional layer
        # # print(f'Getting {sampling_budget} Samples')
        # # train_x = lhq_sample_mj(0.2, 1.7, n, diff=0.1)
        # # train_x = fixed_feed_lhq_sample_mj(t_index_min, Q_fixed_feed, 0.2, 1.7, n, diff=0.1)
        # train_all = fixed_m1_and_m4_lhq_sample_mj(m_max, m_min, m_min, m_max, sampling_budget, 1, diff=0.1)
        # print(f'train_all: {train_all}')
        # print(f'Done Getting {sampling_budget} Samples')

        # print(f'Solving Over {sampling_budget} Samples')
        # print(f'\n\ntriangle_guess: {triangle_guess}')
        # print(f'train_all: {train_all}')
        print('\n\n----------------------------------')
        print(f'Trangle Guess Reuslts')
        print('----------------------------------')
        print(f'Inputs:')
        Rec, Pur, mjs = obj_con(triangle_guess)
        print(f'Ouputs:')
        print(f'Recoveries [Raff, Ext]: {Rec}, \nPurities [Raff, Ext]: {Pur}')
        
        all_outputs = np.hstack((Rec, Pur))

        return triangle_guess, all_outputs
    
    # ------------------ BO FUNTIONS
    # --- Surrogate model creation ---
    def surrogate_model(X_train, y_train):
        X_train = np.atleast_2d(X_train)
        y_train = np.atleast_1d(y_train)

        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        # kernel = C(1.0, (1e-4, 10.0)) * RBF(1.0, (1e-4, 10.0))
        kernel = Matern(length_scale=1.0, nu=1.5)

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=5)

        gp.fit(X_train, y_train)

        return gp

#     from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


    def surrogate_model(X_train, y_train):
        X_train = np.atleast_2d(X_train)
        y_train = np.atleast_1d(y_train)

        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        n_dims = X_train.shape[1]

        # --- ARD Matern kernel (? = 3/2) ---
        # Give each dimension its own length scale
        # You can start with all ones or heuristic based on data spread
        initial_lengthscales = np.ones(n_dims)

        # Optional: allow optimizer to tune within reasonable bounds
        # length_scale_bounds = [(1e-3, 15.0)] * n_dims

        # kernel = C(1.0, (1e-3, 15.0)) * Matern(length_scale=initial_lengthscales,
        #                                     length_scale_bounds=length_scale_bounds,
        #                                     nu=1.5)
        
        kernel = Matern(length_scale=1.0, nu=1.5)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )

        gp.fit(X_train, y_train)

        # Print learned ARD lengthscales for inspection
        # print("Learned ARD lengthscales:", gp.kernel_.k2.length_scale)

        return gp


    # --- AQ funcs:
    # def dim5_distance_bonus_scalar(x_candidate, X_obs, scale=1.0):
    #     if X_obs.shape[0] == 0:
    #         return 1.0
    #     dists = np.abs(X_obs[:, 4] - x_candidate[4])
    #     min_dist = np.min(dists)
    #     max_range = np.max(X_obs[:,4]) - np.min(X_obs[:,4])
    #     if max_range <= 0:
    #         return 0.0
    #     return (min_dist / max_range) * scale
    
    # def acquisition_with_dim5_boost(X_obs, X_cand, ei, alpha=5.0):
    #     # mu, sigma = gp.predict(X_cand, return_std=True)
    #     # ei = expected_improvement(mu, sigma, y_best, xi=xi)
    #     boosted = np.zeros_like(ei)
    #     for i, x in enumerate(X_cand):
    #         bonus = dim5_distance_bonus_scalar(x, X_obs, scale=1.0)
    #         boosted[i] = ei[i] * (1.0 + alpha * bonus)
    #     return boosted



    # # --- AQ func: Expected Constrained Improvement ---
    # def log_expected_constrained_improvement(x, surrogate_obj_gp, constraint_gps, constraint_thresholds, y_best,job, PF_weight, xi):
    #     x = np.asarray(x).reshape(1, -1)
    #     print(f'x: {x}')
    #     # x = np.round(x, 4)
    #     mu_obj, sigma_obj = surrogate_obj_gp.predict(x, return_std=True)
    #     # print(f'mu_obj: {mu_obj}, sigma_obj: {sigma_obj} ')



    #     with np.errstate(divide='warn'):
    #     # Note that, because we are maximizing, y_best > mu_obj - always,
    #     # So Z is always positive. And if Z is positive then its on the "right-hand-side" of the mean
    #     # Because norm.cdf calcualtes the integral from left to right, it will by default calculate-
    #     # the probability of begin LESS than (or equal to), Z.
    #     # Since we want to probability of being greater than or equal to Z, we have two options
    #     # (1.) Calculate Z -> and compute 1-norm.cdf(Z) 
    #     # (2.) Calculate abs(Z) -> make Z negative, -abs(Z) -> integrate norm.cdf(-abs(Z))
        
    #     # The (2.) option is more robust - because by simply omitting the negative sign, it will also work for cases where we are minimizing the objective
    #     # (2.) is used below
        
    #         if job == 'maximize':
    #                 # print(f'{job}ing')
    #                 # Calulate Z and enforce, -ve
    #                 Z = -np.abs(y_best - mu_obj) / sigma_obj
    #                 #  Note: ei is always positive

    #                 # 1. 
    #                 # use logcdf() to get the log probability for cases where Z is very small
    #                 # Since Z < 0, logcdf() < 0 (not a probability)
    #                 log_cdf_term = norm.logcdf(Z) 
    #                 # recover the actual probabitlty
    #                 cdf_term = np.exp(log_cdf_term)

    #                 # 2. Do the same for the pdf term
    #                 # Since Z < 0, logpdf() < 0 (not a likihood)
    #                 log_pdf_term = norm.logcdf(Z) 
    #                 # recover the actual likihood
    #                 pdf_term = np.exp(log_pdf_term)
    #                 # ---------------------------------------------------
    #                 ei = (y_best - mu_obj) * cdf_term + sigma_obj * pdf_term * xi
    #                 # ei = sigma_obj * pdf_term * xi

    #         elif job == 'minimize':
    #                 # Calulate Z and enforce, -ve
    #                 Z = np.abs(y_best - mu_obj) / sigma_obj
    #                 #  Note: ei is always positive

    #                 # 1. 
    #                 # use logcdf() to get the log probability for cases where Z is very small
    #                 # Since Z < 0, logcdf() < 0 (not a probability)
                    
    #                 log_cdf_term = norm.logcdf(Z) 
    #                 # recover the actual probabitlty
    #                 cdf_term = np.exp(log_cdf_term)

    #                 # 2. Do the same for the pdf term
    #                 # Since Z < 0, logpdf() < 0 (not a likihood)
    #                 log_pdf_term = norm.logcdf(Z) 
    #                 # recover the actual likihood
    #                 pdf_term = np.exp(log_pdf_term)
    #                 # ---------------------------------------------------
    #                 ei = (mu_obj - y_best) * cdf_term + sigma_obj * pdf_term * xi
        
    #         # print(f'ei: {ei}')

    #     # print(f'ei: {ei}')


    #     # Calcualte the probability of Feasibility, "prob_feas"
    #     prob_feas = 1.0 # initialize

    #     for gp_c, lam in zip(constraint_gps, constraint_thresholds):

    #         mu_c, sigma_c = gp_c.predict(x, return_std=True)

    #         # lam -> inf = 1- (-inf -> lam)
    #         prob_that_LESS_than_mu = norm.cdf((lam - mu_c) / sigma_c)

    #         prob_that_GREATER_than_mu = 1 - prob_that_LESS_than_mu

    #         pf = prob_that_GREATER_than_mu

    #         # pf is a vector,
    #         # We just want the non-zero part
    #         pf = pf[pf != 0]
    #         # If theres no, non-zero part, we need to Avoid pf being an empty array:
    #         if pf.size == 0 or pf < 1e-12:
    #             pf = 1e-8


    #         # print(f'pf: {pf}')

    #         # if we assume that the condtions are independent,
    #         # then we can "multiply" the weights to get the "joint probability" of feasility
    #         prob_feas *= pf

    #     # print(f'ei: {ei}')
    #     # print(f'prob_feas: {prob_feas}')

    #     PF = np.clip(prob_feas, 1e-12, 1.0)
    #     PFI = ei * PF

    #     ei_boost = acquisition_with_dim5_boost(x, x[:,-1], ei, alpha=5.0)
        
    #     log_eic = np.log(ei + 1e-12) + np.log(PF + 1e-12) * np.sqrt(PF_weight)
    #     # log_eic = np.log(ei) + PF_weight*np.log(prob_feas)

    #     # print(f'log_eic: {log_eic}')
    #     # print(f'Convert to float')

    #     log_eic = float(np.squeeze(log_eic))  # Convert to scalar
    #     PFI = float(np.squeeze(PFI))  # Convert to scalar


    #     return -log_eic

    def dim5_distance_bonus_scalar(x_candidate, X_obs, scale=1.0):
        """
        Returns a bonus multiplier (>= 1) that increases with how far x_candidate[4]
        is from all observed points in dimension 5.
        """
        if X_obs.shape[0] == 0:
            return 1.0  # no prior data, neutral bonus
        
        dists = np.abs(X_obs[:, 4] - x_candidate[0, 4])  # distance in 5th dim
        min_dist = np.min(dists)
        max_range = np.max(X_obs[:, 4]) - np.min(X_obs[:, 4])
        
        if max_range <= 0:
            return 1.0  # all points identical in dim5
        
        # Scale the distance to [0, 1]
        norm_dist = min_dist / max_range
        # Compute a bonus multiplier
        return 1.0 + scale * norm_dist

    def wrapped_log_expected_constrained_improvement(
        x, surrogate_obj_gp, constraint_gps, constraint_thresholds,
        y_best, job, PF_weight, xi, X_obs, dim5_boost_scale,
        fixed_dims=None
    ):
        """
        Wrapper that optionally fixes the first 4 dimensions and only optimizes the 5th.
        """
        x = np.asarray(x).reshape(1, -1)

        # If we're in the exploration phase, overwrite first 4 dims with fixed values
        if fixed_dims is not None:
            x[0, :4] = fixed_dims[:4]  # Fix dimensions 1-4

        return log_expected_constrained_improvement(
            x=x,
            surrogate_obj_gp=surrogate_obj_gp,
            constraint_gps=constraint_gps,
            constraint_thresholds=constraint_thresholds,
            y_best=y_best,
            job=job,
            PF_weight=PF_weight,
            xi=xi,
            X_obs=X_obs,
            dim5_boost_scale=dim5_boost_scale
        )

    def log_expected_constrained_improvement(
        x, surrogate_obj_gp, constraint_gps, constraint_thresholds, 
        y_best, job, PF_weight, xi, X_obs=None, dim5_boost_scale=5.0
    ):
        x = np.asarray(x).reshape(1, -1)
        
        mu_obj, sigma_obj = surrogate_obj_gp.predict(x, return_std=True)
        
        # --- Expected Improvement calculation (same as before) ---
        if job == 'maximize':
            Z = -np.abs(y_best - mu_obj) / sigma_obj
        else:
            Z = np.abs(y_best - mu_obj) / sigma_obj

        log_cdf_term = norm.logcdf(Z)
        cdf_term = np.exp(log_cdf_term)
        log_pdf_term = norm.logcdf(Z)
        pdf_term = np.exp(log_pdf_term)
        
        if job == 'maximize':
            ei = (y_best - mu_obj) * cdf_term + sigma_obj * pdf_term * x_i
        else:
            ei = (mu_obj - y_best) * cdf_term + sigma_obj * pdf_term * x_i

        # --- Probability of Feasibility ---
        prob_feas = 1.0
        for gp_c, lam in zip(constraint_gps, constraint_thresholds):
            mu_c, sigma_c = gp_c.predict(x, return_std=True)
            pf = 1 - norm.cdf((lam - mu_c) / sigma_c)
            pf = np.clip(pf, 1e-12, 1.0)
            prob_feas *= pf
        
        PF = np.clip(prob_feas, 1e-12, 1.0)
        
        # --- Apply exploration bonus in dimension 5 ---
        # if X_obs is not None:
        #     dim5_bonus = dim5_distance_bonus_scalar(x, X_obs, scale=dim5_boost_scale)
        # else:
        #     dim5_bonus = 1.0

        # # Apply the boost directly to EI (functional form)
        # ei_boosted = ei

        # --- Combine into log Expected Feasible Improvement ---
        log_pfi = np.log(ei + 1e-12) + np.sqrt(PF_weight) * np.log(PF + 1e-12)
        pfi = ei * PF
        # //////////////////////////
        pfi = float(np.squeeze(pfi))  # Convert to scalar
        log_pfi = float(np.squeeze(log_pfi))

        return -log_pfi   # log_pfi, pfi


    # 1. Expected Improvement ---
    def expected_improvement(x, surrogate_gp, y_best, x_i):
        """
        Computes the Expected Improvement at a point x.
        Scalarizes the surrogate predictions using Tchebycheff, then computes EI.

        Note that the surrogate GP already has the weights applied to it
        """
        x = np.array(x).reshape(1, -1)

        mu, sigma = surrogate_gp.predict(x, return_std=True)

        # print(f'mu: {mu}')
        # print(f'y_best: {y_best}')
        # Compute EI

        xi = 0.2 # the greater the value of xi, the more we encourage exploration
        with np.errstate(divide='warn'):
            Z = ( mu - y_best - xi) / sigma
            ei = np.abs(mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z) * x_i
            ei[sigma == 0.0] = 0.0

        return -ei[0]  # Negative for minimization

        # 2. Probability of Imporovement:
    def probability_of_improvement(x, surrogate_gp, y_best, xi=0.005):
        """
        Computes the Probability of Improvement (PI) acquisition function.

        Parameters:
        - mu: np.array of predicted means (shape: [n_samples])
        - sigma: np.array of predicted std deviations (shape: [n_samples])
        - best_f: scalar, best objective value observed so far
        - xi: float, small value to balance exploration/exploitation (default: 0.01)

        Returns:
        - PI: np.array of probability of improvement values
        """
        x = np.array(x).reshape(1, -1)

        mu, sigma = surrogate_gp.predict(x, return_std=True)

        # Avoid division by zero
        if sigma == 0:
            sigma = 1e-8

        z = (y_best - mu - xi) / sigma

        pi = 1 - norm.cdf(z)

        return -pi
    
    
    def constrained_BO(
        optimization_budget, bounds, all_initial_inputs, all_initial_ouputs,
        job_max_or_min, constraint_thresholds, xi
    ):
        # xi = exploration parameter (the larger it is, the more we explore)

        # --- INITIALIZATION ---
        f1_vals = all_initial_ouputs[:, 0]
        f2_vals = all_initial_ouputs[:, 1]
        c1_vals = all_initial_ouputs[:, 2]
        c2_vals = all_initial_ouputs[:, 3]

        population = all_initial_inputs
        all_inputs = all_initial_inputs  # [m1, m2, m3, m4, t_index]

        population_all = []
        all_constraint_1_gps = []
        all_constraint_2_gps = []
        ei_all = []

        for gen in range(optimization_budget):
            print(f"\n\n Iteration {gen+1}")

            # --- SCALARIZATION ---
            lam = np.random.rand()
            weights = [lam, 1 - lam]
            phi = 0.05
            scalarized_f_vals = (
                np.maximum(weights[0] * f1_vals, weights[1] * f2_vals)
                + phi * (weights[0] * f1_vals + weights[1] * f2_vals)
            )
            scalarized_f_vals = weights[0] * f1_vals + weights[1] * f2_vals

            # --- FIT GPs ---
            scalarized_surrogate_gp = surrogate_model(population, scalarized_f_vals)
            scalarized_surrogate_gp_mean, scalarized_surrogate_gp_std = scalarized_surrogate_gp.predict(
                population, return_std=True
            )
            y_best = np.max(scalarized_surrogate_gp_mean)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # print("Glu Raff Purity")
                constraint_1_gp = surrogate_model(population, c1_vals)
                all_constraint_1_gps.append(constraint_1_gp)

                # print("Fru Ext Purity")
                constraint_2_gp = surrogate_model(population, c2_vals)
                all_constraint_2_gps.append(constraint_2_gp)

            # --- CONSTRAINTS ---
            eps = 0.01
            small_value = 1e-6

            def get_safe_tindex(x):
                return max(x[-1] * 60 * t_reff, small_value)

            def constraint_m1_gt_m2(x): return x[0] - (1 + eps) * x[1]
            def constraint_m1_gt_m4(x): return x[0] - (1 + eps) * x[3]
            def constraint_m2_lt_m1(x): return (1 - eps) * x[0] - x[1]
            def constraint_m2_lt_m3(x): return (1 - eps) * x[2] - x[1]
            def constraint_m3_gt_m2(x): return x[2] - (1 + eps) * x[1]
            def constraint_m3_gt_m4(x): return x[2] - (1 + eps) * x[3]
            def constraint_m4_lt_m1(x): return (1 - eps) * x[0] - x[3]
            def constraint_m4_lt_m3(x): return (1 - eps) * x[2] - x[3]

            def constraint_feed_pump_upper(x): return m_diff_max - (x[2] - x[1]) / get_safe_tindex(x)
            def constraint_feed_pump_lower(x): return (x[2] - x[1]) / get_safe_tindex(x) - m_diff_min
            def constraint_desorb_pump_upper(x): return m_diff_max - (x[0] - x[3]) / get_safe_tindex(x)
            def constraint_desorb_pump_lower(x): return (x[0] - x[3]) / get_safe_tindex(x) - m_diff_min
            def constraint_raff_pump_upper(x): return m_diff_max - (x[2] - x[3]) / get_safe_tindex(x)
            def constraint_raff_pump_lower(x): return (x[2] - x[3]) / get_safe_tindex(x) - m_diff_min
            def constraint_extract_pump_upper(x): return m_diff_max - (x[0] - x[1]) / get_safe_tindex(x)
            def constraint_extract_pump_lower(x): return (x[0] - x[1]) / get_safe_tindex(x) - m_diff_min

            nonlinear_constraints = [
                NonlinearConstraint(constraint_m1_gt_m2, 0, np.inf),
                NonlinearConstraint(constraint_m1_gt_m4, 0, np.inf),
                NonlinearConstraint(constraint_m2_lt_m1, 0, np.inf),
                NonlinearConstraint(constraint_m2_lt_m3, 0, np.inf),
                NonlinearConstraint(constraint_m3_gt_m2, 0, np.inf),
                NonlinearConstraint(constraint_m3_gt_m4, 0, np.inf),
                NonlinearConstraint(constraint_m4_lt_m1, 0, np.inf),
                NonlinearConstraint(constraint_m4_lt_m3, 0, np.inf),
                NonlinearConstraint(constraint_feed_pump_upper, 0, np.inf),
                NonlinearConstraint(constraint_feed_pump_lower, 0, np.inf),
                NonlinearConstraint(constraint_desorb_pump_upper, 0, np.inf),
                NonlinearConstraint(constraint_desorb_pump_lower, 0, np.inf),
                NonlinearConstraint(constraint_raff_pump_upper, 0, np.inf),
                NonlinearConstraint(constraint_raff_pump_lower, 0, np.inf),
                NonlinearConstraint(constraint_extract_pump_upper, 0, np.inf),
                NonlinearConstraint(constraint_extract_pump_lower, 0, np.inf),
            ]

            # --- VALIDATION HELPERS ---
            def passes_manual_check(vec):
                m1, m2, m3, m4 = vec[0:4]
                return (m1 > m2 and m3 > m2 and m4 < m3 and m4 < m1)

            # def is_similar_to_previous(x_candidate, X_history, tol=1e-3):
            #     """
            #     Check if x_candidate is too similar to any vector in X_history.
            #     Comparison is done after rounding to 3 decimal places.

            #     Parameters
            #     ----------
            #     x_candidate : array-like
            #         Candidate vector to check.
            #     X_history : array-like
            #         Array of previously evaluated candidates.
            #     tol : float, optional
            #         Distance threshold below which the candidate is considered similar.

            #     Returns
            #     -------
            #     bool
            #         True if candidate is similar to ANY previous candidate.
            #     """
            #     if X_history is None or len(X_history) == 0:
            #         return False

            #     # Convert to numpy arrays
            #     x_candidate = np.asarray(x_candidate)
            #     X_history = np.asarray(X_history)

            #     # Round to 3 decimal places before comparison
            #     x_candidate = np.round(x_candidate, 3)
            #     X_history = np.round(X_history, 3)

            #     # Normalize for consistent scaling
            #     mins = X_history.min(axis=0)
            #     maxs = X_history.max(axis=0)
            #     ranges = np.clip(maxs - mins, 1e-12, np.inf)
            #     normalized_candidate = (x_candidate - mins) / ranges
            #     normalized_history = (X_history - mins) / ranges

            #     # Compute distance to each previous point (pairwise)
            #     dists = np.linalg.norm(normalized_history - normalized_candidate, axis=1)

            #     # Return True if candidate is close to ANY previous point
            #     return np.any(dists < tol)
            def is_similar_to_previous(x_candidate, X_history, t_reff, similarity_threshold=0.02):
                """
                Check if x_candidate is too similar to any vector in X_history.
                The comparison is done after rounding to 3 decimal places.
                If the absolute difference between the candidate and a previous
                selection is less than 2% of that historical value in *all dimensions*,
                the candidate is considered similar (fails the test).

                Parameters
                ----------
                x_candidate : array-like
                    Candidate vector to check.
                X_history : array-like
                    Array of previously evaluated candidates.
                
                    similarity_threshold : float, optional
                    Relative difference threshold below which the candidate is considered similar.
                    0.02 = 2% difference.

                Returns
                -------
                bool
                    True if candidate is similar to ANY previous candidate (fails similarity test).
                """
                if X_history is None or len(X_history) == 0:
                    return False

                # Convert to numpy arrays
                x_cand = x_candidate.copy()
                x_cand[-1] = x_cand[-1] * t_reff  # Scale time component
                x_cand = np.round(np.asarray(x_cand), 3)
                X_history = np.round(np.asarray(X_history), 3)

                # Compare candidate to each historical point
                for prev in X_history:
                    # Avoid division by zero by adding a small epsilon
                    epsilon = 1e-12
                    relative_diff = np.abs(x_cand - prev) / (np.abs(prev) + epsilon)
                    # print("Relative differences:", relative_diff)

                    # Check if all dimensions differ by less than 1%
                    if np.all(relative_diff <= similarity_threshold):
                        return True  # candidate too similar

                return False  # candidate is sufficiently unique

            def generate_unique_candidate(bounds, X_history, max_attempts=200):
                bounds = np.array(bounds)
                lows, highs = bounds[:, 0], bounds[:, 1]
                for _ in range(max_attempts):
                    candidate = np.random.uniform(lows, highs)
                    if not is_similar_to_previous(candidate, X_history, t_reff, similarity_threshold):
                        return candidate
                print("Warning: Could not find unique candidate after several attempts.")
                return np.random.uniform(lows, highs)

            # --- OPTIMIZATION LOOP ---
            # print("Maxing ECI")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = differential_evolution(
                    func=log_expected_constrained_improvement,
                    bounds=bounds,
                    args=(
                        scalarized_surrogate_gp,
                        [constraint_1_gp, constraint_2_gp],
                        constraint_thresholds,
                        y_best,
                        job_max_or_min,
                        PF_weight,
                        xi,
                    ),
                    strategy='best1bin',
                    maxiter=600,
                    popsize=30,
                    disp=False,
                    constraints=(nonlinear_constraints),
                )

                x_candidate = result.x  # [m1, m2, m3, m4, t_index_min]

                # print(f'gen: {gen}')
                
                # 1. Check for the Uniquness of the new candidate
                # print("Checking similarity to previous samples...")
                if is_similar_to_previous(x_candidate, all_inputs, t_reff):
                    print("Candidate too similar to previous. Generating a new one...")
                    x_candidate = generate_unique_candidate(bounds, all_inputs)
                    while is_similar_to_previous(x_candidate, all_inputs, t_reff) == True:
                        x_candidate = generate_unique_candidate(bounds, all_inputs)
                    print(f"New candidate: {x_candidate} is sufficiently unique")

                # 2. Check if candidate is valid according to pattern
                if passes_manual_check(x_candidate):
                    x_new = x_candidate
                else:
                    print("Tweaking vector to satisfy pattern...")
                    x_new = x_candidate.copy().tolist()
                    m1, m2, m3, m4 = x_new[0:4]

                    if not (m1 > m2*1.1): m1 = m2 + abs(m2) * 0.3 + 1e-6
                    if not (m3 > m2*1.1): m3 = m2 + abs(m2) * 0.1 + 1e-6
                    if not (m4 < m3*0.9): m4 = m3 - abs(m3) * 0.1 - 1e-6
                    if not (m4 < m1*0.9): m4 = min(m4, m1 - abs(m1) * 0.1 - 1e-6)

                    x_new[0:4] = [m1, m2, m3, m4]
                    x_new = np.array(x_new)
                    print(f"Adjusted candidate: {x_new}")

                # 3. If passes all checks, evaluate and store the candidate
                x_new[-1] = x_new[-1] * t_reff
                f_new, c_new, mj_and_t_new = obj_con(x_new)

                all_inputs = np.vstack((all_inputs, mj_and_t_new))
                population_all.append(population)
                population = np.vstack((population, x_new))

                f1_vals = np.vstack([f1_vals.reshape(-1, 1), f_new[0]])
                f2_vals = np.vstack([f2_vals.reshape(-1, 1), f_new[1]])
                c1_vals = np.vstack([c1_vals.reshape(-1, 1), c_new[0]])
                c2_vals = np.vstack([c2_vals.reshape(-1, 1), c_new[1]])

                print(
                    f"Inputs:{x_new[:-1]}, "
                    f"{x_new[-1]} min [m1, m2, m3, m4, t_index]|\n"
                    f"Outputs: G_f1: {f_new[0]*100} %, F_f2: {f_new[1]*100} % | "
                    f"GPur, FPur: {c_new[0]*100}%, {c_new[1]*100}%"
                )

        rec_raff_vals, rec_ext_vals, pur_raff_vals, pur_ext_vals = f1_vals, f2_vals, c1_vals, c2_vals
        return rec_raff_vals, rec_ext_vals, pur_raff_vals, pur_ext_vals, all_inputs


    def load_initial_samples(json_inputs_path, json_outputs_path):
        """
        Loads previous SMB optimization samples from JSON for warm-starting RC-BO.

        Returns
        -------
        all_initial_inputs  : (N, 5) array  → [m1, m2, m3, m4, t_index_min]
        all_initial_outputs : (N, 4) array  → [rec_raff, rec_ext, pur_raff, pur_ext]
        inputs_dict         : full saved dict (for reference)
        """

        with open(json_inputs_path, "r") as f:
            inputs_dict = json.load(f)


        with open(json_outputs_path, "r") as f:
            data_dict = json.load(f)

        # Extract inputs
        m1 = np.array(inputs_dict["m1"])
        m2 = np.array(inputs_dict["m2"])
        m3 = np.array(inputs_dict["m3"])
        m4 = np.array(inputs_dict["m4"])
        t_index = np.array(inputs_dict["t_index_min"])


        all_initial_inputs = np.vstack([m1, m2, m3, m4, t_index]).T

        # Extract outputs
        rec_raff = np.array(data_dict["rec_raff_vals"])
        rec_ext  = np.array(data_dict["rec_ext_vals"])
        pur_raff = np.array(data_dict["pur_raff_vals"])
        pur_ext  = np.array(data_dict["pur_ext_vals"])

        rec_raff = np.squeeze(rec_raff)
        rec_ext  = np.squeeze(rec_ext)
        pur_raff = np.squeeze(pur_raff)
        pur_ext  = np.squeeze(pur_ext)
 

        all_initial_outputs = np.vstack([rec_raff, rec_ext, pur_raff, pur_ext]).T

        return all_initial_inputs, all_initial_outputs, inputs_dict

        
        
    
    Q_max = Q_max/3.6 # l/h => ml/s
    Q_min = Q_min/3.6 # l/h => ml/s
    # SUMMARY
    print(f'\n\n OPTIMIZATION INPUTS SUMMARY: \n')
    print(f'Column Volume: {V_col} cm^3 | {V_col/1000} L')
    print(f'Column CSA: {A_col} cm^2')
    print(f'Column Length: {L} cm')
    print(f'')
    if grouping_type == []:
        print(f'Configuration: {zone_config}, No grouping')
    else:
        print(f"Configuration: {zone_config}, With Grouping")
    print(f'Column Diameter: {d_col} cm')
    print(f'Optimization Budget: {optimization_budget}')
    print(f'pF_weight: {PF_weight}')
    print(f'exploration (xi): {x_i}')
    print(f'thresholds: {constraint_threshold}')
    print(f'[Q_max, Q_min] = [{Q_max*3.6}, {Q_min*3.6}] L/h')
    print(f"bounds:\nm1: ({bounds[0][0]}, {bounds[0][1]})\nm2: ({bounds[1][0]}, {bounds[1][1]})\nm3: ({bounds[2][0]}, {bounds[2][1]})\nm4: ({bounds[3][0]}, {bounds[3][1]})\nt_index: ({bounds[4][0]*t_reff}, {bounds[4][1]*t_reff}) min")
        
#%%
    # generate iniital samples
    start_test = time.time()
    all_initial_inputs, all_initial_outputs = generate_initial_data(triangle_guess)
    end_test = time.time()
    test_duration = end_test-start_test
    print(f'----------------------')
    print(f'\nGenerated Initial Samples in {test_duration/60} min')
    print(f'----------------------')

    # import json
    # import numpy as np
    # import os
    # # Pre-loading steps if optimization is done:    
    # # 1. Load j.son file with the all_inputs disct from the previous optimization run
    # # all_inputs location = 'SMB_OPT_inputs_example.json'
    # json_inputs_path = r"C:\Users\nawau\OneDrive\Desktop\MEng_Code\case0_Literature\Feed_Rec_Obj\3333NORMGlu_Fru_commision_opt_51iter_config_all_input_20251025_213827.json"
    # json_outputs_path = r"C:\Users\nawau\OneDrive\Desktop\MEng_Code\case0_Literature\Feed_Rec_Obj\3333NORMGlu_Fru_commision_opt_51iter_config_all_output_20251025_213827.json"
    # all_initial_inputs, all_initial_outputs, inputs_dict = load_initial_samples(json_inputs_path, json_outputs_path)

    # # 2. Store in optimization numpy arrays: all_initial_inputs, all_initial_outputs
    
    # # print(f'\n\n Generated Initial Samples in {test_duration/60} min')
    # print(f'all_initial_inputs (size: {np.shape(all_initial_inputs)})\n{ all_initial_inputs}')
    # print(f'all_initial_outputs (size: {np.shape(all_initial_outputs)})\n{ all_initial_outputs}')

#%%
    # OPTIMIZATION
    
    rec_raff_vals, rec_ext_vals, pur_raff_vals, pur_ext_vals, all_inputs  = constrained_BO(optimization_budget, bounds, all_initial_inputs, all_initial_outputs, job_max_or_min, constraint_threshold, x_i)

#%%
    # ----------- SAVE
    # Get snapshot of results in console:
    output_check = {np.concatenate((rec_raff_vals, rec_ext_vals, pur_raff_vals, pur_ext_vals), axis=1)}
    print(f'\n\nSUMMARY MATRICES:\nInputs Matrix\ncolumns:[m1, m2, m3, m4, t_index (min)]:\n {all_inputs}')
    print(f'Outputs Matrix\ncolumns:[Raff_rec, Ext_rec, Raff_pur, Ext_pur]:\n {output_check}')

    # Inputs:
    all_inputs_dict = {
        "Description": Description,
        "m1":all_inputs[:,0].tolist(),
        "m2":all_inputs[:,1].tolist(),
        "m3":all_inputs[:,2].tolist(),
        "m4":all_inputs[:,3].tolist(),
        "t_index_min":all_inputs[:,4].tolist(),
        
        "Q1_(L/h)":((3.6*all_inputs[:,0]*V_col*(1-e) + V_col*e)/(t_index_min*60)).tolist(),
        "Q2_(L/h)":((3.6*all_inputs[:,1]*V_col*(1-e) + V_col*e)/(t_index_min*60)).tolist(),
        "Q3_(L/h)":((3.6*all_inputs[:,2]*V_col*(1-e) + V_col*e)/(t_index_min*60)).tolist(),
        "Q4_(L/h)":((3.6*all_inputs[:,3]*V_col*(1-e) + V_col*e)/(t_index_min*60)).tolist(),

        "V_col_(mL)":[V_col],
        "L_col_(cm)": [L],
        "A_col_(cm)": [A_col],
        "d_col_(cm)": [d_col],
        "config": zone_config.tolist(),
        "e":[e]

    }

    # Outputs:
    data_dict = {
        "Description": Description,

        "rec_raff_vals": rec_raff_vals.tolist(),
        "rec_ext_vals": rec_ext_vals.tolist(),

        "pur_raff_vals": pur_raff_vals.tolist(),
        "pur_ext_vals": pur_ext_vals.tolist(),
    }



    import json
    from datetime import datetime

    # === Create date-time stamp ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Append timestamp to filenames
    save_name_inputs = f"{save_name_inputs.rstrip('.json')}_{timestamp}.json"
    save_name_outputs = f"{save_name_outputs.rstrip('.json')}_{timestamp}.json"

    # === SAVE all_inputs to JSON ===
    with open(save_name_inputs, "w") as f:
        json.dump(all_inputs_dict, f, indent=4)

    # === SAVE recoveries_and_purities to JSON ===
    with open(save_name_outputs, "w") as f:
        json.dump(data_dict, f, indent=4)

    print(f"OPTIMIZATION COMPLETE")
    print(f"Files saved successfully:")
    print(f" - Inputs:  {save_name_inputs}")
    print(f" - Outputs: {save_name_outputs}")

    return (data_dict, all_inputs_dict)



    # %%
    



# # call the respective input files