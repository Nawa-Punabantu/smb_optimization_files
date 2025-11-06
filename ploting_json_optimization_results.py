#%%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, NonlinearConstraint
import json
from matplotlib.ticker import MaxNLocator, MultipleLocator

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy import integrate
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt

def find_pareto_front(x_vals, y_vals):
    """
    Identify Pareto-optimal points for maximization problems.
    Returns a boolean mask where True = point is on Pareto frontier.
    """
    points = np.vstack((x_vals, y_vals)).T
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if np.any((points[:, 0] >= p[0]) & (points[:, 1] > p[1])) or np.any((points[:, 0] > p[0]) & (points[:, 1] >= p[1])):
            is_pareto[i] = False
    return is_pareto


def plot_raff_ext_pareto(inputs_path, outputs_path, Names, purity_constraints, xy_padding = 1):
    """
    Plots Pareto frontiers for raffinate (Glucose) and extract (Fructose).
    Also reports:
      - Optimal points per component
      - Dual optimal points (in both fronts)
      - Feasible points (purity >= constraint)
      - Corresponding input variables (m1–m4, t_index_min)
    """

    # --- UNPACK DATA ---
    print("starting to unpack")
    inputs_dict, data_dict = load_inputs_outputs(inputs_path, outputs_path)
    print("done loading")
    comp_1_name= Names[0]
    comp_2_name= Names[1]
    # print(f'data_dict: {data_dict')
    # print(f'inputs_dict: {inputs_dict')
    purity_constraints = [purity_constraints[i]*100 for i in range(len(Names))]
    f1_vals = np.array(data_dict["rec_raff_vals"])  * 100  # Glucose recovery   (%)
    f2_vals = np.array(data_dict["rec_ext_vals"])   * 100   # Fructose recovery (%)
    c1_vals = np.array(data_dict["pur_raff_vals"])  * 100  # Glucose purity     (%)
    c2_vals = np.array(data_dict["pur_ext_vals"])   * 100   # Fructose purity   (%)

    iter_num = len(f1_vals)
    iter_range = np.arange(iter_num)

    # print(f'inputs_dict: {inputs_dict[0]}')
    m1 = np.array(inputs_dict["m1"])
    m2 = np.array(inputs_dict["m2"])
    m3 = np.array(inputs_dict["m3"])
    m4 = np.array(inputs_dict["m4"])
    t_index = np.array(inputs_dict["t_index_min"])
    # print(f'shape m1: {np.shape(m1)}')


    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].plot(iter_range, m1, label='m1')
    axs[0].scatter(iter_range, m1)


    axs[0].plot(iter_range, m2, label='m2')
    axs[0].scatter(iter_range, m2)

    axs[0].plot(iter_range, m3, label='m3')
    axs[0].scatter(iter_range, m3)

    axs[0].plot(iter_range, m4, label='m4')
    axs[0].scatter(iter_range, m4)
    axs[0].legend()
    axs[0].set_xlabel(f"Iteration", fontsize=14)
    axs[0].set_ylabel(f"Flowrate Ratio", fontsize=14)
    axs[0].set_title("Flowrate Ratio Trajectory", fontsize=14)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=iter_num))


    axs[0].grid(True)
    

    axs[1].plot(iter_range, t_index, label='Indexing Time (min)')
    axs[1].scatter(iter_range, t_index)
    axs[1].set_xlabel(f"Iteration", fontsize=14)
    axs[1].set_ylabel(f"Indexing Time (min)", fontsize=14)
    axs[1].set_title("Indexing Time Trajectory", fontsize=14)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=iter_num))

    axs[1].grid(True)

    # plt.title('Optimization Input Variables over Iterations', fontsize=16)

    plt.xlabel('Iteration', fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    # plt.plot(iter_range, t_index, label='t_index_min')

    # --- PARETO FRONT MASKS ---
    comp1_mask = find_pareto_front(f1_vals, c1_vals)
    comp2_mask = find_pareto_front(f2_vals, c2_vals)
    both_mask = comp1_mask & comp2_mask

    # -----------------------------------------------------------
    # ---- PLOTTING 2: 
    fig, axs = plt.subplots(1, 2, figsize=(15,5))


    axs[0].scatter(iter_range, f1_vals, color='red', label=f'Raffiante Recovery (%)')
    axs[0].plot(iter_range, f1_vals, color='red')

    axs[0].scatter(iter_range, f2_vals, color='blue', label=f'Extract Recovery (%)')
    axs[0].plot(iter_range, f2_vals, color='blue')


    axs[0].set_xlabel(f"Iteration", fontsize=14)
    axs[0].set_ylabel(f"Recovery (%)", fontsize=14)
    axs[0].set_title("Recovery Trade-off Trajectories", fontsize=14)
    axs[0].grid(True)
    axs[0].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=iter_num))
    
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].legend()

    axs[1].scatter(iter_range, c1_vals, color='orange', label=f'Raffiante Purity (%)')
    axs[1].plot(iter_range, c1_vals, color='orange')


    axs[1].scatter(iter_range, c2_vals, color='green', label=f'Extract Purity (%)')
    axs[1].plot(iter_range, c2_vals, color='green')

    axs[1].set_xlabel(f"Iteration", fontsize=14)
    axs[1].set_ylabel(f"Purity (%)", fontsize=14)
    axs[1].set_title("Purity Trade-off Trajectories", fontsize=14)
    axs[1].grid(True)
    axs[1].axhline(purity_constraints[0], color='k', linestyle='--', linewidth=1.5,
                   label=f'{Names[0]} Constraint: {purity_constraints[0]:.1f}%')
    
    axs[1].axhline(purity_constraints[1], color='k', linestyle='--', linewidth=1.5,
                   label=f'{Names[1]} Constraint: {purity_constraints[1]:.1f}%')

    axs[1].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=iter_num))
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].legend()
    plt.show()




    # ---PLOTTING 2

    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Raffinate (Glucose)
    # Normalize indices to [0, 1] for colormap mapping
    num_points = len(f1_vals)
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, num_points))  # 0.3 to 0.9 avoids pure white/black

    # Plot with grayscale gradient
    axs[0].scatter(f1_vals, f2_vals, color=colors)

    axs[0].set_xlabel(f"{comp_1_name} Raffinate Recovery (%)", fontsize=14)
    axs[0].set_ylabel(f"{comp_2_name} Extract Recovery (%)", fontsize=14)
    axs[0].set_title("Recovery Trade-off", fontsize=14)
    axs[0].grid(True)

    
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_xlim(0 - xy_padding, 100 + xy_padding)
    axs[0].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[0].legend()


    # Normalize indices to [0, 1] for colormap mapping
    num_points = len(f1_vals)
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, num_points))  # 0.3 to 0.9 avoids pure white/black

    # Plot with grayscale gradient
    axs[1].scatter(c1_vals, c2_vals, color=colors)
    axs[1].axvline(purity_constraints[0], color='k', linestyle='--', linewidth=1.5,
                   label=f'{Names[0]} Constraint: {purity_constraints[1]:.1f}%')
    
    axs[1].axhline(purity_constraints[1], color='k', linestyle='--', linewidth=1.5,
                label=f'{Names[1]} Constraint: {purity_constraints[1]:.1f}%')
    
    axs[1].set_xlabel(f"{comp_1_name} Raffinate Purity (%)", fontsize=14)
    axs[1].set_ylabel(f"{comp_2_name} Extract Purity (%)", fontsize=14)
    axs[1].set_title("Purity Trade-off", fontsize=14)
    axs[1].grid(True)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_xlim(0 - xy_padding, 100 + xy_padding)
    axs[1].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[1].legend()
    plt.show()

    # initalize:
    pur_sorted = c1_vals[comp1_mask].copy()
    rec_sorted = f1_vals[comp1_mask].copy()
    # print(f'pur_sorted before sort: {pur_sorted}')
    # print(f'rec_sorted before sort: {rec_sorted}')
    # get sorted indeceis
    idx_sort = np.argsort(pur_sorted, axis=0)

    # print(f'idx_sort_ax[0]: {idx_sort}')

    # apply
    pur_sorted = np.squeeze(pur_sorted[idx_sort])
    rec_sorted = np.squeeze(rec_sorted[idx_sort])
    # print(f'pur_sorted after sort: {pur_sorted}')
    # print(f'rec_sorted after sort: {rec_sorted}')

    axs[0].plot(pur_sorted, rec_sorted, color='red', linestyle='--', alpha=0.5,label="Trend Line")

    axs[0].set_title(f"{comp_1_name} (Raffinate) Recovery vs Purity", fontsize=14)
    axs[0].set_xlabel(f"{comp_1_name} Purity (%)", fontsize=14)
    axs[0].set_ylabel(f"{comp_1_name} Recovery (%)", fontsize=14)
    axs[0].grid(True)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_xlim(0 - xy_padding, 100 + xy_padding)
    axs[0].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[0].legend()






    # --- PRINT SUMMARY INFO ---
    print("\n===== Pareto Front Summary =====")
    print(f"{comp_1_name} Pareto-optimal points: {np.sum(comp1_mask)}")
    print(f"{comp_2_name} Pareto-optimal points: {np.sum(comp2_mask)}")
    print(f"Dual-optimal points (both fronts): {np.sum(both_mask)}\n")

    # --- FEASIBILITY TEST (Purity ≥ threshold) ---
    feasible_mask = (c1_vals >= purity_constraints[0]) & (c2_vals >= purity_constraints[0])
    print(f"Feasible points (Purity ≥ {purity_constraints[0]}%): {np.sum(feasible_mask)}\n")

    # --- REPORT FEASIBLE POINTS IN TABLE ---
    if np.sum(feasible_mask) > 0:
        feasible_table = {
            "m1": np.array(inputs_dict["m1"])[feasible_mask],
            "m2": np.array(inputs_dict["m2"])[feasible_mask],
            "m3": np.array(inputs_dict["m3"])[feasible_mask],
            "m4": np.array(inputs_dict["m4"])[feasible_mask],
            "t_index_min": np.array(inputs_dict["t_index_min"])[feasible_mask],
            f"{comp_1_name}_Rec(%)": f1_vals[feasible_mask],
            f"{comp_1_name}_Pur(%)": c1_vals[feasible_mask],
            f"{comp_2_name}_Rec(%)": f2_vals[feasible_mask],
            f"{comp_2_name}_Pur(%)": c2_vals[feasible_mask],
        }

        # Include flow rates if present
        for j in range(1, 5):
            key = f"Q{j}_(L/h)"
            if key in inputs_dict:
                feasible_table[key] = np.array(inputs_dict[key])[feasible_mask]

        print("Feasible Points Summary:")
        print(f"{'m1':>6} {'m2':>6} {'m3':>6} {'m4':>6} {'t_idx':>8} | "
              f"{comp_1_name[:3]}_Rec  {comp_1_name[:3]}_Pur  {comp_2_name[:3]}_Rec  {comp_2_name[:3]}_Pur")
        print("-" * 80)
        for i in range(np.sum(feasible_mask)):
            print(f"{feasible_table['m1'][i]:6.3f} {feasible_table['m2'][i]:6.3f} {feasible_table['m3'][i]:6.3f} "
                  f"{feasible_table['m4'][i]:6.3f} {feasible_table['t_index_min'][i]:8.3f} | "
                  f"{feasible_table[f'{comp_1_name}_Rec(%)'][i]:7.2f} {feasible_table[f'{comp_1_name}_Pur(%)'][i]:8.2f} "
                  f"{feasible_table[f'{comp_2_name}_Rec(%)'][i]:8.2f} {feasible_table[f'{comp_2_name}_Pur(%)'][i]:8.2f}")
        print("-" * 80)

    # --- PLOTTING ---
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Raffinate (Glucose)
    axs[0].scatter(c1_vals[~comp1_mask], f1_vals[~comp1_mask], color='lightgray', label="Non-Pareto")
    axs[0].scatter(c1_vals[comp1_mask], f1_vals[comp1_mask], color='red', facecolors='none', marker='o',
                   label=f"{comp_1_name} Pareto Frontier")
    axs[0].scatter(c1_vals[both_mask], f1_vals[both_mask], color='red', marker='o', label="Dual Optimal")
    axs[0].axvline(purity_constraints[0], color='k', linestyle='--', linewidth=1.5,
                   label=f'{Names[0]} Constraint: {purity_constraints[0]:.1f}%')
    


    # initalize:
    pur_sorted = c1_vals[comp1_mask].copy()
    rec_sorted = f1_vals[comp1_mask].copy()
    # print(f'pur_sorted before sort: {pur_sorted}')
    # print(f'rec_sorted before sort: {rec_sorted}')
    # get sorted indeceis
    idx_sort = np.argsort(pur_sorted, axis=0)

    # print(f'idx_sort_ax[0]: {idx_sort}')

    # apply
    pur_sorted = np.squeeze(pur_sorted[idx_sort])
    rec_sorted = np.squeeze(rec_sorted[idx_sort])
    # print(f'pur_sorted after sort: {pur_sorted}')
    # print(f'rec_sorted after sort: {rec_sorted}')

    axs[0].plot(pur_sorted, rec_sorted, color='red', linestyle='--', alpha=0.5,label="Trend Line")

    axs[0].set_title(f"{comp_1_name} (Raffinate) Recovery vs Purity", fontsize=14)
    axs[0].set_xlabel(f"{comp_1_name} Purity (%)", fontsize=14)
    axs[0].set_ylabel(f"{comp_1_name} Recovery (%)", fontsize=14)
    axs[0].grid(True)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_xlim(0 - xy_padding, 100 + xy_padding)
    axs[0].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[0].legend()

    # Extract (Fructose)
    axs[1].scatter(c2_vals[~comp2_mask], f2_vals[~comp2_mask], color='lightgray', label="Non-Pareto")
    axs[1].scatter(c2_vals[comp2_mask], f2_vals[comp2_mask], color='green', facecolors='none', marker='o',
                   label=f"{comp_2_name} Pareto Frontier")
    axs[1].scatter(c2_vals[both_mask], f2_vals[both_mask], color='green', marker='o', label="Dual Optimal")
    axs[1].axvline(purity_constraints[1], color='k', linestyle='--', linewidth=1.5,
                   label=f'{Names[1]} Constraint: {purity_constraints[1]:.1f}%')

    # initalize:
    pur_sorted = c2_vals[comp2_mask].copy()
    rec_sorted = f2_vals[comp2_mask].copy()
    # print(f'pur_sorted before sort: {pur_sorted}')
    # print(f'rec_sorted before sort: {rec_sorted}')

    # get sorted indeceis
    idx_sort = np.argsort(pur_sorted, axis=0)
    # print(f'idx_sort_ax[1]: {idx_sort}')

    # apply
    pur_sorted = np.squeeze(pur_sorted[idx_sort])       
    rec_sorted = np.squeeze(rec_sorted[idx_sort])
    # print(f'pur_sorted after sort: {pur_sorted}')
    # print(f'rec_sorted after sort: {rec_sorted}')

    axs[1].plot(pur_sorted, rec_sorted, color='green', linestyle='--', alpha=0.5, label="Trend Line")

    axs[1].set_title(f"{comp_2_name} (Extract) Recovery vs Purity", fontsize=14)
    axs[1].set_xlabel(f"{comp_2_name} Purity (%)", fontsize=14)
    axs[1].set_ylabel(f"{comp_2_name} Recovery (%)", fontsize=14)
    axs[1].grid(True)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_xlim(0 - xy_padding, 100 + xy_padding)
    axs[1].set_ylim(0 - xy_padding, 100 + xy_padding)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    

    

    

#%% Run nd Define the Funcitons
def load_inputs_outputs(inputs_path, outputs_path):
    """
    Loads all_inputs and output values (f1, f2, c1, c2) from saved JSON files and reconstructs them as numpy arrays.
    
    Args:
        inputs_path (str): Path to 'all_inputs.json'.
        outputs_path (str): Path to 'all_outputs.json'.

    Returns:
        all_inputs (np.ndarray): Loaded inputs array.
        f1_vals (np.ndarray): Glucose recovery values.
        f2_vals (np.ndarray): Fructose recovery values.
        c1_vals (np.ndarray): Glucose purity values.
        c2_vals (np.ndarray): Fructose purity values.
    """
    # Load inputs
    with open(inputs_path, "r") as f:
        all_inputs_list = json.load(f)
    # all_inputs_list = np.array(all_inputs_list)

    # Load outputs
    with open(outputs_path, "r") as f:
        data_dict = json.load(f)

    return all_inputs_list, data_dict


#%%
#------------------------------------------------------- 1. Table

def create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget):
    # Create a data table with recoveries first
    data = np.column_stack((f1_vals*100, f2_vals*100, c1_vals*100, c2_vals*100))
    columns = ['Recovery F1 (%)', 'Recovery F2 (%)', 'Purity C1 (%)', 'Purity C2 (%)']
    rows = [f'Iter {i+1}' for i in range(len(c1_vals))]

    # Identify "star" entries (where f1_vals, f2_vals > 70 and c1_vals, c2_vals > 90)
    # star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 95))[0]
    star_indices = np.where((c2_vals*100 > 99.5))[0]
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(c1_vals) * 0.2))
    # ax.set_title("Optimization Iterations: Recovery & Purity Table", fontsize=12, fontweight='bold', pad=5)  # Reduced padding
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data.round(2),
                     colLabels=columns,
                     rowLabels=rows,
                     cellLoc='center',
                     loc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.auto_set_column_width(col=list(range(len(columns))))

    # Apply colors
    for i in range(len(c1_vals)):
        for j in range(len(columns)):
            cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
            if i < sampling_budget:
                cell.set_facecolor('lightgray')  # Grey out first 20 rows
            if i in star_indices:
                cell.set_facecolor('yellow')  # Highlight star entries in yellow

    # Save the figure as an image
    image_filename = "output_optimization_table.png"
    fig.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename



def calculate_flowrates(input_array, V_col, e):
    # Initialize the external flowrate array with the same shape as input_array
    internal_flowrate = np.zeros_like(input_array[:,:-1])
    external_flowrate = np.zeros_like(input_array)
    
    # Reshape the last column to be a 2D array for broadcasting
    input_last_col = input_array[:, -1]
    
    for i, t_index in enumerate(input_last_col):
        # Calculate the flow rates using the provided formula
        # Fill each row in external_flowrate:
        print(f't_index: {t_index}')
        internal_flowrate[i, :] = (input_array[i, :-1] * V_col * (1 - e) + V_col * e) / (t_index * 60)  # cm^3/s
    

    internal_flowrate = internal_flowrate*3.6 # cm^3/s => L/h
    print(f'internal_flowrate: {internal_flowrate}')
    # Calculate Internal FLowtates:
    Qfeed = internal_flowrate[:,2] - internal_flowrate[:,1] # Q_III - Q_II 
    Qraffinate = internal_flowrate[:,2] - internal_flowrate[:,3] # Q_III - Q_IV 
    Qdesorbent = internal_flowrate[:,0] - internal_flowrate[:,3] # Q_I - Q_IV 
    Qextract = internal_flowrate[:,0] - internal_flowrate[:,1] # Q_I - Q_II

    external_flowrate[:,0] = Qfeed
    external_flowrate[:,1] = Qraffinate
    external_flowrate[:,2] = Qdesorbent
    external_flowrate[:,3] = Qextract
    external_flowrate[:,4] = input_last_col

    return internal_flowrate, external_flowrate

def create_input_optimization_table(input_array, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals):
    # Calculate flow rates
    internal_flowrate, external_flowrate = calculate_flowrates(input_array, V_col, e)
    flowrates = external_flowrate
    # Create a data table with flow rates
    data = external_flowrate
    columns = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract(L/h)', 'Index Time (min)']
    rows = [f'Iter {i+1}' for i in range(len(input_array))]

    # star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 95))[0]
    star_indices = np.where((c2_vals*100 > 95))[0]
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(input_array) * 0.2))
    # ax.set_title("Optimization Iterations: Flowrate Table", fontsize=12, fontweight='bold', pad=1)  # Reduced padding
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data.round(3),
                     colLabels=columns,
                     rowLabels=rows,
                     cellLoc='center',
                     loc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.auto_set_column_width(col=list(range(len(columns))))

    # Apply colors
    for i in range(len(input_array)):
        for j in range(len(columns)):
            cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
            if i < sampling_budget:
                cell.set_facecolor('lightgray')  # Grey out first sampling_budget rows
            if i in star_indices:
                cell.set_facecolor('yellow')  # Highlight star entries in yellow

    # Save the figure as an image
    image_filename = "input_optimization_table.png"
    fig.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename





#------------------------------------------------------- 2. Recovery Pareto

def create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget):
    # Convert to percentages
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(f1, f2):
        pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(f1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue
    plt.scatter(f1_vals_plot[~pareto_mask], f2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
    # Plot Pareto-optimal points in red
    plt.scatter(f1_vals_plot[pareto_mask], f2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

    # Plot initial samples in grey
    # plt.scatter(f1_initial, f2_initial, c='grey', marker='o', label='Initial Samples')

    # Labels and formatting
    plt.title(f'Pareto Curve of Recoveries\nGlucose in Raffinate vs Fructose in Extract\nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.xlabel('Glucose Recovery in Raffinate (%)', fontsize=12)
    plt.ylabel('Fructose Recovery in Extract (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)

    # Set x-axis limits with buffer to avoid clipping edge markers
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = "recovery_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


# ------ comparing multiple pareot
import numpy as np

def compare_pareto_similarity(f1_vals_100, f2_vals_100, f1_vals_20, f2_vals_20, tolerance_percent=5):
    """
    Compares Pareto fronts from 100-iteration and 20-iteration runs.
    Returns number and fraction of 20-iteration Pareto points within X% of any 100-iteration point.
    """

    # Convert to percent scale
    f1_100 = f1_vals_100 * 100
    f2_100 = f2_vals_100 * 100
    f1_20 = f1_vals_20 * 100
    f2_20 = f2_vals_20 * 100

    # Find Pareto masks
    pareto_mask_100 = find_pareto_front(f1_100, f2_100)
    pareto_mask_20 = find_pareto_front(f1_20, f2_20)

    pareto_100 = np.column_stack((f1_100[pareto_mask_100], f2_100[pareto_mask_100]))
    pareto_20 = np.column_stack((f1_20[pareto_mask_20], f2_20[pareto_mask_20]))

    count_within = 0

    for point in pareto_20:
        f1_p, f2_p = point
        for f1_ref, f2_ref in pareto_100:
            f1_close = abs(f1_p - f1_ref) <= (tolerance_percent / 100) * f1_ref
            f2_close = abs(f2_p - f2_ref) <= (tolerance_percent / 100) * f2_ref
            if f1_close and f2_close:
                count_within += 1
                break  # Move to next point in 20-front

    total_points = len(pareto_20)
    fraction_within = count_within / total_points if total_points > 0 else 0

    print(f"{count_within} out of {total_points} points in the 20-iteration Pareto front "
          f"are within {tolerance_percent}% of a point in the 100-iteration front "
          f"({fraction_within:.2%})")

    return count_within, total_points, fraction_within

def compare_recovery_pareto_plot(f1_vals_20, f2_vals_20, f1_vals_100, f2_vals_100, c1_vals_20, c2_vals_20, c1_vals_100, c2_vals_100):
    # Convert to percentages
    f1_vals_plot_100 = f1_vals_100 * 100
    f2_vals_plot_100 = f2_vals_100 * 100
    c1_vals_plot_100 = c1_vals_100 * 100
    c2_vals_plot_100 = c2_vals_100 * 100
    # -----
    f1_vals_plot_20 = f1_vals_20 * 100
    f2_vals_plot_20 = f2_vals_20 * 100
    c1_vals_plot_20 = c1_vals_20 * 100
    c2_vals_plot_20 = c2_vals_20 * 100



    # Function to find Pareto front
    def find_pareto_front(f1, f2):
        pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(f1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask_100 = find_pareto_front(f1_vals_plot_100, f2_vals_plot_100)
    pareto_mask_20 = find_pareto_front(f1_vals_plot_20, f2_vals_plot_20)

    plt.figure(figsize=(10, 6))

    # WE ARE ONLY INTERESTED ON COMPARING PARETO FRONTS
    # Compare Recovery
    plt.scatter(f1_vals_plot_100[pareto_mask_100], f2_vals_plot_100[pareto_mask_100], c='red', marker='o', label='100 iterations Recovery Pareto Frontier')
    plt.scatter(f1_vals_plot_20[pareto_mask_20], f2_vals_plot_20[pareto_mask_20], c='purple', marker='s', label='20 iterations Recovery Pareto Frontier')
    # Compare Purity
    plt.scatter(c1_vals_plot_100[pareto_mask_100], c2_vals_plot_100[pareto_mask_100], c='grey', marker='o', label='100 iterations Purity at Recovery Frontier')
    plt.scatter(c1_vals_plot_20[pareto_mask_20], c2_vals_plot_20[pareto_mask_20], c='grey', marker='s', label='20 iterations Purity at Recovery Frontier')
    # Labels and formatting
    plt.title(f'Comparison of Sampling Efficeiny\n{len(f2_vals_20)-1} vs  {len(f2_vals_100)-1} iterations')
    plt.xlabel('Glucose (%)', fontsize=12)
    plt.ylabel('Fructose (%)', fontsize=12)

    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)

    # Set x-axis limits with buffer to avoid clipping edge markers
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = "20_vs_100_recovery_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


#------------------------------------------------------- 2. Purity Pareto

def create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(c1, c2):
        pareto_mask = np.ones(len(c1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(c1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((c1 >= c1[i]) & (c2 >= c2[i]) & ((c1 > c1[i]) | (c2 > c2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(c1_vals_plot, c2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue
    plt.scatter(c1_vals_plot[~pareto_mask], c2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
    # Plot Pareto-optimal points in red
    plt.scatter(c1_vals_plot[pareto_mask], c2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

    # Plot initial samples in grey
    # plt.scatter(c1_initial, c2_initial, c='grey', marker='o', label='Initial Samples')

    # Labels and formatting
    plt.title(f'Pareto Curve of Purities\nGlucose in Raffinate vs Fructose in Extract\nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.xlabel('Glucose Purity in Raffinate (%)', fontsize=12)
    plt.ylabel('Fructose Purity in Extract (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)
    plt.grid(True)
    # plt.legend()

    # Save the figure as an image
    image_filename = "purity_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


def create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(c1, c2):
        pareto_mask = np.ones(len(c1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(c1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((c1 >= c1[i]) & (c2 >= c2[i]) & ((c1 > c1[i]) | (c2 > c2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    # Glucose
    pareto_mask_glu = find_pareto_front(c1_vals_plot, f1_vals_plot)
    #Fructose
    pareto_mask_fru = find_pareto_front(c2_vals_plot, f2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue

    plt.scatter(c1_vals_plot[~pareto_mask_glu], f1_vals_plot[~pareto_mask_glu], c='grey', marker='o', label='Glucose')
    plt.scatter(c2_vals_plot[~pareto_mask_fru], f2_vals_plot[~pareto_mask_fru], c='grey', marker='^', label='Fructose')

    plt.scatter(c1_vals_plot[pareto_mask_glu], f1_vals_plot[pareto_mask_glu], c='red', marker='o',label='Glucose Pareto Front')

    plt.scatter(c2_vals_plot[pareto_mask_fru], f2_vals_plot[pareto_mask_fru], c='orange', marker='^',label='Fructose Pareto Front')

    # Labels and formatting
    plt.title(f'Pareto Curves of {comp} Recovery vs Purity \nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.ylabel(f'Recovery (%)', fontsize=12)
    plt.xlabel(f'Purity (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = f"{comp}_recovery_vs_purity_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename



#------------------------------------------------------- 4. Pareto Outputs Trace
def find_pareto_front(f1, f2):
    pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

    for i in range(len(f1)):
        if pareto_mask[i]:  # Check only if not already removed
            pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

    return pareto_mask

def plot_inputs_vs_iterations(input_array, f1_vals, f2_vals):
    input_names = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract (L/h)', 'Index Time (min)']
    # Convert to percentages
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

    # Filter input_array for Pareto-optimal points
    pareto_inputs = input_array[pareto_mask]
    internal_flowrate, external_flowrate = calculate_flowrates(pareto_inputs, V_col, e)

    # Plot inputs vs iterations for Pareto-optimal points
    iterations = np.arange(1, len(external_flowrate) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot all inputs except the last one
    for i in range(external_flowrate.shape[1] - 1):
        ax1.plot(iterations, external_flowrate[:, i], marker='o', label=f'{input_names[i]}')

    ax1.set_xlabel('Position on Pareto Front (Left-to-Right)', fontsize=12)
    ax1.set_ylabel('Flowrates (L/h)', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Create a second y-axis for the indexing time
    ax2 = ax1.twinx()
    ax2.plot(iterations, external_flowrate[:, -1], marker='o', color='grey', linestyle = "--", label=f'Input {input_names[-1]}')
    ax2.set_ylabel('Index Time (min)')
    ax2.legend(loc='upper left', bbox_to_anchor=(2.05, 1.0), borderaxespad=0.)
    # Ensure integer ticks only (no half-values)
    ax2.xaxis.set_major_locator(MultipleLocator(1))

    plt.title('Operating Conditions at Pareto-Optimal Operating Points')
    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()


def plot_outputs_vs_iterations(f1_vals, f2_vals, c1_vals, c2_vals):
    input_names = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract (L/h)', 'Index Time (min)']
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Plot inputs vs iterations for Pareto-optimal points
    iterations = np.arange(1, len(f1_vals_plot) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot all inputs except the last one
    ax1.plot(iterations[50:], f1_vals_plot[50:], marker='o', label=f'Glucose Recovery')
    ax1.plot(iterations[50:], f2_vals_plot[50:], marker='o', label=f'Fructose Recovery')
    # ax1.plot(iterations, c1_vals_plot, marker='o', label=f'Glucose Purity')
    # ax1.plot(iterations, c1_vals_plot, marker='o', label=f'Fructose Purity')

    ax1.set_xlabel('Function Calls', fontsize=12)
    ax1.set_ylabel('Recovery Objective Functions (%)', fontsize=12)
    # ax1.grid(True)
    # ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Ensure integer ticks only (no half-values)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    plt.title('Plot of Recovery Objectives and Purity Constraints vs Number of Iterations')
    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()





# -------------------------- Constraint Porgression Over-Time
def constraints_vs_iterations(c1_vals, c2_vals):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100

    iterations = np.arange(0, len(c1_vals))

    fig, ax1 = plt.subplots()

    # Plot data
    ax1.scatter(iterations, c1_vals_plot, marker='o', label='Glucose Purity')
    ax1.scatter(iterations, c2_vals_plot, marker='o', label='Fructose Purity')
    ax1.axhline(y=99.5, linestyle="--", color="red", label='Constraint Threshold')

    # Axis labels and grid
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('(%)', fontsize=12)
    ax1.grid(True)

    # Set x-axis limits with buffer to avoid clipping edge markers
    x_end = max(iterations)
    ax1.set_xlim(-0.5, x_end + 0.5)

    # Ensure integer ticks only (no half-values)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    # Place legend outside the plot (upper right)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def find_pareto_front(x, y):
    """Return boolean mask of Pareto-optimal points."""
    pareto_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if pareto_mask[i]:
            pareto_mask[i] = not np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])))
    return pareto_mask

import numpy as np
import matplotlib.pyplot as plt

def find_pareto_front(x, y):
    """Return boolean mask of Pareto-optimal points."""
    pareto_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if pareto_mask[i]:
            pareto_mask[i] = not np.any(
                (x >= x[i]) & (y >= y[i]) &
                ((x > x[i]) | (y > y[i]))
            )
    return pareto_mask


import numpy as np

def generate_dummy_inputs(n, V_col, L, A_col, d_col, e, zone_config, Description="Dummy Optimization Run", bounds = [
                                                                                                        (2.5, 5),   # m1
                                                                                                        (1, 2),     # m2
                                                                                                        (2.5, 5),   # m3
                                                                                                        (1, 2),     # m4
                                                                                                        (2, 10)    # t_index/t_ref
                                                                                                    ]):
    """
    Generate dummy optimization data consistent with all_inputs_dict format.

    Args:
        n (int): Number of iterations (rows of all_inputs).
        bounds (list of tuple): Bounds for each variable [(low, high), ...].
        V_col (float): Column volume (mL).
        L (float): Column length (cm).
        A_col (float): Column area (cm²).
        d_col (float): Column diameter (cm).
        e (float): Voidage.
        zone_config (list): Zone configuration.
        Description (str): Description of the dataset.

    Returns:
        dict: all_inputs_dict with dummy data.
    """
    m = len(bounds)  # number of decision variables
    all_inputs = np.zeros((n, m))

    # Sample random values within the provided bounds
    for j, (low, high) in enumerate(bounds):
        all_inputs[:, j] = np.random.uniform(low, high, n)

    # Extract t_index_min (last column)
    t_index_min = all_inputs[:, 4]

    # Build dictionary
    all_inputs_dict = {
        "Description": Description,
        "m1": all_inputs[:, 0].tolist(),
        "m2": all_inputs[:, 1].tolist(),
        "m3": all_inputs[:, 2].tolist(),
        "m4": all_inputs[:, 3].tolist(),
        "t_index_min": t_index_min.tolist(),

        "Q1_(L/h)": ((3.6 * all_inputs[:, 0] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q2_(L/h)": ((3.6 * all_inputs[:, 1] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q3_(L/h)": ((3.6 * all_inputs[:, 2] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q4_(L/h)": ((3.6 * all_inputs[:, 3] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),

        "V_col_(mL)": [V_col],
        "L_col_(cm)": [L],
        "A_col_(cm)": [A_col],
        "d_col_(cm)": [d_col],
        "config": zone_config,
        "e": [e]
    }

    return all_inputs_dict
import numpy as np

def parse_inputs_dict(all_inputs_dict, include_t_index_as="col"):
    """
    Parse optimization results dictionary into structured matrices and scalars.

    Args:
        all_inputs_dict (dict): Input dictionary from optimization run.
        include_t_index_as (str): "row" or "col" - include t_index_min as extra row or column.

    Returns:
        dict with:
            - "M_matrix": numpy array of m ratios (+ t_index_min)
            - "Q_matrix": numpy array of Q values (+ t_index_min)
            - "vectors": dict of vectors (lists)
            - "scalars": dict of scalars (single values)
    """
    # Extract mj ratios
    m1 = np.array(all_inputs_dict["m1"])
    m2 = np.array(all_inputs_dict["m2"])
    m3 = np.array(all_inputs_dict["m3"])
    m4 = np.array(all_inputs_dict["m4"])
    t_index_min = np.array(all_inputs_dict["t_index_min"])

    # Extract Q values
    Q1 = np.array(all_inputs_dict["Q1_(L/h)"])
    Q2 = np.array(all_inputs_dict["Q2_(L/h)"])
    Q3 = np.array(all_inputs_dict["Q3_(L/h)"])
    Q4 = np.array(all_inputs_dict["Q4_(L/h)"])

    # Stack into matrices
    M_matrix = np.vstack([m1, m2, m3, m4]).T
    Q_matrix = np.vstack([Q1, Q2, Q3, Q4]).T

    if include_t_index_as == "col":
        M_matrix = np.column_stack([M_matrix, t_index_min])
        Q_matrix = np.column_stack([Q_matrix, t_index_min])
    elif include_t_index_as == "row":
        M_matrix = np.vstack([M_matrix, t_index_min])
        Q_matrix = np.vstack([Q_matrix, t_index_min])
    else:
        raise ValueError("include_t_index_as must be 'row' or 'col'")

    # Collect vectors and scalars
    vectors = {
        "config": np.array(all_inputs_dict["config"])
    }

    scalars = {
        "V_col_(mL)": all_inputs_dict["V_col_(mL)"][0],
        "L_col_(cm)": all_inputs_dict["L_col_(cm)"][0],
        "A_col_(cm)": all_inputs_dict["A_col_(cm)"][0],
        "d_col_(cm)": all_inputs_dict["d_col_(cm)"][0],
        "e": all_inputs_dict["e"][0]
    }

    return {
        "M_matrix": M_matrix,
        "Q_matrix": Q_matrix,
        "vectors": vectors,
        "scalars": scalars
    }


import numpy as np

def generate_dummy_data(n_points=50, description="Dummy Optimization Results"):
    """
    Generate dummy purity and recovery data for raffinate and extract streams.
    Returns a data_dict in the required format.
    """
    # --- Raffinate Purity ---
    c1_raff = np.clip(np.random.normal(0.7, 0.15, n_points), 0, 1)
    c2_raff = np.clip(1 - c1_raff + np.random.normal(0, 0.1, n_points), 0, 1)

    # --- Extract Purity ---
    c1_ext = np.clip(np.random.normal(0.6, 0.2, n_points), 0, 1)
    c2_ext = np.clip(1 - c1_ext + np.random.normal(0, 0.12, n_points), 0, 1)

    # --- Raffinate Recovery ---
    r1_raff = np.clip(np.random.normal(0.75, 0.1, n_points), 0, 1)
    r2_raff = np.clip(1 - r1_raff + np.random.normal(0, 0.08, n_points), 0, 1)

    # --- Extract Recovery ---
    r1_ext = np.clip(np.random.normal(0.65, 0.12, n_points), 0, 1)
    r2_ext = np.clip(1 - r1_ext + np.random.normal(0, 0.1, n_points), 0, 1)

    # --- Pack into dict ---
    data_dict = {
        "Description": description,
        "c1_vals": [c1_raff.tolist(), c1_ext.tolist()],   # Purity Comp1
        "c2_vals": [c2_raff.tolist(), c2_ext.tolist()],   # Purity Comp2
        "f1_vals": [r1_raff.tolist(), r1_ext.tolist()],   # Recovery Comp1
        "f2_vals": [r2_raff.tolist(), r2_ext.tolist()],   # Recovery Comp2
    }

    return data_dict

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_flows_and_indexing(all_inputs_dict):
    """
    Plot Q1-Q4 flowrates (L/h) and indexing time (min) across optimization iterations.
    Only markers are plotted (no lines), with grey vertical lines for readability.
    Legend is drawn in a separate figure.
    """

    # Extract values
    Q1 = np.array(all_inputs_dict["Q1_(L/h)"])
    Q2 = np.array(all_inputs_dict["Q2_(L/h)"])
    Q3 = np.array(all_inputs_dict["Q3_(L/h)"])
    Q4 = np.array(all_inputs_dict["Q4_(L/h)"])
    t_index = np.array(all_inputs_dict["t_index_min"])  # minutes

    iterations = np.arange(len(Q1))  # x-axis

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Grey vertical lines at each iteration
    for i in iterations:
        ax1.axvline(x=i, color="grey", linestyle="--", alpha=0.3)

    # Plot only markers for flowrates
    m1 = ax1.plot(iterations, Q1, "o", label="Q1 (L/h)")
    m2 = ax1.plot(iterations, Q2, "s", label="Q2 (L/h)")
    m3 = ax1.plot(iterations, Q3, "^", label="Q3 (L/h)")
    m4 = ax1.plot(iterations, Q4, "d", label="Q4 (L/h)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Flow rate (L/h)")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))  # 2 sig figs

    # Right-hand axis for indexing time
    ax2 = ax1.twinx()
    m5 = ax2.plot(iterations, t_index, "x", color="k", label="Indexing Time (min)")
    ax2.set_ylabel("Indexing Time (min)", color="k")
    ax2.tick_params(axis="y", labelcolor="k")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))  # 2 sig figs

    plt.title("Flow Rates and Indexing Time per Iteration")
    plt.tight_layout()
    plt.show()

    # --- Legend in separate figure ---
    fig_leg = plt.figure(figsize=(8, 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    handles = m1 + m2 + m3 + m4 + m5
    labels = [h.get_label() for h in handles]
    ax_leg.legend(handles, labels, loc="center", ncol=5, frameon=False)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_flows_indexing_grid(all_inputs_dict):
    """
    Plot Q1-Q4 flowrates (L/h) and indexing time (min) across optimization iterations
    on a (2,2) grid. Each subplot highlights one Q in color while others remain grey.
    Indexing time is always shown in black.
    """

    # Extract values
    Qs = [
        np.array(all_inputs_dict["Q1_(L/h)"]),
        np.array(all_inputs_dict["Q2_(L/h)"]),
        np.array(all_inputs_dict["Q3_(L/h)"]),
        np.array(all_inputs_dict["Q4_(L/h)"]),
    ]
    Q_labels = ["Q1 (L/h)", "Q2 (L/h)", "Q3 (L/h)", "Q4 (L/h)"]
    Q_markers = ["o", "s", "^", "d"]  # different symbols for each
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    t_index = np.array(all_inputs_dict["t_index_min"])  # minutes
    iterations = np.arange(len(Qs[0]))

    # Setup subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for k in range(4):  # loop over Q1-Q4
        ax1 = axes[k]

        # Grey vertical lines at each iteration
        for i in iterations:
            ax1.axvline(x=i, color="grey", linestyle="--", alpha=0.2)

        # # Plot all Q’s in grey
        # for j in range(4):
        #     ax1.plot(iterations, Qs[j], Q_markers[j], 
        #              color="grey", alpha=0.5, label=f"{Q_labels[j]} (other)" if j != k else None)

        # Highlight only the target Q in color
        ax1.plot(iterations, Qs[k], Q_markers[k], color=colors[k], label=Q_labels[k])

        # Left y-axis (flow rates)
        ax1.set_ylabel("Flow rate (L/h)")
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

        # Right-hand axis for indexing time
        ax2 = ax1.twinx()
        ax2.plot(iterations, t_index, "x", color="k", label="Indexing Time (min)")
        ax2.set_ylabel("Indexing Time (min)", color="k")
        ax2.tick_params(axis="y", labelcolor="k")
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

        ax1.set_title(f"Highlight: {Q_labels[k]}")
        ax1.set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
def find_pareto_front(x, y):
    """Return boolean mask of Pareto-optimal points."""
    pareto_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if pareto_mask[i]:
            pareto_mask[i] = not np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])))
    return pareto_mask

# def find_pareto_front(f1, f2):
#     """
#     Finds Pareto front for 2D objective space (f1, f2).
#     Accepts inputs as lists, 1D arrays, or 2D arrays (2, n_points).
#     Returns boolean mask of Pareto-optimal points.
#     """
#     f1 = np.asarray(f1)
#     f2 = np.asarray(f2)

#     # --- Flatten if shape is (2, n_points) or (n_points, 2)
#     if f1.ndim > 1:
#         if f1.shape[0] == 2:   # (2, n_points)
#             f1 = f1[0, :]
#             f2 = f1[1, :] if f2.ndim > 1 else f2
#         elif f1.shape[1] == 2: # (n_points, 2)
#             f1, f2 = f1[:, 0], f1[:, 1]
#         else:
#             raise ValueError(f"Unexpected shape for f1/f2: {f1.shape}")

#     if f2.ndim > 1:
#         if f2.shape[0] == 2:
#             f2 = f2[1, :]
#         elif f2.shape[1] == 2:
#             f2 = f2[:, 1]
#         else:
#             raise ValueError(f"Unexpected shape for f2: {f2.shape}")

#     n_points = len(f1)
#     is_optimal = np.ones(n_points, dtype=bool)

#     for i in range(n_points):
#         if is_optimal[i]:
#             # Point i is dominated if another point is >= in both and > in at least one
#             is_optimal[is_optimal] = ~(
#                 (f1[i] <= f1[is_optimal]) &
#                 (f2[i] <= f2[is_optimal]) &
#                 ((f1[i] < f1[is_optimal]) | (f2[i] < f2[is_optimal]))
#             )

#     return is_optimal


def plot_flows_indexing_grid_with_dict(input_dict, output_dict):
    """
    Plots Q1-Q4 flows and indexing times across optimization iterations.
    Vertical dashed lines at each iteration.
    Red lines = purity-optimal, Blue lines = recovery-optimal.
    Grid: 2x2, each plot highlights one Q flow in color.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Extract matrices from input_dict ---
    m_mat = np.vstack([input_dict["m1"], input_dict["m2"],
                       input_dict["m3"], input_dict["m4"]]).T
    Q_mat = np.vstack([input_dict["Q1_(L/h)"], input_dict["Q2_(L/h)"],
                       input_dict["Q3_(L/h)"], input_dict["Q4_(L/h)"]]).T
    t_index = np.array(input_dict["t_index_min"])

    n_iter = Q_mat.shape[0]

    # --- Extract Pareto info ---
    f1_vals = np.array(output_dict["f1_vals"][0])  # raffinate
    f2_vals = np.array(output_dict["f2_vals"][0])  # raffinate
    c1_vals = np.array(output_dict["c1_vals"][0])
    c2_vals = np.array(output_dict["c2_vals"][0])

    pur_mask = find_pareto_front(c1_vals, c2_vals)
    rec_mask = find_pareto_front(f1_vals, f2_vals)

    # --- Colors & markers ---
    flow_colors = ["C0", "C1", "C2", "C3"]
    flow_labels = ["Q1", "Q2", "Q3", "Q4"]
    marker_t = "x"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i in range(4):
        ax = axs[i]
        for j in range(4):
            color = flow_colors[j] if i == j else "lightgray"
            for k in range(n_iter):
                line_color = "red" if pur_mask[k] else ("blue" if rec_mask[k] else "gray")
                ax.axvline(k, color=line_color, linestyle="--", alpha=0.5)
            ax.scatter(range(n_iter), Q_mat[:, j], color=color, s=40, label=flow_labels[j] if i==j else "")
        # Indexing time in all plots
        ax.scatter(range(n_iter), t_index, color="k", marker=marker_t, label="t_index")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Flow (L/h) / t_index (min)")
        ax.set_title(f"Highlighting {flow_labels[i]}")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.2g}")

    # --- Legend in separate figure ---
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    for i, lbl in enumerate(flow_labels):
        ax2.scatter([], [], color=flow_colors[i], label=lbl, s=40)
    ax2.scatter([], [], color="k", marker=marker_t, label="t_index")
    ax2.axvline(0, color="red", linestyle="--", label="Purity Optimal")
    ax2.axvline(0, color="blue", linestyle="--", label="Recovery Optimal")
    ax2.axis("off")
    ax2.legend(loc="center", ncol=3)
    plt.show()
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Normalize iteration index for grayscale (0 = light, 1 = dark)
    grayscale = np.linspace(0.3, 0.9, n_iter)
    colors = [str(1 - g) for g in grayscale]  # matplotlib allows grayscale as strings

    # (1,1): Recovery tradeoff
    axs[0].scatter(f1_vals, f2_vals, c=colors, edgecolors="k", s=80)
    axs[0].set_title("Recovery Tradeoff: Raffinate vs Extract", fontsize=18)
    axs[0].set_xlabel("Raffinate Recovery (%)", fontsize=18)
    axs[0].set_ylabel("Extract Recovery (%)", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # (1,2): Purity tradeoff
    axs[1].scatter(c1_vals, c2_vals, c=colors, edgecolors="k", s=80)
    axs[1].set_title("Purity Tradeoff: Raffinate vs Extract", fontsize=18)
    axs[1].set_xlabel("Raffinate Purity (%)", fontsize=18)
    axs[1].set_ylabel("Extract Purity (%)", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    plt.show()








# %% ------------------------------------------ EXTRACT DATA ---------------------------------------------------
#%% Define in the Inputs
# Input the location if the saved jsons
# Make sure the path include the folder AND the file names!
# inputs_path = r"C:\Users\nawau\OneDrive\Desktop\MEng_Code\case0_Literature\Feed_Rec_Obj\3333NORMGlu_Fru_commision_opt_51iter_config_all_input_20251025_213827.json"
# outputs_path = r"C:\Users\nawau\OneDrive\Desktop\MEng_Code\case0_Literature\Feed_Rec_Obj\3333NORMGlu_Fru_commision_opt_51iter_config_all_output_20251025_213827.json"

# # # Load the file and the data

# comp_1_name = "Borate"
# comp_2_name = "HCl"



# Run the Functions and Visualise
# --- Paretos
# rec_pareto_image_filename = create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget)
# pur_pareto_image_filename = create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget)




# Exceute Plots
# plot_raff_ext_pareto(c1_vals, c2_vals, f1_vals, f2_vals)
# plot_raff_ext_pareto(outputs_path, inputs_path, purity_constraint=99.5, comp_1_name="Glucose", comp_2_name="Fructose")


# plot_dual_pareto(
#     ext_pur_comp1, ext_pur_comp2,
#     ext_rec_comp1, ext_rec_comp2,
#     title_prefix="Extract"
# )

# create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_1_name)
# # create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_2_name)
# plot_outputs_vs_iterations(f1_vals, f2_vals, c1_vals, c2_vals)
# compare_recovery_pareto_plot(f1_vals_50, f2_vals_50, f1_vals_100, f2_vals_100, c1_vals_50, c2_vals_50, c1_vals_100, c2_vals_100) # "rec" "pur"

# # --- Constraints
# constraints_vs_iterations(c1_vals, c2_vals)
# plot_inputs_vs_iterations(all_inputs, f1_vals, f2_vals)

# # # # ---- Tables
# opt_table_for_outputs_image_filename = create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget)
# opt_table_for_inputs_image_filename = create_input_optimization_table(all_inputs, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals)
# # compare_pareto_similarity(f1_vals_100, f2_vals_100, f1_vals_50, f2_vals_50, tolerance_percent=15)


# %%
