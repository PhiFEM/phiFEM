from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse
import sys

sys.path.append("/home/rbulle/Code/Other")
from marker.marker import marker

parent_dir = os.path.dirname(__file__)
pd.set_option('display.expand_frame_repr', False)
plt.style.use(['ggplot',"./plots.mplstyle"])

def plot_fit(ax, data_dir, norm, label, cutoff=-3, style="-^", list_dofs=None, list_vals=None):
    df = pd.read_csv(os.path.join(data_dir, "results.csv"))
    dofs = df["dofs"].values
    key2label = {"eta_G_H10": r"$\eta_{\varepsilon}$",
                 "eta_E_H10": r"$\eta_J$",
                 "eta_T_H10": r"$\eta_r$",
                 "H10 estimator": r"$\eta$"}

    slopes = defaultdict(list)
    for key in [k for k in df.keys() if ("eta" in k) and ("H10" in k) and ("efficiency" not in k) or (k == 'H10 estimator')]:
        # if "G" in key:
        #     cutoff = -14
        vals = df[key].values
        keylab = key2label[key]
        leg_label = keylab
        a, b = np.polyfit(np.log(dofs[cutoff:]), np.log(vals[cutoff:]), 1)
        slopes[key].append(np.round(a,2))

        ax.loglog(dofs, vals, style, label=leg_label)
        if "G" in key:
            print(a)
            if list_dofs is not None:
                list_dofs[0].append(dofs)
            if list_vals is not None:
                list_vals[0].append(vals)
        if "estimator" in key:
            print(a)
            if list_dofs is not None:
                list_dofs[1].append(dofs)
            if list_vals is not None:
                list_vals[1].append(vals)
    slopes_df = pd.DataFrame(slopes)
    slopes_df.to_csv(os.path.join(data_dir, "slopes.csv"))

if __name__=="__main__":
    demos_list = [demo for demo in next(os.walk("."))[1] if "__" not in demo]

    parser = argparse.ArgumentParser(prog="plot",
                                     description="Generate a loglog convergence plot.")

    parser.add_argument("test_case",  type=str, choices=demos_list)
    parser.add_argument("mode",       type=str, choices=["FEM",
                                                         "phiFEM",
                                                         "all"])
    parser.add_argument("norm",       type=str, choices=["H10",
                                                         "L2"])
    parser.add_argument("refinement", type=str, choices=["uniform",
                                                         "H10",
                                                         "L2",
                                                         "uniform-H10",
                                                         "uniform-L2"])
    args = parser.parse_args()
    test_case = args.test_case
    mode = args.mode
    norm = args.norm
    ref_strats = args.refinement

    fig = plt.figure()
    ax = fig.add_subplot()
    save_dir = os.path.join(parent_dir, test_case, "output_" + mode)
    
    tc2title = {"pdt_sines": "Product of sines",
                "lshaped": "L-shaped",
                "flower": "Flower"}

    if mode=="all":
        # plt.title(f"Test case: {tc2title[test_case]}.")
        save_dir = os.path.join(parent_dir, test_case)
        solvers=["FEM", "phiFEM"]
        markers=["o",   "^"]
    else:
        # plt.title(f"Test case: {test_case}, {mode}.")
        save_dir = os.path.join(parent_dir, test_case, "output_" + mode)
        solvers=[mode]
        markers=["o" if solvers[0]=="FEM" else "^"]

    list_ref_strats = ref_strats.split("-")
    if ref_strats in ["uniform-H10", "uniform-L2"]:
        label_ref_strats = {"H10": "adapt.", "L2": "adapt.", "uniform": "unif."}
    else:
        label_ref_strats = {"H10": "", "L2": "", "uniform": ""}
    label_solver = {"FEM": "FEM", "phiFEM": r"$\phi$-FEM"}
    styles_dict={"H10": "--", "L2": "--", "uniform": "-"}
    
    tuple_lst_dofs = [[], []]
    tuple_lst_vals= [[], []]
    for solver, marker_symb in zip(solvers, markers):
        for ref_strat in list_ref_strats:
            style = styles_dict[ref_strat]
            data_dir = os.path.join(parent_dir, test_case, "output_" + solver, ref_strat)
            print("Data dir:", data_dir)
            label_ref = label_ref_strats[ref_strat]
            if mode=="all":
                label = f"{label_solver[solver]}{label_ref}"
            else:
                label = f"{label_ref}"
            plot_fit(ax, data_dir, norm, label, style=style+marker_symb, list_dofs=tuple_lst_dofs, list_vals=tuple_lst_vals)
    
    # for i, gap, slope in zip(range(2), [-0.05, 0.03], [-1.1, -0.5]):
    lst_dofs = tuple_lst_dofs[1]
    lst_vals = tuple_lst_vals[1]
    if len(lst_vals) > 0:
        last_vals = [vls[-1] for vls in lst_vals]
        min_index = np.argmin(last_vals)
        dofs = lst_dofs[min_index]
        vals = lst_vals[min_index]
        marker(ax, dofs, [vals], 0.65, 0.03, -0.5, color="dimgrey")
    plt.xlabel("dofs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "plot_residuals_" + norm + "_" + ref_strats + ".pdf"), bbox_inches="tight")