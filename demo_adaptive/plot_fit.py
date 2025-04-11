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
    key2label = {"H10 estimator": r"$\eta$",
                 "L2 estimator":  r"$\nu$",
                 "H10 error":     r"$|u_{\mathrm{ref}} - u_h|_{H^1(\Omega)}$",
                 "L2 error":      r"$\|u_{\mathrm{ref}} - u_h\|_{L^2(\Omega)}$"}

    slopes = defaultdict(list)
    for key in [k for k in df.keys() if (norm in k) and ("efficiency" not in k) and ("eta" not in k)]:
        vals = df[key].values
        keylab = key2label[key]
        leg_label = keylab + " (" + label + ")"
        a, b = np.polyfit(np.log(dofs[cutoff:]), np.log(vals[cutoff:]), 1)
        slopes[key].append(np.round(a,2))
        ax.loglog(dofs, vals, style, label=leg_label)
        # if "error" in key:
        #     if list_dofs is not None:
        #         list_dofs[0].append(dofs)
        #     if list_vals is not None:
        #         list_vals[0].append(vals)
        if "estimator" in key:
            if list_dofs is not None:
                list_dofs = dofs
            if list_vals is not None:
                list_vals = vals
    slopes_df = pd.DataFrame(slopes)
    slopes_df.to_csv(os.path.join(data_dir, "slopes.csv"))
    print(data_dir)
    print(slopes_df)
    return list_dofs, list_vals

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
    label_solver = {"FEM": "FEM", "phiFEM": r"$\varphi$-FEM"}
    styles_dict={"H10": "--", "L2": "--", "uniform": "-"}
    # cutoffs_dict={"H10": -10, "L2": -10, "uniform": -5}
    cutoffs_dict={"H10": 1, "L2": 1, "uniform": 1}
    
    tuple_lst_dofs = [[], []]
    tuple_lst_vals= [[], []]
    for solver, marker_symb in zip(solvers, markers):
        for i, ref_strat in enumerate(list_ref_strats):
            style = styles_dict[ref_strat]
            cutoff = cutoffs_dict[ref_strat]
            data_dir = os.path.join(parent_dir, test_case, "output_" + solver, ref_strat)
            label_ref = label_ref_strats[ref_strat]
            if mode=="all":
                label = f"{label_solver[solver]}{label_ref}"
            else:
                label = f"{label_ref}"
            list_dofs, list_vals = plot_fit(ax, data_dir, norm, label, style=style+marker_symb, cutoff=cutoff, list_dofs=tuple_lst_dofs[i], list_vals=tuple_lst_vals[i])
            tuple_lst_dofs[i] = list_dofs
            tuple_lst_vals[i] = list_vals
    
    for i, slope, gap in zip(range(2), [-0.4, -0.5], [0.04, -0.04]):
        lst_dofs = tuple_lst_dofs[i]
        lst_vals = tuple_lst_vals[i]
        if len(lst_vals) > 0:
            #last_vals = [vls[-1] for vls in lst_vals]
            #min_index = np.argmin(last_vals)
            dofs = np.asarray(lst_dofs)#[min_index]
            vals = np.asarray(lst_vals)#[min_index]
            marker(ax, dofs, [vals], 0.75, gap, slope, color="dimgrey", round_num=2)
    plt.xlabel("dofs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "plot_convergence_" + norm + "_" + ref_strats + ".pdf"), bbox_inches="tight")