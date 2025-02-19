import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

parent_dir = os.path.dirname(__file__)
pd.set_option('display.expand_frame_repr', False)
plt.style.use(['ggplot',"./plots.mplstyle"])

def plot_efficiencies(data_dir, norm, label, style="-^"):
    df = pd.read_csv(os.path.join(data_dir, "results.csv"))
    iterations = np.arange(len(df["dofs"].values))
    key2label = {"H10 efficiency": r"$\eta/|u_{\mathrm{ref}} - u_h|_{H^1(\Omega)}$",
                 "L2 efficiency":  r"$\nu/\|u_{\mathrm{ref}} - u_h\|_{L^2(\Omega)}$"}
    
    for key in [k for k in df.keys() if "efficiency" in k and norm in k]:
        print(key)
        vals = df[key].values
        keylab = key2label[key]
        plt.plot(iterations, vals, style, label=keylab + ", " + label)

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
                "lshaped":   "L-shaped",
                "flower":    "Flower"}

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
    cutoffs_dict={"H10": -10, "L2": -10, "uniform": -5}
    
    for solver, marker in zip(solvers, markers):
        for ref_strat in list_ref_strats:
            style = styles_dict[ref_strat]
            cutoff = cutoffs_dict[ref_strat]
            data_dir = os.path.join(parent_dir, test_case, "output_" + solver, ref_strat)
            print("Data dir:", data_dir)
            label_ref = label_ref_strats[ref_strat]
            if mode=="all":
                label = f"{label_solver[solver]}{label_ref}"
            else:
                label = f"{label_ref}"
            plot_efficiencies(data_dir, norm, label, style=style+marker)
    
    plt.xlabel("Iteration num.")
    ax.set_ylim([0., 10.])
    plt.legend()
    plt.savefig(os.path.join(save_dir, "plot_efficiencies_" + norm + "_" + ref_strats + ".pdf"), bbox_inches="tight")