import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl

parent_dir = os.path.dirname(__file__)

test_case_list = [tc for tc in next(os.walk("."))[1] if "__" not in tc]

parser = argparse.ArgumentParser(prog="Run the demo.",
                                 description="Run iterations of FEM or phiFEM with uniform or adaptive refinement on the given test case.")

parser.add_argument("test_case", type=str, choices=test_case_list, help="Name of the test case.")
parser.add_argument("parameters", type=str, help="List of yaml parameters files (separated by commas ',').")
parser.add_argument("quantities", type=str, help="List of quantities to compare (separated by commas ',')")
parser.add_argument("--trunc", type=int, default=-4, help="Truncation to compute convergence rates.")

args = parser.parse_args()
test_case       = args.test_case
parameters_list = args.parameters.split(sep=",")
quantities_list = args.quantities.split(sep=",")
trunc           = args.trunc

test_case_path = os.path.join(parent_dir, test_case)

fig = plt.figure()
ax = fig.subplots()
output_name = ""
for parameter in parameters_list:
    results_path = os.path.join(test_case_path, parameter, "results.csv")
    df = pl.read_csv(results_path)
    dofs = df["dofs"]
    output_name += parameter + "_"
    for quantity in quantities_list:
        qty = df[quantity]
        rate = np.polyfit(np.log(dofs[trunc:]), np.log(qty[trunc:]), 1)[0]
        ax.loglog(dofs, qty, "^--", label=parameter + " " + quantity + " " + "(rate: " + str(np.round(rate,2)) + ")")
        if "estimator" in quantity:
            output_name += "est" + "_"
        elif "error" in quantity:
            output_name += "err" + "_"
plt.xlabel("dofs")
plt.legend()
plt.savefig(os.path.join(test_case_path, output_name + ".png"), dpi=300, bbox_inches="tight")