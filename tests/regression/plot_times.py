import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="plot_times.py", description="Plot the ellapsed times results."
)

parser.add_argument("results", type=str, help="Choose the results to plot.")

args = parser.parse_args()
results_list = args.results.split(sep=",")

vals = {
    "interpolation_levelset_detection": [],
    "compute_tags_measures": [],
    "interpolation_levelset": [],
    "interpolation_source": [],
    "assemble_phifem_system": [],
    "set_up_petsc_solver": [],
    "solve": [],
    "multiply_solution_levelset": [],
}
for result in results_list:
    df = pl.read_csv(os.path.join("tests_data", result + ".csv"))

    dct = df.to_dict()
    for key, val in dct.items():
        if key != "measure" and key != "total":
            vals[key].append(val.to_list()[0])

bottom = np.zeros(len(results_list))
fig = plt.figure()
ax = fig.subplots()
for key, val in vals.items():
    ax.bar(results_list, val, 0.5, label=key, bottom=bottom)
    bottom += val
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.tick_params(axis="x", rotation=20)
plt.ylabel("Wallclock time (s)")
plt.savefig("times_plot.png", dpi=300, bbox_inches="tight")
