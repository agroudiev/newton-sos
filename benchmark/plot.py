import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

filename = "1D-long"

df = pd.read_csv(f"benchmark/dfs/{filename}.csv")
df = df.melt(
    id_vars="n_samples",
    value_vars=["python_newton_time", "rust_newton_time", "mosek_time"],
)

df["variable"] = df["variable"].replace(
    {
        "python_newton_time": "Damped Newton (Python)",
        "rust_newton_time": "Damped Newton (Rust)",
        "mosek_time": "MOSEK",
    }
)


# plot python_newton_time vs rust_newton_time as a function of n_samples
plt1 = sns.lineplot(data=df, x="n_samples", y="value", hue="variable")
plt1.set(
    yscale="log",
    xlabel="Size of the SDP (n_samples)",
    ylabel="Time (s)",
    title="1D Sinusoid",
)
plt1.legend(title="Solver")
plt1.figure.savefig(f"benchmark/plots/{filename}.pdf", bbox_inches="tight")

# close plot
plt.close()

# compute the average time for each n_samples and each method
df_avg = df.groupby(["n_samples", "variable"]).mean().reset_index()
# pivot the dataframe to have python and rust times in columns
df_pivot = df_avg.pivot(
    index="n_samples", columns="variable", values="value"
).reset_index()
# compute the ratio of python over rust
df_pivot["ratio"] = (
    df_pivot["Damped Newton (Python)"] / df_pivot["Damped Newton (Rust)"]
)

# filter to only keep ratio
df_ratio = df_pivot[["n_samples", "ratio"]]
print(df_ratio)

plt2 = sns.lineplot(data=df_ratio, x="n_samples", y="ratio")
plt2.set(
    # yscale="log",
    xlabel="Size of the SDP (n_samples)",
    ylabel="Time Ratio (Python / Rust)",
    title="1D Sinusoid",
)
plt2.legend().remove()
plt2.figure.savefig(f"benchmark/plots/{filename}_ratio.pdf", bbox_inches="tight")
