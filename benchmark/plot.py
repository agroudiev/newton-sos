import pandas as pd
import seaborn as sns

df = pd.read_csv("benchmark/dfs/benchmark_results.csv")
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
plt = sns.lineplot(data=df, x="n_samples", y="value", hue="variable")
plt.set(
    yscale="log",
    xlabel="Size of the SDP (n_samples)",
    ylabel="Time (s)",
    title="1D Sinusoid",
)
plt.legend(title="Solver")
plt.figure.savefig("benchmark/plots/benchmark_plot.pdf", bbox_inches="tight")
