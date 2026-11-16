"""Plot P95 latency and error rate against requests/s from a Locust ramp run.

    python loadtest/plot_capacity.py reports/capacity_1r_stats_history.csv [...more files]
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

for path in sys.argv[1:]:
    label = path.split("/")[-1].replace("_stats_history.csv", "")
    df = pd.read_csv(path)
    df = df[(df["Name"] == "Aggregated") & (df["Requests/s"] > 0)].copy()
    df["bin"] = (df["Requests/s"] // 10) * 10          # group by 10 req/s to smooth the points
    g = df.groupby("bin").agg(p95=("95%", "mean"),
                              rps=("Requests/s", "mean"),
                              fails=("Failures/s", "mean"),
                              reqs=("Requests/s", "mean")).reset_index()
    g["error_pct"] = 100 * g["fails"] / g["reqs"]
    ax1.plot(g["rps"], g["p95"], "o-", label=f"P95 {label}")
    ax2.plot(g["rps"], g["error_pct"], "s--", alpha=0.6, label=f"errors {label}")

ax1.axhline(200, ls=":", color="grey", label="200 ms budget")
ax1.set_xlabel("Requests per second")
ax1.set_ylabel("P95 latency (ms)")
ax2.set_ylabel("Error rate (%)")
ax2.set_ylim(bottom=0)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
ax1.set_title("Capacity: P95 latency and errors vs throughput")
fig.tight_layout()
fig.savefig("reports/capacity.png", dpi=150)
print("saved reports/capacity.png")