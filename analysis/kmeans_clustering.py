"""
Global Election Systems: K-Means Clustering Analysis
=====================================================
Research focus: California (USA) — positioned against global comparators
on two dimensions:
  X  count_score  — ballot counting window (1=same night, 10=unresolved/permanent dispute)
  Y  id_score     — voter ID strictness    (1=none, 10=biometric)

Algorithm: K-Means++ (k=3), with semantic label remapping so that
  Cluster 2 (green)  = mature democracies  (low X, high Y)
  Cluster 1 (yellow) = transitional systems (mid X, mid Y)
  Cluster 0 (red)    = fragile systems      (high X, low Y)

Usage:
    pip install pandas scikit-learn matplotlib seaborn
    python analysis/kmeans_clustering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/election_systems.csv")
X = df[["count_score", "id_score"]].values

# ── 1. K-Means++ clustering (k=3) ────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=50, random_state=42)
kmeans.fit(X)
df["raw_cluster"] = kmeans.labels_
centers = kmeans.cluster_centers_

# ── 2. Semantic remapping ─────────────────────────────────────────────────────
# Score each center by (id_score - count_score): higher = more "mature"
scores = centers[:, 1] - centers[:, 0]
rank = np.argsort(scores)[::-1]          # descending: mature → transitional → fragile
remap = {rank[0]: 2, rank[1]: 1, rank[2]: 0}
df["cluster"] = df["raw_cluster"].map(remap)
new_centers = centers[rank]              # reordered centers

CLUSTER_NAMES = {
    2: "Mature Democracies",
    1: "Transitional Systems",
    0: "Fragile / Disputed"
}
CLUSTER_COLORS = {2: "#06d6a0", 1: "#ffd166", 0: "#ff6b6b"}
CA_COLOR = "#ff9f1c"

# ── 3. Print summary ──────────────────────────────────────────────────────────
print("=" * 60)
print("GLOBAL ELECTION SYSTEMS — K-MEANS CLUSTERING RESULTS")
print("=" * 60)
for cid in [2, 1, 0]:
    members = df[df["cluster"] == cid]["country"].tolist()
    cx, cy = new_centers[2 - cid] if cid != 1 else centers[rank[1]]
    # get correct center
    orig = [k for k, v in remap.items() if v == cid][0]
    cx, cy = centers[orig]
    print(f"\n{'─'*40}")
    print(f"Cluster {cid}: {CLUSTER_NAMES[cid]}")
    print(f"  Centroid → count={cx:.2f}, id={cy:.2f}")
    print(f"  Members  → {', '.join(members)}")

print("\n")
ca = df[df["focus"] == 1].iloc[0]
ca_cluster = CLUSTER_NAMES[ca["cluster"]]
print(f"★ FOCUS — California (USA)")
print(f"   count_score = {ca['count_score']}  |  id_score = {ca['id_score']}")
print(f"   Cluster     = {ca_cluster}")
print("=" * 60)

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("#0b0e14")
ax.set_facecolor("#131720")

# Grid
for v in range(0, 11, 2):
    ax.axvline(v, color="#1e2535", linewidth=0.8, zorder=0)
    ax.axhline(v, color="#1e2535", linewidth=0.8, zorder=0)

# Cluster halos
for cid in [2, 1, 0]:
    orig = [k for k, v in remap.items() if v == cid][0]
    cx, cy = centers[orig]
    col = CLUSTER_COLORS[cid]
    for r, a in [(2.2, 0.06), (1.5, 0.10), (0.8, 0.14)]:
        circle = plt.Circle((cx, cy), r, color=col, alpha=a, zorder=1)
        ax.add_patch(circle)
    # centroid marker
    ax.plot(cx, cy, "+", color=col, markersize=12, markeredgewidth=1.5,
            alpha=0.6, zorder=3)

# Data points
for _, row in df.iterrows():
    is_ca = row["focus"] == 1
    col = CA_COLOR if is_ca else CLUSTER_COLORS[row["cluster"]]
    size = 120 if is_ca else 55
    zord = 6 if is_ca else 4

    ax.scatter(row["count_score"], row["id_score"],
               color=col, s=size, zorder=zord,
               edgecolors="white" if is_ca else col,
               linewidths=1.8 if is_ca else 0.5,
               alpha=0.95)

    # Label
    name = row["country"]
    ox, oy = 0.18, 0.18
    ha = "left"
    if row["count_score"] > 8.0:
        ox, ha = -0.18, "right"
    if row["id_score"] < 1.8:
        oy = -0.35
    if is_ca:
        ax.annotate(f"★ {name}",
                    xy=(row["count_score"], row["id_score"]),
                    xytext=(row["count_score"] + ox, row["id_score"] + oy + 0.15),
                    color=CA_COLOR, fontsize=9.5, fontweight="bold", ha=ha, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ff9f1c18",
                              ec=CA_COLOR, lw=0.8))
    else:
        ax.annotate(name,
                    xy=(row["count_score"], row["id_score"]),
                    xytext=(row["count_score"] + ox, row["id_score"] + oy),
                    color=col, fontsize=7.8, ha=ha, zorder=5, alpha=0.9)

# Axis styling
ax.set_xlim(-0.3, 10.5)
ax.set_ylim(-0.3, 10.5)
ax.set_xticks(range(0, 11, 2))
ax.set_yticks(range(0, 11, 2))
ax.tick_params(colors="#5a6580", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2535")

ax.set_xlabel("Ballot Counting Window Score  (1 = same night → 10 = permanent dispute)",
              color="#8090b0", fontsize=10, labelpad=10)
ax.set_ylabel("Voter ID Strictness Score  (1 = none → 10 = biometric)",
              color="#8090b0", fontsize=10, labelpad=10)
ax.set_title("Global Election Systems: K-Means Clustering (k=3)\n"
             "Dimensions: Counting Speed × Voter ID Strictness",
             color="#d4dbe8", fontsize=13, fontweight="bold", pad=16)

# Legend
patches = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
           for c in [2, 1, 0]]
patches.append(mpatches.Patch(color=CA_COLOR, label="★ California (USA) — Research Focus"))
legend = ax.legend(handles=patches, loc="lower left",
                   facecolor="#1a2030", edgecolor="#2e3a56",
                   labelcolor="#d4dbe8", fontsize=8.5,
                   framealpha=0.9, borderpad=0.8)

# Score rubric annotation
rubric = (
    "Scoring Rubric\n"
    "Count Score:  1=same night · 3=1-2 days\n"
    "              5=5-7 days · 7=2-3 weeks · 9=30+ days\n"
    "ID Score:     1=none · 3=roll check · 5=signature\n"
    "              7=non-photo gov ID · 9=mandatory photo ID"
)
ax.text(10.4, 10.4, rubric, color="#5a6580", fontsize=6.8,
        va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0d1018", ec="#1e2535", lw=0.8))

plt.tight_layout()
plt.savefig("figures/cluster_plot.png", dpi=180,
            bbox_inches="tight", facecolor="#0b0e14")
print("\nFigure saved → figures/cluster_plot.png")
plt.show()
