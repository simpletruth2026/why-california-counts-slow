"""
Global Election Systems: K-Means Clustering Analysis
=====================================================
Usage:
    pip install pandas scikit-learn matplotlib adjustText
    python analysis/kmeans_clustering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from adjustText import adjust_text
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/election_systems.csv")
X = df[["count_score", "id_score"]].values

# ── 1. K-Means++ (k=3) ───────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=50, random_state=42)
kmeans.fit(X)
df["raw_cluster"] = kmeans.labels_
centers = kmeans.cluster_centers_

# ── 2. Semantic remapping ─────────────────────────────────────────────────────
scores = centers[:, 1] - centers[:, 0]
rank   = np.argsort(scores)[::-1]
remap  = {rank[0]: 2, rank[1]: 1, rank[2]: 0}
df["cluster"] = df["raw_cluster"].map(remap)

CLUSTER_NAMES  = {2: "Cluster 1: Mature Democracies",
                  1: "Cluster 2: Transitional Systems",
                  0: "Cluster 3: Fragile / Disputed"}
CLUSTER_COLORS = {2: "#06d6a0", 1: "#ffd166", 0: "#ff6b6b"}
CA_COLOR = "#ff9f1c"

# ── 3. Print summary ──────────────────────────────────────────────────────────
print("=" * 60)
print("GLOBAL ELECTION SYSTEMS — K-MEANS CLUSTERING RESULTS")
print("=" * 60)
for cid in [2, 1, 0]:
    orig = [k for k, v in remap.items() if v == cid][0]
    cx, cy = centers[orig]
    members = df[df["cluster"] == cid]["country"].tolist()
    print(f"\n{'─'*40}")
    print(f"{CLUSTER_NAMES[cid]}")
    print(f"  Centroid → count={cx:.2f}, id={cy:.2f}")
    print(f"  Members  → {', '.join(members)}")
ca = df[df["focus"] == 1].iloc[0]
print(f"\n★ FOCUS — California (USA)")
print(f"   count={ca['count_score']}  id={ca['id_score']}  →  {CLUSTER_NAMES[ca['cluster']]}")
print("=" * 60)

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17, 11))
fig.patch.set_facecolor("#0b0e14")
ax.set_facecolor("#131720")

# Grid
for v in range(0, 11, 2):
    ax.axvline(v, color="#1e2535", linewidth=0.8, zorder=0)
    ax.axhline(v, color="#1e2535", linewidth=0.8, zorder=0)

# Scatter + collect text objects for adjustText
texts = []
for _, row in df.iterrows():
    is_ca = row["focus"] == 1
    col   = CA_COLOR if is_ca else CLUSTER_COLORS[row["cluster"]]
    size  = 150 if is_ca else 60
    zord  = 6   if is_ca else 4

    ax.scatter(row["count_score"], row["id_score"],
               color=col, s=size, zorder=zord,
               edgecolors="white" if is_ca else col,
               linewidths=2.2 if is_ca else 0.6,
               alpha=0.95)

    if is_ca:
        ax.plot(row["count_score"], row["id_score"] + 0.5,
                marker="*", color=CA_COLOR, markersize=11, zorder=7)

    fs  = 11.5 if is_ca else 10
    fw  = "bold" if is_ca else "normal"
    txt = ax.text(row["count_score"], row["id_score"],
                  row["country"],
                  color=col, fontsize=fs, fontweight=fw,
                  ha="left", va="bottom", zorder=5)
    texts.append(txt)

# Auto-adjust label positions to avoid overlaps
adjust_text(
    texts,
    x=df["count_score"].values,
    y=df["id_score"].values,
    ax=ax,
    expand=(1.4, 1.6),
    force_text=(0.5, 0.8),
    force_points=(0.3, 0.5),
    arrowprops=dict(arrowstyle="-", color="#ffffff33", lw=0.6),
    lim=500,
)

# Axes
ax.set_xlim(-0.5, 11.0)
ax.set_ylim(-0.5, 11.2)
ax.set_xticks(range(0, 11, 2))
ax.set_yticks(range(0, 11, 2))
ax.tick_params(colors="#5a6580", labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2535")

ax.set_xlabel("Ballot Counting Window Score  (1 = same night  →  10 = permanent dispute)",
              color="#8090b0", fontsize=11, labelpad=12)
ax.set_ylabel("Voter ID Strictness Score  (1 = none  →  10 = biometric)",
              color="#8090b0", fontsize=11, labelpad=12)
ax.set_title("Global Election Systems: K-Means Clustering (k = 3)\n"
             "Counting Speed  ×  Voter ID Strictness  ·  n = 34",
             color="#d4dbe8", fontsize=14, fontweight="bold", pad=18)

# Legend
patches = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
           for c in [2, 1, 0]]
patches.append(mpatches.Patch(color=CA_COLOR,
               label="★  California (USA) — Research Focus"))
ax.legend(handles=patches, loc="lower left",
          facecolor="#1a2030", edgecolor="#2e3a56",
          labelcolor="#d4dbe8", fontsize=9.5,
          framealpha=0.92, borderpad=1.0)

# Rubric
rubric = ("Scoring Rubric\n"
          "Count:  1 = same night · 3 = 1-2 days\n"
          "        5 = 5-7 days   · 7 = 2-3 weeks · 9 = 30+ days\n"
          "ID:     1 = none       · 3 = roll check\n"
          "        7 = non-photo  · 9 = mandatory photo ID")
ax.text(10.9, 11.1, rubric, color="#5a6580", fontsize=7.5,
        va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0d1018", ec="#1e2535", lw=0.8))

plt.tight_layout()
plt.savefig("figures/cluster_plot.png", dpi=180,
            bbox_inches="tight", facecolor="#0b0e14")
print("\nFigure saved → figures/cluster_plot.png")
plt.show()
