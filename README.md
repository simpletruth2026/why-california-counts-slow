# Global Election Systems: K-Means Clustering Analysis

> **Research Focus:** California (USA) — examining its ballot-counting window and voter ID practices against 33 global comparators.

**[→ View Interactive Visualization](https://simpletruth2026.github.io/why-california-counts-slow/)**

---

## Overview

This project applies **K-Means++ clustering (k=3)** to classify 34 countries and jurisdictions along two dimensions:

| Dimension | Description | Scale |
|-----------|-------------|-------|
| **X — Ballot Counting Window** | How long before a credible result is known | 1 (same night) → 10 (permanent dispute) |
| **Y — Voter ID Strictness** | Strength of identity verification at the polls | 1 (none) → 10 (biometric) |

The analysis surfaces a striking finding: **California clusters with transitional democracies** (Colombia, Panama, Guinea) rather than with its peer developed nations — solely because of its institutionalized extended mail-ballot counting window, not because of any integrity failure.

---

## Cluster Results

### Cluster 1 — Mature Democracies 🟢
*Fast results (X: 1–3) · Strong ID (Y: 6–9.5)*

Germany, France, Netherlands, Sweden, Norway, Belgium, Spain, Italy, Switzerland, Austria, Denmark, Finland, Portugal, Ireland, Japan, South Korea, Singapore, Taiwan, Israel, Canada, United Kingdom

### Cluster 2 — Transitional Systems 🟡
*Moderate counting window (X: 4–7) · Moderate ID (Y: 3–5)*

**California (USA) ★**, Colombia, Panama, Guinea

### Cluster 3 — Fragile / Disputed 🔴
*Long windows (X: 7–10) · Weak ID (Y: 1–3.5)*

Venezuela, Nicaragua, Haiti, Bolivia, Peru, Honduras, Guatemala, DR Congo, Somalia, Sudan, Mali

---

## Repository Structure

```
election-systems-clustering/
├── README.md
├── data/
│   └── election_systems.csv        # Raw scores for all 34 entries
├── analysis/
│   └── kmeans_clustering.py        # Reproducible Python analysis + static plot
├── visualization/
│   └── index.html                  # Interactive canvas visualization
├── figures/
│   └── cluster_plot.png            # Static output from Python script
└── docs/
    └── index.html                  # GitHub Pages deployment (same as visualization/)
```

---

## Methodology

### Scoring Rubric

**X — Ballot Counting Window**
```
1.0–2.0   Results same night (within hours)
2.5–3.5   Overnight to next morning
4.0–5.0   3–7 days
6.0–7.0   1–3 weeks or contested
7.5–8.5   Weeks to months / institutionally blocked
9.0–10.0  Permanent dispute or results withheld
```

**Y — Voter ID Strictness**
```
1.0–2.0   No verification at polls
3.0–4.0   Voter roll check / signature matching
5.0–6.0   Voter notification card (non-photo)
7.0–7.5   Non-photo government-issued ID
8.0–9.0   Mandatory photo ID
9.5–10.0  Biometric / thumbprint required
```

### Algorithm

K-Means++ initialization ensures well-separated initial centroids, reducing sensitivity to random seed choice. After convergence, clusters are semantically remapped by ranking centroids on the composite score `(Y − X)`:

- Highest score → **Cluster 1** (mature: high ID, low count window)
- Middle score  → **Cluster 2** (transitional)
- Lowest score  → **Cluster 3** (fragile: low ID, high count window)

---

## Reproducing the Analysis

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/election-systems-clustering.git
cd election-systems-clustering

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 3. Run the analysis (generates figures/cluster_plot.png)
python analysis/kmeans_clustering.py

# 4. Open the interactive visualization
open visualization/index.html   # macOS
# or: start visualization/index.html  (Windows)
# or: xdg-open visualization/index.html  (Linux)
```

---

## Data Sources

Scoring is based on the following sources, verified as of 2024–2025:

- **Counting windows:** Official electoral commission timelines; Carter Center, OAS, and EU observation mission reports; AP/Reuters final certification dates
- **Voter ID requirements:** IFES Election Guide; ACE Electoral Knowledge Network; national electoral authority documentation
- **Disputed elections:** Freedom House *Nations in Transit*; IDEA Global State of Democracy reports

---

## Key Finding

California's placement in Cluster 2 reflects a **structural design choice** — institutionalized mail-ballot acceptance windows — rather than electoral dysfunction. Among the 21 Cluster 1 mature democracies, **none** combine a counting window score above 3.0 with an ID score below 5.0. California scores 6.5 on the counting dimension while peer states like Germany (1.8), France (1.8), and Canada (2.5) resolve results the same evening.

---

## License

MIT License. Data and methodology are freely reusable with attribution.

---

## Citation

```
Han, B. (2025). Global Election Systems: K-Means Clustering Analysis.
GitHub. https://github.com/YOUR_USERNAME/election-systems-clustering
```
