import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv("./notebooks/groundingdino_florence_comparison_results/comparison_summary.csv")

# ── Clean: only rows where both models detected ───────────────────────────────
df_valid = df[
    (df["gdino_diameter_px"] > 0) &
    (df["florence_diameter_px"] > 0) &
    (df["gdino_trunk_area"] > 0) &
    (df["florence_trunk_area"] > 0)
].copy()

total     = len(df)
valid     = len(df_valid)

print(f"Total images : {total}")
print(f"Both detected: {valid} ({valid/total*100:.1f}%)")

# ── 1. BBox IoU stats ─────────────────────────────────────────────────────────
iou = df_valid["bbox_iou"]

# ── 2. Diameter stats ─────────────────────────────────────────────────────────
df_valid["diameter_mae"]  = df_valid["diameter_diff_px"].abs()
df_valid["diameter_mape"] = df_valid["diameter_mae"] / df_valid["gdino_diameter_px"] * 100
r_diam, p_diam = stats.pearsonr(df_valid["gdino_diameter_px"], df_valid["florence_diameter_px"])

# ── 3. Mask area stats ────────────────────────────────────────────────────────
df_valid["area_diff_pct"] = (
    abs(df_valid["gdino_trunk_area"] - df_valid["florence_trunk_area"])
    / df_valid["gdino_trunk_area"] * 100
)
r_area, p_area = stats.pearsonr(df_valid["gdino_trunk_area"], df_valid["florence_trunk_area"])

# ── 4. BBox area stats ────────────────────────────────────────────────────────
df_valid["bbox_area_diff_pct"] = (
    abs(df_valid["gdino_bbox_area"] - df_valid["florence_bbox_area"])
    / df_valid["gdino_bbox_area"] * 100
)

# ── 5. Angle stats (only where both have values) ──────────────────────────────
angle_valid = df_valid[
    df_valid["gdino_angle_deg"].notna() &
    df_valid["florence_angle_deg"].notna() &
    (df_valid["gdino_angle_deg"] != "N/A") &
    (df_valid["florence_angle_deg"] != "N/A")
].copy()
angle_valid["angle_diff"] = abs(
    angle_valid["gdino_angle_deg"].astype(float) -
    angle_valid["florence_angle_deg"].astype(float)
)

# ── Print paper-ready table ───────────────────────────────────────────────────
print("\n" + "="*65)
print("  PAPER SUMMARY TABLE")
print("="*65)
print(f"  {'Metric':<40} {'GDino':>10} {'Florence':>10}")
print(f"  {'-'*62}")
print(f"  {'Mean Trunk Mask Area (px²)':<40} {df_valid['gdino_trunk_area'].mean():>10,.0f} {df_valid['florence_trunk_area'].mean():>10,.0f}")
print(f"  {'Std Trunk Mask Area (px²)':<40} {df_valid['gdino_trunk_area'].std():>10,.0f} {df_valid['florence_trunk_area'].std():>10,.0f}")
print(f"  {'Mean Area Diff (%)':<40} {df_valid['area_diff_pct'].mean():>10.2f}% {'—':>9}")
print(f"  {'-'*62}")
print(f"  {'Mean BBox Area (px²)':<40} {df_valid['gdino_bbox_area'].mean():>10,.0f} {df_valid['florence_bbox_area'].mean():>10,.0f}")
print(f"  {'Mean BBox Area Diff (%)':<40} {df_valid['bbox_area_diff_pct'].mean():>10.2f}% {'—':>9}")
print(f"  {'Mean BBox IoU':<40} {iou.mean():>10.4f} {'—':>9}")
print(f"  {'Std BBox IoU':<40} {iou.std():>10.4f} {'—':>9}")
print(f"  {'IoU > 0.90 (%)':<40} {(iou>0.90).mean()*100:>9.1f}% {'—':>9}")
print(f"  {'IoU > 0.70 (%)':<40} {(iou>0.70).mean()*100:>9.1f}% {'—':>9}")
print(f"  {'-'*62}")
print(f"  {'Mean Diameter (px)':<40} {df_valid['gdino_diameter_px'].mean():>10.1f} {df_valid['florence_diameter_px'].mean():>10.1f}")
print(f"  {'Std Diameter (px)':<40} {df_valid['gdino_diameter_px'].std():>10.1f} {df_valid['florence_diameter_px'].std():>10.1f}")
print(f"  {'Mean Abs Error / MAE (px)':<40} {df_valid['diameter_mae'].mean():>10.2f} {'—':>9}")
print(f"  {'Std MAE (px)':<40} {df_valid['diameter_mae'].std():>10.2f} {'—':>9}")
print(f"  {'MAPE (%)':<40} {df_valid['diameter_mape'].mean():>10.2f}% {'—':>9}")
print(f"  {'Pearson r (diameter)':<40} {r_diam:>10.4f} {'—':>9}")
print(f"  {'p-value (diameter)':<40} {p_diam:>10.2e} {'—':>9}")
print(f"  {'-'*62}")
if len(angle_valid) > 0:
    print(f"  {'Mean Trunk Angle (°)':<40} {angle_valid['gdino_angle_deg'].astype(float).mean():>10.2f} {angle_valid['florence_angle_deg'].astype(float).mean():>10.2f}")
    print(f"  {'Mean Angle Diff (°)':<40} {angle_valid['angle_diff'].mean():>10.2f} {'—':>9}")
print("="*65)

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Grounding DINO vs Florence-2", fontsize=13, fontweight="bold")

# Panel 1: Diameter scatter + correlation
ax = axes[0]
ax.scatter(df_valid["gdino_diameter_px"], df_valid["florence_diameter_px"],
           alpha=0.6, color="#2980B9", s=40, edgecolors="white", linewidth=0.5)
lim_min = min(df_valid["gdino_diameter_px"].min(), df_valid["florence_diameter_px"].min()) * 0.95
lim_max = max(df_valid["gdino_diameter_px"].max(), df_valid["florence_diameter_px"].max()) * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="Perfect agreement")
ax.set_xlabel("GDino Diameter (px)", fontsize=11)
ax.set_ylabel("Florence-2 Diameter (px)", fontsize=11)
ax.set_title(f"Diameter Correlation\nr={r_diam:.4f}, p={p_diam:.2e}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: Bland-Altman (diameter)
ax = axes[1]
means     = (df_valid["gdino_diameter_px"] + df_valid["florence_diameter_px"]) / 2
diffs     = df_valid["gdino_diameter_px"]  - df_valid["florence_diameter_px"]
mean_diff = diffs.mean()
loa_u     = mean_diff + 1.96 * diffs.std()
loa_l     = mean_diff - 1.96 * diffs.std()
ax.scatter(means, diffs, alpha=0.6, color="#27AE60", s=40, edgecolors="white", linewidth=0.5)
ax.axhline(mean_diff, color="red",  linestyle="-",  linewidth=2,   label=f"Bias = {mean_diff:.1f}px")
ax.axhline(loa_u,     color="gray", linestyle="--", linewidth=1.5, label=f"+1.96SD = {loa_u:.1f}px")
ax.axhline(loa_l,     color="gray", linestyle="--", linewidth=1.5, label=f"-1.96SD = {loa_l:.1f}px")
ax.set_xlabel("Mean Diameter (px)", fontsize=11)
ax.set_ylabel("GDino − Florence (px)", fontsize=11)
ax.set_title("Bland-Altman Plot\n(Diameter Agreement)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: BBox IoU histogram
ax = axes[2]
ax.hist(iou, bins=20, color="#8E44AD", edgecolor="white", alpha=0.85)
ax.axvline(iou.mean(), color="red",    linestyle="--", linewidth=2, label=f"Mean = {iou.mean():.3f}")
ax.axvline(iou.median(),color="orange",linestyle=":",  linewidth=2, label=f"Median = {iou.median():.3f}")
ax.set_xlabel("BBox IoU", fontsize=11)
ax.set_ylabel("Number of Images", fontsize=11)
ax.set_title("BBox IoU Distribution\n(GDino vs Florence-2)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./comparison_output/paper_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Saved → comparison_output/paper_metrics.png")