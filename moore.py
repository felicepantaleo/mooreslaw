import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from adjustText import adjust_text

# ---------------------------------------------------
# 1) Helper functions
# ---------------------------------------------------
def parse_transistor_count(raw_str: str):
    """
    Cleans bracketed refs, commas, etc., then finds numeric values.
    Returns the smallest integer if multiple numbers appear, or None if no match.
    """
    if not isinstance(raw_str, str):
        return None
    cleaned = re.sub(r"\[.*?\]", "", raw_str)  # remove bracketed refs
    cleaned = cleaned.replace(",", "")          # remove commas
    numbers = re.findall(r"\d+", cleaned)
    if not numbers:
        return None
    return min(int(num) for num in numbers)     # or max(...), if you prefer

def parse_year(raw_str: str):
    """
    Removes bracketed refs, parentheses, etc., then finds a 4-digit year.
    """
    if not isinstance(raw_str, str):
        return None
    cleaned = re.sub(r"\[.*?\]", "", raw_str)   # remove bracketed refs
    cleaned = re.sub(r"\(.*?\)", "", cleaned)   # remove parenthetical text
    cleaned = cleaned.strip()
    match = re.search(r"\b(19|20)\d{2}\b", cleaned)
    return int(match.group(0)) if match else None

def remove_parentheses(text: str):
    """
    Repeatedly remove innermost parentheses until no parentheses remain.
    """
    if not isinstance(text, str):
        return text
    while True:
        new_text = re.sub(r"\([^()]*\)", "", text)
        if new_text == text:
            break
        text = new_text
    return text.strip()

def fit_and_remove_outliers(df):
    """
    Performs a log-space linear fit on TransistorCountNumeric vs. YearNumeric,
    removing points with residual > 2 std dev. Returns inlier rows only.
    """
    x = df["YearNumeric"].values
    y = df["TransistorCountNumeric"].values
    log_y = np.log(y)

    # Linear fit in log space: log(y) = a + b*x
    coeffs = np.polyfit(x, log_y, 1)  # returns [b, a]
    fitted_log_y = np.polyval(coeffs, x)
    residuals = log_y - fitted_log_y

    # Outlier mask
    std_res = np.std(residuals)
    mask = np.abs(residuals) < 2 * std_res
    return df[mask].copy()

# ---------------------------------------------------
# 2) Load data
# ---------------------------------------------------
url = "https://en.wikipedia.org/wiki/Transistor_count"
df_list = pd.read_html(url)
print(f"Found {len(df_list)} tables on the page.")

# - df_list[3]: CPUs
# - df_list[4]: GPUs
df_cpus = df_list[3].copy()
df_gpus = df_list[4].copy()

# ---------------------------------------------------
# 3) Parse each table + remove outliers
# ---------------------------------------------------
datasets = [
    ("CPUs", df_cpus, "blue", 'o'),
    ("GPUs", df_gpus, "red",  'x'),
]

inliers_list = []

for label, df_raw, color, marker in datasets:
    # Parse numeric columns
    df_raw["TransistorCountNumeric"] = df_raw["Transistor count"].apply(parse_transistor_count)
    df_raw["YearNumeric"] = df_raw["Year"].apply(parse_year)

    # Drop rows without valid numeric data
    df_clean = df_raw.dropna(subset=["TransistorCountNumeric", "YearNumeric"])
    if df_clean.empty:
        print(f"No data available after cleaning for: {label}")
        continue

    # Outlier removal
    df_inliers = fit_and_remove_outliers(df_clean)
    print(f"{label}: {len(df_inliers)} inliers kept after outlier removal.")

    # Strip parentheses from "Processor" column for labeling
    df_inliers["ProcessorStripped"] = df_inliers["Processor"].apply(remove_parentheses)

    # Store for plotting
    inliers_list.append((label, df_inliers, color, marker))

# ---------------------------------------------------
# 4) Plot all data on one figure
# ---------------------------------------------------
plt.style.use(hep.style.CMS)
fig, ax = plt.subplots()

ax.set_yscale("log")
ax.set_xlabel("Year of Introduction")
ax.set_ylabel("Transistor Count (log scale)")
ax.set_title("Moore's Law: Transistor Count (CPUs & GPUs)")

for label, df_inliers, color, marker in inliers_list:
    ax.scatter(
        df_inliers["YearNumeric"],
        df_inliers["TransistorCountNumeric"],
        alpha=0.7,
        color=color,
        marker=marker,
        label=label
    )

# 5.5.a) Gather all inlier points so adjustText knows to avoid them
all_x = []
all_y = []

for label, df_inliers, color, marker in inliers_list:
    all_x.extend(df_inliers["YearNumeric"].values)
    all_y.extend(df_inliers["TransistorCountNumeric"].values)

plt.grid(True, which="both", linestyle="--", alpha=0.7)
ax.legend(loc="upper left")
ax.set_ylim(1e3, 9.9e11)  # optional y-limit
ax.set_xlim(1968, 2029)  # optional x-limit

# ---------------------------------------------------
# 5) Binning in log10() space, label one point per bin
# ---------------------------------------------------
texts = []

# We'll define how many bins we want in log space
num_bins = 8  # pick a suitable number of bins for labeling

for label, df_inliers, color, marker in inliers_list:

    if df_inliers.empty:
        continue

    # 5.1) Compute log10 of transistor counts
    log_counts = np.log10(df_inliers["TransistorCountNumeric"])

    # 5.2) Create equally spaced bin edges from min->max
    bin_edges = np.linspace(log_counts.min(), log_counts.max(), num_bins)

    # 5.3) Digitize each row to see which bin it belongs to
    df_inliers["log_bin"] = np.digitize(log_counts, bin_edges)

    # 5.4) Sample 1 row from each bin
    # e.g. if a bin is empty, groupby().apply() won't yield a row
    df_bins_sampled = (
        df_inliers.groupby("log_bin", group_keys=False)
        .apply(lambda grp: grp.sample(n=1, random_state=12))
        .reset_index(drop=True)
    )

    # Now we have at most 1 point from each bin, giving up to num_bins points
    for _, row in df_bins_sampled.iterrows():
        x_val = row["YearNumeric"]
        y_val = row["TransistorCountNumeric"]
        txt = row["ProcessorStripped"]
        t = ax.text(x_val, y_val, txt, fontsize=14)
        texts.append(t)

# 5.5) Use adjustText to avoid overlaps among *all* labels
adjust_text(
    texts,
    x=all_x,      
    y=all_y,
    arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
    expand_points=(10, 10),
    force_points=(10, 10),
    force_static=(10, 10)
)

# ---------------------------------------------------
# 6) Author text & show
# ---------------------------------------------------
ax.text(
    0.95, 0.05,
    "Authors: F. Pantaleo, A. Perego, CERN, 03/2025\nData source: Wikipedia",
    color="gray",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9
)

plt.tight_layout()
plt.savefig("moore_law.pdf")
plt.show()
