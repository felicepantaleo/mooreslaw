import requests
import re

import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep

url = "https://en.wikipedia.org/wiki/Transistor_count"
df_list = pd.read_html(url)

# df_list is now a list of DataFrames, one for each HTML table found on the page
print(f"Found {len(df_list)} tables on the page.")

# # For instance, let's look at the first table:
# for i, df in enumerate(df_list):
#     print(f"### Table {i} ###")
#     print(df.head())

df = df_list[3]
print(df.head())

def parse_transistor_count(raw_str: str):
  """
  Remove bracketed references, remove commas, extract the largest integer found.
  E.g.: "74,442 (5,360 excl. ROM & RAM)[14][15]" -> 74442
  If no valid number is found, returns None.
  """
  # Remove bracketed references [14][15], [16], etc.
  cleaned_str = re.sub(r"\[.*?\]", "", raw_str)
  # Remove spaces around parentheses to unify
  cleaned_str = cleaned_str.replace(",", "")
  # Extract all sequences of digits
  numbers = re.findall(r"\d+", cleaned_str)
  if not numbers:
    return None
  # If there are multiple numbers (e.g., "74,442 (5360 excl. ...)")
  # we’ll take the largest one. You could choose the first instead.
  return min(int(num) for num in numbers)

def parse_year(raw_str: str):
  """
  Remove bracketed references, parentheses, etc., and extract a 4-digit year.
  E.g.: "1970[12][a]" -> 1970
  If no valid year is found, returns None.
  """
  cleaned_str = re.sub(r"\[.*?\]", "", raw_str)   # remove bracketed refs
  cleaned_str = re.sub(r"\(.*?\)", "", cleaned_str)  # remove parentheses
  cleaned_str = cleaned_str.strip()
  # Find 4-digit numbers
  match = re.search(r"\b(19|20)\d{2}\b", cleaned_str)
  if match:
    return int(match.group(0))
  return None

# --- 2) Apply these parsing functions to create numeric columns ---

df["TransistorCountNumeric"] = df["Transistor count"].apply(parse_transistor_count)
df["YearNumeric"] = df["Year"].apply(parse_year)

# --- 3) Drop rows with missing values in the numeric columns (if any) ---

df_clean = df.dropna(subset=["TransistorCountNumeric", "YearNumeric"])

# --- 4) Plot transistor count vs. year on a log scale ---

plt.style.use(hep.style.CMS)
plt.figure()
plt.scatter(df_clean["YearNumeric"], df_clean["TransistorCountNumeric"], marker='o')
plt.yscale('log')
plt.xlabel("Year of Introduction")
plt.ylabel("Transistor Count (log scale)")
plt.title("Moore's Law: Transistor Count Over Time ")
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.savefig("moore_law.png")



import matplotlib.pyplot as plt
from adjustText import adjust_text

fig, ax = plt.subplots()

ax.scatter(df_clean["YearNumeric"], df_clean["TransistorCountNumeric"])
ax.set_yscale("log")

texts = []
for _, row in df_clean.sample(20).iterrows():
    x = row["YearNumeric"]
    y = row["TransistorCountNumeric"]
    txt = row["Processor"]
    texts.append(ax.text(x, y, txt, fontsize=8))

adjust_text(
    texts,
    x=df_clean.sample(20)["YearNumeric"],  # or use the same points from above
    y=df_clean.sample(20)["TransistorCountNumeric"],
    arrowprops=dict(arrowstyle="->", color='black', lw=0.5)
)

plt.xlabel("Year of Introduction")
plt.ylabel("Transistor Count (log scale)")
plt.title("Moore's Law: Transistor Count Over Time")
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("moore_law_2.png")


def remove_parentheses(text):
    """
    Repeatedly remove innermost parentheses and their contents
    until no more parentheses remain.
    E.g. 
      "(multi-chip module, 24 cores, 128 GB GPU memory + 256 MB (LLC/L3) cache)"
    becomes "" (completely removed, if it’s all inside parentheses).
    """
    while True:
        # This pattern finds pairs of parentheses containing no further parentheses inside
        new_text = re.sub(r"\([^()]*\)", "", text)
        if new_text == text:
            # No further changes, so we have removed everything we can
            break
        text = new_text
    return text.strip()


import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Suppose your cleaned DataFrame is df_clean, with columns:
#   "YearNumeric", "TransistorCountNumeric", "Processor"

# 1) Compute log10 of the transistor counts:
log_counts = np.log10(df_clean["TransistorCountNumeric"])

# 2) Create bins spanning the range of log10(count):
#    e.g. 8 bins (this is arbitrary—adjust as you like)
num_bins = 24
bin_edges = np.linspace(log_counts.min(), log_counts.max(), num_bins)

# 3) Assign each row to a bin
df_clean["log_bin"] = np.digitize(log_counts, bin_edges)

# 4) From each bin, pick 1 row at random (or more if you want)
df_bins_sampled = df_clean.groupby("log_bin", group_keys=False).apply(
    lambda x: x.sample(n=1, random_state=42)  # pick 1 per bin
)

# This ensures you have at most 1 point from each bin => up to 8 total.
# If your dataset is large, you could pick 2-3 in each bin for more coverage.

# Now we have a smaller subset of points that’s spread across the full log range.
# Let’s label those with adjustText.

fig, ax = plt.subplots()
ax.scatter(df_clean["YearNumeric"], df_clean["TransistorCountNumeric"], marker='o')

ax.set_yscale("log")
ax.set_xlabel("Year of Introduction")
ax.set_ylabel("Transistor Count (log scale)")
ax.set_title("Moore's Law: Transistor Count Over Time")
ax.set_ylim(1e3, 1e11)  # limit the y-axis to avoid crowding

texts = []
for _, row in df_bins_sampled.iterrows():
    x = row["YearNumeric"]
    y = row["TransistorCountNumeric"]
    txt = row["Processor"]
    # Create a Text artist for each chosen point
    texts.append(ax.text(x, y, remove_parentheses(txt), fontsize=14))

# 5) Let adjustText minimize overlap:
adjust_text(    
    texts,
    x=df_bins_sampled["YearNumeric"],
    y=df_bins_sampled["TransistorCountNumeric"],
    arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
    expand_points=(5, 2),
    force_points=(2, 2),
)

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("moore_law_3.pdf")
