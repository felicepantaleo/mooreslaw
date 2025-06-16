import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# helper: convert a title to filename-friendly string
def urlify(s):
    s = re.sub(r"[^\w\s]", '', s)
    return re.sub(r"\s+", '-', s)

# try HEP‐CMS style if installed
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

# ------------------------------------------------------------------
# Numbers come from public spec sheets;
# use None where not available / not supported.
# ------------------------------------------------------------------
data = [
    # NVIDIA Gaming
    {"GPU": "NVIDIA GTX 1080", "Year": 2016, "Segment": "Gaming",
     "FP32_TFLOPS": 8.87,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": None,      "INT8_TC": None,      "FP64_TFLOPS": 0.28,
     "Memory_GB": 8,       "Mem_BW_GBs": 320,    "Price_USD": 599},
    {"GPU": "NVIDIA RTX 2080", "Year": 2018, "Segment": "Gaming",
     "FP32_TFLOPS": 10.1,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 81.3,      "INT8_TC": 327,       "FP64_TFLOPS": 0.31,
     "Memory_GB": 8,       "Mem_BW_GBs": 448,    "Price_USD": 699},
    {"GPU": "NVIDIA RTX 3080", "Year": 2020, "Segment": "Gaming",
     "FP32_TFLOPS": 29.8,  "TF32_TC": 149,       "BF16_TC": 149,
     "FP16_TC": 238,       "INT8_TC": 475,       "FP64_TFLOPS": 0.93,
     "Memory_GB": 10,      "Mem_BW_GBs": 760,    "Price_USD": 699},
    {"GPU": "NVIDIA RTX 4080", "Year": 2022, "Segment": "Gaming",
     "FP32_TFLOPS": 48.7,  "TF32_TC": 390,       "BF16_TC": 780,
     "FP16_TC": 780,       "INT8_TC": 1559,      "FP64_TFLOPS": 0.76,
     "Memory_GB": 16,      "Mem_BW_GBs": 717,    "Price_USD": 1199},
    {"GPU": "NVIDIA RTX 4090", "Year": 2022, "Segment": "Gaming",
     "FP32_TFLOPS": 82.6,  "TF32_TC": 661,       "BF16_TC": 1323,
     "FP16_TC": 1323,      "INT8_TC": 2646,      "FP64_TFLOPS": 1.29,
     "Memory_GB": 24,      "Mem_BW_GBs": 1008,   "Price_USD": 1599},
    # AMD Gaming
    {"GPU": "AMD RX 6800 XT", "Year": 2020, "Segment": "Gaming",
     "FP32_TFLOPS": 20.7,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": None,      "INT8_TC": None,      "FP64_TFLOPS": 0.0013,
     "Memory_GB": 16,      "Mem_BW_GBs": 512,    "Price_USD": 579},
    {"GPU": "AMD RX 7900 XTX", "Year": 2022, "Segment": "Gaming",
     "FP32_TFLOPS": 61.4,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 122.8,     "INT8_TC": 245.6,     "FP64_TFLOPS": 1.92,
     "Memory_GB": 24,      "Mem_BW_GBs": 960,    "Price_USD": 999},
    # NVIDIA Datacenter Small
    {"GPU": "NVIDIA T4", "Year": 2018, "Segment": "Datacenter Small",
     "FP32_TFLOPS": 8.1,   "TF32_TC": None,      "BF16_TC": 65,
     "FP16_TC": 130,       "INT8_TC": 260,       "FP64_TFLOPS": 0.25,
     "Memory_GB": 16,      "Mem_BW_GBs": 320,    "Price_USD": 2500},
    {"GPU": "NVIDIA L4", "Year": 2023, "Segment": "Datacenter Small",
     "FP32_TFLOPS": 30.3,  "TF32_TC": 242,       "BF16_TC": 485,
     "FP16_TC": 485,       "INT8_TC": 970,       "FP64_TFLOPS": 0.47,
     "Memory_GB": 24,      "Mem_BW_GBs": 300,    "Price_USD": 2600},
    # NVIDIA Datacenter Large
    {"GPU": "NVIDIA P100", "Year": 2016, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 10.0,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 21,        "INT8_TC": None,      "FP64_TFLOPS": 5.0,
     "Memory_GB": 16,      "Mem_BW_GBs": 720,    "Price_USD": 5699},
    {"GPU": "NVIDIA V100", "Year": 2017, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 14.1,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 125,       "INT8_TC": 250,       "FP64_TFLOPS": 7.07,
     "Memory_GB": 16,      "Mem_BW_GBs": 900,    "Price_USD": 10664},
    {"GPU": "NVIDIA A100", "Year": 2020, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 19.5,  "TF32_TC": 156,       "BF16_TC": 312,
     "FP16_TC": 312,       "INT8_TC": 624,       "FP64_TFLOPS": 9.75,
     "Memory_GB": 40,      "Mem_BW_GBs": 1555,   "Price_USD": 10000},
    {"GPU": "NVIDIA H100", "Year": 2023, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 51.2,  "TF32_TC": 989,       "BF16_TC": 1979,
     "FP16_TC": 1979,      "INT8_TC": 3959,      "FP64_TFLOPS": 26.0,
     "Memory_GB": 80,      "Mem_BW_GBs": 2039,   "Price_USD": 25000},
    {"GPU": "NVIDIA L40", "Year": 2023, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 90.5,  "TF32_TC": 724,       "BF16_TC": 1448,
     "FP16_TC": 1448,      "INT8_TC": 2896,      "FP64_TFLOPS": 0.0014,
     "Memory_GB": 48,      "Mem_BW_GBs": 912,    "Price_USD": 8000},
    {"GPU": "NVIDIA B200", "Year": 2025, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 62.1,  "TF32_TC": 2200,      "BF16_TC": 4400,
     "FP16_TC": 4400,      "INT8_TC": 8800,      "FP64_TFLOPS": 34.0,
     "Memory_GB": 192,     "Mem_BW_GBs": 4800,   "Price_USD": 60000},
    # AMD Datacenter Large
    {"GPU": "AMD MI100", "Year": 2020, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 23.1,  "TF32_TC": None,      "BF16_TC": 184.6,
     "FP16_TC": 184.6,     "INT8_TC": 368,       "FP64_TFLOPS": 11.5,
     "Memory_GB": 32,      "Mem_BW_GBs": 1228,   "Price_USD": 6400},
    {"GPU": "AMD MI250X", "Year": 2021, "Segment": "Datacenter Large",
     "FP32_TFLOPS": 47.9,  "TF32_TC": None,      "BF16_TC": 383,
     "FP16_TC": 383,       "INT8_TC": 1532,      "FP64_TFLOPS": 47.9,
     "Memory_GB": 128,     "Mem_BW_GBs": 3276,   "Price_USD": 12000},
    # NVIDIA Workstation
    {"GPU": "NVIDIA RTX 6000 Turing", "Year": 2018, "Segment": "Workstation",
     "FP32_TFLOPS": 16.3,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 130,       "INT8_TC": 260,       "FP64_TFLOPS": 0.509,
     "Memory_GB": 24,      "Mem_BW_GBs": 672,    "Price_USD": 6300},
    {"GPU": "NVIDIA RTX A6000", "Year": 2020, "Segment": "Workstation",
     "FP32_TFLOPS": 38.7,  "TF32_TC": 309,       "BF16_TC": 309,
     "FP16_TC": 309,       "INT8_TC": 618,       "FP64_TFLOPS": 0.605,
     "Memory_GB": 48,      "Mem_BW_GBs": 768,    "Price_USD": 4999},
    {"GPU": "NVIDIA RTX 6000 Ada", "Year": 2023, "Segment": "Workstation",
     "FP32_TFLOPS": 91.1,  "TF32_TC": 728,       "BF16_TC": 1454,
     "FP16_TC": 1454,      "INT8_TC": 2908,      "FP64_TFLOPS": 1.423,
     "Memory_GB": 48,      "Mem_BW_GBs": 960,    "Price_USD": 6800},
    # AMD Workstation
    {"GPU": "AMD Radeon Pro W6800", "Year": 2021, "Segment": "Workstation",
     "FP32_TFLOPS": 17.8,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": None,      "INT8_TC": None,      "FP64_TFLOPS": 0.0011,
     "Memory_GB": 32,      "Mem_BW_GBs": 512,    "Price_USD": 2249},
    {"GPU": "AMD Radeon Pro W7900", "Year": 2023, "Segment": "Workstation",
     "FP32_TFLOPS": 61.3,  "TF32_TC": None,      "BF16_TC": None,
     "FP16_TC": 122.6,     "INT8_TC": 245.2,     "FP64_TFLOPS": 1.916,
     "Memory_GB": 48,      "Mem_BW_GBs": 960,    "Price_USD": 3999},
]

# create DataFrame
df = pd.DataFrame(data)
df["Price_per_FP32"] = df["Price_USD"] / df["FP32_TFLOPS"]
df["Price_per_FP64"] = df["Price_USD"] / df["FP64_TFLOPS"]
df["Price_per_TF32_TC"] = df["Price_USD"] / df["TF32_TC"]
# plotting helpers
marker_map = {"Gaming": "o", "Datacenter Small": "v", "Datacenter Large": "s", "Workstation": "^"}
color_map = {"Gaming": "#1f77b4", "Datacenter Small": "#ff7f0e", "Datacenter Large": "#2ca02c", "Workstation": "#d62728"}

def make_scatter(metric, title, ylabel, logy=False):
    fig, ax = plt.subplots()
    if logy:
        ax.set_yscale("log")
    for segment, seg_df in df.groupby("Segment"):
        ax.scatter(seg_df["Year"], seg_df[metric],
                   label=segment, marker=marker_map[segment],
                   color=color_map[segment], alpha=0.8, s=60)
    ax.set_xlabel("Year of Introduction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    for _, row in df.iterrows():
        if not (np.isnan(row["Year"]) or np.isnan(row[metric])):
            ax.text(row["Year"]+0.15, row[metric], row["GPU"], fontsize=10, va="bottom")
    ax.text(
    0.95, 0.05,
    "F. Pantaleo, CERN, 06/2025",
    color="gray",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10)
    plt.tight_layout()
    plt.savefig(urlify(title)+".pdf")
    plt.show()

# generate plots
make_scatter("FP32_TFLOPS", "FP32 Throughput Evolution", "FP32 TFLOPS")
make_scatter("TF32_TC", "Tensor TF32 Performance", "TF32 Tensor TFLOPS", logy=True)
make_scatter("BF16_TC", "Tensor BFLOAT16 Performance", "BF16 Tensor TFLOPS", logy=True)
make_scatter("FP16_TC", "Tensor FP16 Performance", "FP16 Tensor TFLOPS", logy=True)
make_scatter("INT8_TC", "Tensor INT8 Performance", "INT8 TOPS", logy=True)
make_scatter("FP64_TFLOPS", "FP64 Throughput Evolution", "FP64 TFLOPS", logy=True)
make_scatter("Memory_GB", "GPU Memory Size Evolution", "Memory (GB)")
make_scatter("Mem_BW_GBs", "Memory Bandwidth Evolution", "Bandwidth (GB/s)", logy=True)
make_scatter("Price_per_FP32", "Cost per FP32 TFLOP", "USD per FP32 TFLOP", logy=True)
make_scatter("Price_per_FP64", "Cost per FP64 TFLOP", "USD per FP64 TFLOP", logy=True)
make_scatter("Price_per_TF32_TC", "Cost per Tensor TF32 TFLOP", "USD per TF32 Tensor TFLOP", logy=True)
make_scatter("Price_USD", "GPU Price Evolution", "Price (USD)", logy=True)