import pandas as pd
import numpy as np
import ast

############# Cleaning the Datasets Intensities.csv and Mixtures.csv #############

# --------------------------------------------------------
# - adding a primary key to Mixtures.csv
# - calculate dominance ratio
# - label Dominant or Balanced mixture
# --------------------------------------------------------

# Path to the CSV (relative to where you run the script)
mixture_csv_path = "demo/Mixture.csv"
intensity_csv_path = "demo/Intensities.csv"

# Read the  CSV
mixture_df = pd.read_csv(mixture_csv_path)
intensity_df = pd.read_csv(intensity_csv_path)

# ----------------------------------
# Adding Primary Key to Mixtures.csv
# ----------------------------------

# Create a new primary key by concatenating Dataset and Mixture Label
mixture_df["mixture_id"] = (
    mixture_df["Dataset"].astype(str) + "_" +
    mixture_df["Mixture Label"].astype(str)
)

# Move the new column to the front
cols = ["mixture_id"] + [c for c in mixture_df.columns if c != "mixture_id"]
mixture_df = mixture_df[cols]

# --------------------------------------------------------
# Extracting Intensities for each molecule in each mixture
# --------------------------------------------------------
intensity_dict = dict(
    zip(intensity_df["CID"], intensity_df["INTENSITY"])
)

metadata_cols = ["mixture_id", "Dataset", "Mixture Label"]
cid_cols = [c for c in mixture_df.columns if c not in metadata_cols]

rows = []

for _, row in mixture_df.iterrows():
    mixture_id = row["mixture_id"]

    intensities = []

    for cid_col in cid_cols:
        cid = row[cid_col]

        # Skip empty or placeholder values
        if pd.isna(cid) or cid == 0:
            continue

        # Only keep molecules with known intensity
        if cid in intensity_dict:
            intensities.append(intensity_dict[cid])

    # -----------------------------
    # IMPORTANT: skip mixtures with < 2 molecules
    # -----------------------------
    if len(intensities) < 2:
        continue

    # -----------------------------
    # Compute dominance ratio
    # -----------------------------
    sorted_intensities = sorted(intensities, reverse=True)
    max_intensity = sorted_intensities[0]
    second_max_intensity = sorted_intensities[1]

    dominance_ratio = max_intensity / second_max_intensity

    # Dominant vs Balanced label
    # 1.08 threshold value to capture roughly top 10% of dominant mixtures - can be changed
    mixture_type = "Dominant" if dominance_ratio > 1.08 else "Balanced"

    rows.append({
        "mixture_id": mixture_id,
        "Intensities": intensities,
        "dominance_ratio": dominance_ratio,
        "mixture_type": mixture_type
    })

# ----------------------------------------------------------
# Create DataFrame and save
# ----------------------------------------------------------
# Final dataset will have columns:
# Mixture_ID | Intensity | Dominance Ratio | Mixture_Type
#            |           |                 |
#            |           |                 |
# ----------------------------------------------------------
# Note: 
# - Mixture_ID is a string concatenation of Dataset and
# Mixture Label from Mixture.csv with separator '_'
# - Intensity is an array of the intensities of each 
# molecule present in the mixture
# - Dominance Ratio is the ratio of highest intensity to
# second highest intensity for a mixture
# - Mixture_Type is "Dominant" or "Balanced" depending if
# dominance ratio is greater than a threshold, e.g. 1.08
# ----------------------------------------------------------
# See mixture_intensities_with_dominance.csv
# ----------------------------------------------------------
mixture_intensity_df = pd.DataFrame(rows)

output_path = "demo/mixture_intensities_with_dominance.csv"
mixture_intensity_df.to_csv(output_path, index=False)


########################## Adaptive Tau ##########################

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("demo/mixture_intensities_with_dominance.csv")

# Intensities column comes back as string -> convert to list
df["Intensities"] = df["Intensities"].apply(ast.literal_eval)

# -----------------------------
# Weighting function (softMax)
# -----------------------------
def softmax_weights(intensities, tau):
    x = np.array(intensities, dtype=float)
    x = x - np.max(x) # numerical stability
    exp_x = np.exp(x / tau)
    return exp_x / exp_x.sum()


# ----------------------------------------------------------------------------------
# Fixed tau grid search
# ----------------------------------------------------------------------------------
# find the tau with lowest separation score, where
# separation score = mean(dominant mixture weights) - mean(balanced mixture weights)
# ----------------------------------------------------------------------------------
def evaluate_fixed_tau(df, tau):
    top_weights = []

    for intensities in df["Intensities"]:
        w = softmax_weights(intensities, tau)
        top_weights.append(np.max(w))

    df_tmp = df.copy()
    df_tmp["top_weight"] = top_weights

    dominant_mean = df_tmp[df_tmp["mixture_type"] == "Dominant"]["top_weight"].mean()
    balanced_mean = df_tmp[df_tmp["mixture_type"] == "Balanced"]["top_weight"].mean()

    separation_score = dominant_mean - balanced_mean
    return separation_score, dominant_mean, balanced_mean


best_tau = None
best_score = -np.inf
fixed_results = []

for tau in range(1, 51):
    score, dom_mean, bal_mean = evaluate_fixed_tau(df, tau)
    fixed_results.append((tau, score, dom_mean, bal_mean))

    if score > best_score:
        best_score = score
        best_tau = tau

# best_tau is tau with the largest separation score
# large separation score = tau can classify dominant vs balanced mixture
print(f"Best fixed tau: {best_tau}")


# --------------------------------
# Fixed tau top weights (best tau)
# --------------------------------
fixed_top_weights = []

for intensities in df["Intensities"]:
    w = softmax_weights(intensities, best_tau)
    fixed_top_weights.append(np.max(w))

df["top_weight_fixed"] = fixed_top_weights


# ------------------------------------------------------------
# Adaptive tau computation for Dominant Mixtures
# ------------------------------------------------------------
# tau = clip(1/(IQR(intensities) + epislon), tau_min, tau_max)
# returns tau_min if 1/(IQR(intensities) + epislon) < tau_min
# returns tau_max if 1/(IQR(intensities) + epislon) > tau_max
# else returns 1/(IQR(intensities) + epislon)
# -------------------------------------------------------------
def tau_dominant_inverse_iqr(intensities, epsilon=1e-3, tau_min=1.0, tau_max=50.0):
    intensities = np.array(intensities, dtype=float)
    iqr = np.percentile(intensities, 75) - np.percentile(intensities, 25)
    tau = 1.0 / (iqr + epsilon)
    return float(np.clip(tau, tau_min, tau_max))


# ----------------------------------------------
# Adaptive tau computation for Balanced Mixtures
# ----------------------------------------------
# tau = tau0 + smoothing factor
# tau0 - baseline smoothing, bare minimum
# c - controls the strength of smoothing
# delta0 prevents division by 0
# ----------------------------------------------
def tau_balanced_ratio_smoother(
    dominance_ratio,
    tau0=2.0,
    c=0.6,
    delta0=0.02,
    tau_min=3.0,
    tau_max=12.0
):
    delta = float(dominance_ratio) - 1.0
    tau = tau0 + (c / (delta + delta0))
    return float(np.clip(tau, tau_min, tau_max))


# ----------------------------------------------
# Determine which adaptive tau to use depending
# on Dominant or Balanced mixture
# ----------------------------------------------
def tau_piecewise(intensities, dominance_ratio, mixture_type):
    if mixture_type == "Dominant":
        # change epsilon, tau_min, tau_max here
        return tau_dominant_inverse_iqr(intensities, epsilon=1e-3, tau_min=1.0, tau_max=50.0)
    else:
        # change tau0, c, and delta0 here
        return tau_balanced_ratio_smoother(
            dominance_ratio,
            tau0=2.0, c=0.6, delta0=0.02,
            tau_min=3.0, tau_max=12.0
        )


adaptive_top_weights = []
adaptive_taus = []

for _, row in df.iterrows():
    intensities = row["Intensities"]
    dominance_ratio = row["dominance_ratio"]
    mtype = row["mixture_type"]

    tau = tau_piecewise(intensities, dominance_ratio, mtype)

    w = softmax_weights(intensities, tau)
    adaptive_taus.append(tau)
    adaptive_top_weights.append(float(np.max(w)))

df["tau_adaptive"] = adaptive_taus
df["top_weight_adaptive"] = adaptive_top_weights

print("\n=== Adaptive Tau Diagnostics ===")

print("\nOverall tau_adaptive summary:")
print(df["tau_adaptive"].describe())

print("\nNumber of unique tau_adaptive values:")
print(df["tau_adaptive"].nunique())

print("\n10 Most common tau_adaptive values:")
print(df["tau_adaptive"].value_counts().head(10))

print("\nTau summary by mixture type:")
print(
    df.groupby("mixture_type")["tau_adaptive"]
      .describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]]
)


# -----------------------------
# Final summary
# -----------------------------
summary = df.groupby("mixture_type")[["top_weight_fixed", "top_weight_adaptive"]].mean()
print()
print(summary)
