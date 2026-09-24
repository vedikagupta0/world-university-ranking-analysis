"""
eda_pipeline.py
---------------
End-to-end, reproducible EDA pipeline for the Times Higher Education World University Rankings dataset .

Handles the dataset's real-world quirks:
  * Banded overall scores ("39.0-43.5") -> numeric midpoints
  * Formatted stats ("22,005", "43%", "52 : 48") -> numeric

Outputs summary tables to stdout and exports charts to the images/ folder.

Usage:
    python src/eda_pipeline.py --data dataset world_university_rankings_2191.csv --out images/
"""

import argparse
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SKY_BLUE = "#87CEEB"
sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": SKY_BLUE,
    "grid.alpha": 0.35,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "font.size": 10,
})

THEME_COLORS = {
    "primary": "#87CEEB",
    "secondary": "#8B0000",
    "background": "#FFFFFF",
}

SCORE_COLS = [
    "scores_teaching",
    "scores_research",
    "scores_citations",
    "scores_industry_income",
    "scores_international_outlook",
]

PRETTY = {
    "scores_overall_num": "Overall",
    "scores_teaching": "Teaching",
    "scores_research": "Research",
    "scores_citations": "Citations",
    "scores_industry_income": "Industry Income",
    "scores_international_outlook": "Intl. Outlook",
    "students": "Student Population",
    "student_staff_ratio": "Student:Staff Ratio",
    "pc_intl_students": "% Intl. Students",
    "pc_female": "% Female",
}


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_data(path: str, DATA_DIR = "dataset") -> pd.DataFrame:
    """Load the raw rankings CSV."""
    return pd.read_csv(os.path.join(DATA_DIR, path))


def _band_to_midpoint(value) -> float:
    """
    Convert a banded string to a numeric midpoint.

    "39.0-43.5" -> 41.25 | "1501+" -> 1501 | "12%" -> 12
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("=", "")
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash -> hyphen
    if s.endswith("+"):
        s = s[:-1]
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return np.nan
    return float(np.mean([float(n) for n in nums]))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ranks, scores, and institutional stats into numeric columns."""
    df = df.copy()

    # Only currently-listed institutions
    for flag in ("closed", "unaccredited", "disabled"):
        if flag in df.columns:
            df = df[df[flag].fillna(False) != True]  # noqa: E712

    df["scores_overall_num"] = df["scores_overall"].apply(_band_to_midpoint)
    df['rank_num'] = df['rank'].apply(_band_to_midpoint)
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["students"] = pd.to_numeric(
        df["stats_number_students"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df["student_staff_ratio"] = pd.to_numeric(df["stats_student_staff_ratio"], errors="coerce")

    df["pc_intl_students"] = pd.to_numeric(
        df["stats_pc_intl_students"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )
    df["pc_female"] = pd.to_numeric(
        df["stats_female_male_ratio"].astype(str).str.split(":").str[0],
        errors="coerce",
    )
    df.drop(columns=['rank_order', 'scores_overall_rank',
            'scores_teaching_rank', 'scores_research_rank', 'scores_citations_rank', 'scores_industry_income_rank', 'scores_international_outlook_rank', 'iid', 'record_type', 'member_level', 'url', 'nid',
             'aliases', 'closed', 'unaccredited', 'disabled', 'apply_link'], inplace=True, errors='ignore')
    df = df.rename(columns={"name": "university", "location": "country"})
    keep = [
        "rank", "rank_num", "university", "country", "scores_overall_num",
        *SCORE_COLS, "students", "student_staff_ratio", "pc_intl_students", "pc_female",
    ]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for all numeric analysis columns."""
    return df.select_dtypes(include=[np.number]).describe().T.round(2)


def country_summary(df: pd.DataFrame, min_universities: int = 10) -> pd.DataFrame:
    """
    Per-country institution count and mean scores, restricted to countries with
    at least `min_universities` ranked institutions (avoids small-sample noise).
    """
    agg = (
        df.groupby("country")
        .agg(
            universities=("university", "count"),
            mean_overall=("scores_overall_num", "mean"),
            mean_teaching=("scores_teaching", "mean"),
            mean_research=("scores_research", "mean"),
            mean_citations=("scores_citations", "mean"),
            mean_industry=("scores_industry_income", "mean"),
            mean_intl=("scores_international_outlook", "mean"),
        )
        .round(2)
    )
    return agg[agg["universities"] >= min_universities].sort_values(
        "mean_overall", ascending=False
    )


def pillar_correlations(df: pd.DataFrame) -> pd.Series:
    """Correlation of each ranking pillar with the overall score."""
    return (
        df[SCORE_COLS + ["scores_overall_num"]]
        .corr()["scores_overall_num"]
        .drop("scores_overall_num")
        .sort_values(ascending=False)
        .round(3)
    )


def elite_vs_rest(df: pd.DataFrame, n_elite: int = 100) -> pd.DataFrame:
    """Mean pillar scores for the top-N universities vs everyone else."""
    df = df.sort_values("rank_num")
    elite = df.head(n_elite)
    rest = df.iloc[n_elite:]
    out = pd.DataFrame(
        {
            f"Top {n_elite}": elite[SCORE_COLS + ["student_staff_ratio", "pc_intl_students"]].mean(),
            "Rest": rest[SCORE_COLS + ["student_staff_ratio", "pc_intl_students"]].mean(),
        }
    ).round(2)
    out["Gap"] = (out[f"Top {n_elite}"] - out["Rest"]).round(2)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_top_universities(df: pd.DataFrame, out_dir: str, n: int = 20):
    top = df.nsmallest(n, "rank_num")
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=top, y="university", x="scores_overall_num",
        color=THEME_COLORS["primary"], legend=False
    )
    plt.title(f"Top {n} Universities by Overall Score", fontsize=14, weight="bold")
    plt.xlabel("Overall Score")
    plt.ylabel("")
    plt.xlim(85, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top20_universities.png"), dpi=150)
    plt.close()


def plot_country_counts(df: pd.DataFrame, out_dir: str, n: int = 15):
    counts = df["country"].value_counts().head(n)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=counts.values, y=counts.index,
        color=THEME_COLORS["primary"], legend=False
    )
    plt.title(
        f"Top {n} Countries by Number of Ranked Universities",
        fontsize=13, weight="bold"
    )
    plt.xlabel("Ranked Universities")
    plt.ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=5, fontsize=10, fontweight="bold")
    plt.xlim(0, counts.max() * 1.12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "country_counts.png"), dpi=150)
    plt.close()


def plot_country_avg_score(df: pd.DataFrame, out_dir: str, n: int = 15):
    cs = country_summary(df).head(n)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=cs["mean_overall"], y=cs.index,
        color=THEME_COLORS["primary"], legend=False
    )
    plt.title(
        f"Top {n} Countries by Average Overall Score\n"
        "(10+ ranked universities)",
        fontsize=13, weight="bold"
    )
    plt.xlabel("Mean Overall Score")
    plt.ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5, fontsize=10, fontweight="bold")
    plt.xlim(0, cs["mean_overall"].max() * 1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "country_avg_score.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    cols = [
        "scores_overall_num", *SCORE_COLS, "students",
        "student_staff_ratio", "pc_intl_students", "pc_female"
    ]
    corr = df[cols].corr()
    corr.index = [PRETTY.get(c, c) for c in corr.index]
    corr.columns = [PRETTY.get(c, c) for c in corr.columns]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Ranking Metrics vs Institutional Traits", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_staff_ratio_vs_score(df: pd.DataFrame, out_dir: str):
    sub = df.dropna(subset=["student_staff_ratio", "scores_overall_num"])
    sub = sub[sub["student_staff_ratio"] < 100]
    r = sub["student_staff_ratio"].corr(sub["scores_overall_num"])

    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=sub, x="student_staff_ratio", y="scores_overall_num",
        scatter_kws={"alpha": 0.25, "s": 18},
        line_kws={"color": THEME_COLORS["secondary"]}
    )
    plt.title(
        f"Student-to-Staff Ratio vs Overall Score  (r = {r:.2f})",
        fontsize=13, weight="bold"
    )
    plt.xlabel("Students per Staff Member")
    plt.ylabel("Overall Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "staff_ratio_vs_score.png"), dpi=150)
    plt.close()


def plot_pillar_distributions(df: pd.DataFrame, out_dir: str):
    long = df[SCORE_COLS].melt(var_name="Pillar", value_name="Score")
    long["Pillar"] = long["Pillar"].map(PRETTY)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=long, x="Score", y="Pillar", hue="Pillar",
        color=THEME_COLORS["primary"], legend=False
    )
    plt.title(
        "Score Distributions Across the Five Pillars",
        fontsize=13, weight="bold"
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pillar_distributions.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    matplotlib.use("Agg")  
    p = argparse.ArgumentParser(description="THE World University Rankings EDA pipeline")
    p.add_argument("--data", default="dataset/world_university_rankings_2191.csv")
    p.add_argument("--out", default="images/")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = clean_data(load_data(args.data))
    print(f"Cleaned dataset: {len(df)} universities across {df['country'].nunique()} countries\n")

    print("=== Summary statistics ===")
    print(summary_stats(df), "\n")

    print("=== Correlation of each pillar with overall score ===")
    print(pillar_correlations(df), "\n")

    print("=== Top 10 countries by average overall score (10+ universities) ===")
    print(country_summary(df).head(10), "\n")

    print("=== Top 100 vs the rest ===")
    print(elite_vs_rest(df), "\n")

    plot_top_universities(df, args.out)
    plot_country_counts(df, args.out)
    plot_country_avg_score(df, args.out)
    plot_correlation_heatmap(df, args.out)
    plot_staff_ratio_vs_score(df, args.out)
    plot_pillar_distributions(df, args.out)
    print(f"6 charts written to {args.out}")


if __name__ == "__main__":
    main()
