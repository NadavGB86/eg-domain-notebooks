# ==============================================================================
# FILE: supply_chain_pred_core.py
# DOMAIN: Enterprise Supply Chain & Predictive Logistics Framework
# ARCHITECTURE: 7-Layer MLOps System (Utils -> Profiling -> Governance ->
#                                     Cross-Sectional FE -> Longitudinal FE ->
#                                     Modeling -> Operations)
# ==============================================================================
from __future__ import annotations

import copy
import logging
import hashlib
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.tsa.stattools import pacf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score

import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Display & Plot Configuration
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 100,
    'figure.max_open_warning': 0,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,
})
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SC_Core")

# Suppress chatty third-party loggers (Prevent Matplotlib/Seaborn from spamming INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("seaborn").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


# ==============================================================================
# LAYER 0: CORE UTILITIES, VITALITY & VISUALIZATION
# ==============================================================================
class SupplyChainUtils:

    @staticmethod
    def seed32(stage: str, seed: int) -> int:
        b = f"{seed}::{stage}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(b, digest_size=4).digest(), "little")

    @staticmethod
    def rng(stage: str, seed: int) -> np.random.Generator:
        return np.random.default_rng(SupplyChainUtils.seed32(stage, seed))

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Iterate through all columns and modify data type to reduce memory usage."""
        start_mem = df.memory_usage().sum() / 1024 ** 2
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in df.columns:
            col_type = df[col].dtype
            if col_type in numerics:
                c_min, c_max = df[col].min(), df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(
                            np.float32)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: logger.info(
                f'Memory usage reduced to {end_mem:.2f} MB ({(start_mem - end_mem) / start_mem:.1%} reduction)')
        return df

    @staticmethod
    def dataset_health_check(df: pd.DataFrame, sort_by_stat: str = 'std') -> pd.DataFrame:
        """Generates a styled comprehensive summary dataframe for all features."""
        summary = pd.DataFrame(df.dtypes.astype(str), columns=['Dtype'])
        summary['Null_Count'] = df.isnull().sum()
        summary['Null_%'] = (df.isnull().sum() / len(df)) * 100
        summary['Unique_Values'] = df.nunique()
        summary['Cardinality_%'] = (df.nunique() / len(df)) * 100

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            desc = df.describe().transpose()
            summary = summary.join(desc[['min', '25%', '50%', '75%', 'max', 'mean', 'std']], how='left')
            summary.loc[numeric_cols, 'Skewness'] = df[numeric_cols].skew()
            summary['mean'] = pd.to_numeric(summary['mean'], errors='coerce')
            summary['max'] = pd.to_numeric(summary['max'], errors='coerce')
            summary = summary.sort_values(by=['Dtype', sort_by_stat] if numeric_cols else 'Dtype')
            try:
                from IPython.display import display
                display(
                    summary.style.background_gradient(cmap='Reds', subset=['Null_%', 'Cardinality_%']
                                ).background_gradient(cmap='Reds', subset=['Cardinality_%']
                                ).background_gradient(cmap='coolwarm', subset=['Skewness', 'std'], vmin=-3, vmax=3
                                ).bar(color='#d65f5f', subset=['mean', 'max']).format(
                        {'Null_%': "{:.1f}%", 'Cardinality_%': "{:.1f}%", 'mean': "{:.2f}", 'std': "{:.2f}"}))
            except ImportError:
                pass
        return summary

    @staticmethod
    def generate_feature_vitality_report(X_df: pd.DataFrame, y_series: pd.Series,
                                         categorical_cols: list) -> pd.DataFrame:
        """Classical statistical vitality report (Cohen's d & PR-AUC)."""
        records = []
        mask_pos, mask_neg = (y_series == 1), (y_series == 0)
        baseline_pr_auc = y_series.mean()

        for col in X_df.columns:
            if col in categorical_cols: continue
            feature_data = X_df[col].dropna()
            if len(feature_data) == 0: continue
            pos_vals = feature_data[mask_pos.loc[feature_data.index].dropna()]
            neg_vals = feature_data[mask_neg.loc[feature_data.index].dropna()]
            if len(pos_vals) == 0 or len(neg_vals) == 0: continue

            # Cohen's d
            pooled_std = np.sqrt(((len(pos_vals) - 1) * pos_vals.var() + (len(neg_vals) - 1) * neg_vals.var()) / (
                    len(pos_vals) + len(neg_vals) - 2))
            cohens_d = (pos_vals.mean() - neg_vals.mean()) / (pooled_std + 1e-8)

            # Univariate PR-AUC
            try:
                score_data = -feature_data if cohens_d < 0 else feature_data
                pr_auc = average_precision_score(y_series.loc[feature_data.index], score_data)
            except Exception:
                pr_auc = np.nan

            records.append(
                    {'Feature': col, 'Cohens_D': abs(cohens_d), 'Direction': 'Positive' if cohens_d > 0 else 'Negative',
                     'Univariate_PR_AUC': pr_auc, 'Lift_vs_Baseline': pr_auc / baseline_pr_auc if pd.notna(
                            pr_auc) and baseline_pr_auc > 0 else np.nan})
        return pd.DataFrame(records).sort_values(by='Univariate_PR_AUC', ascending=False)

    @staticmethod
    def print_mnar_physics_report(df: pd.DataFrame, target_col: str, feature_col: str) -> None:
        """Restores the detailed textual breakdown of the MNAR physics."""
        print(f"=== MISSING DATA PHYSICS: {feature_col.upper()} ===")
        total_samples = len(df)
        missing_mask = df[feature_col].isna()
        missing_count = int(missing_mask.sum())
        missing_pct = missing_count / total_samples

        print(f"Missing Values: {missing_count:,} ({missing_pct:.2%})")
        if missing_count > 0:
            risk_present = df.loc[~missing_mask, target_col].mean()
            risk_missing = df.loc[missing_mask, target_col].mean()
            risk_multiplier = risk_missing / risk_present if risk_present > 0 else 0

            print(f"Risk when value is PRESENT: {risk_present:.4%}")
            print(f"Risk when value is MISSING: {risk_missing:.4%}")
            print(f"Risk Multiplier (Missing vs Present): {risk_multiplier:.2f}x")

            if risk_multiplier > 1.2 or risk_multiplier < 0.8:
                print(f">>> CONCLUSION: '{feature_col}' exhibits MNAR behavior. Imputation is prohibited.")
            else:
                print(f">>> CONCLUSION: '{feature_col}' exhibits MAR tendencies. Safe for imputation.")

    @staticmethod
    def plot_danger_zone(df: pd.DataFrame, target_col: str, x_col: str = 'min_bank', y_col: str = 'national_inv',
                         seed: int = 42) -> None:
        """Restores Chapter 3: The Danger Zone scatter plot."""
        sample_df = df.sample(n=min(50000, len(df)), random_state=seed)
        sample_df = sample_df[(sample_df[y_col] >= 0) & (sample_df[y_col] <= 200)]
        sample_df = sample_df[(sample_df[x_col] >= 0) & (sample_df[x_col] <= 200)]

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=sample_df, x=x_col, y=y_col, hue=target_col, style=target_col,
                palette={0: 'lightgrey', 1: 'red'}, alpha=0.6, s=30)
        plt.plot([0, 200], [0, 200], 'k--', linewidth=2, label='Safety Boundary (Inv = Min Bank)')
        plt.title("The Danger Zone (Inventory < Min Bank)", fontsize=12, fontweight='bold')
        plt.xlabel("Minimum Recommended Bank (Units)")
        plt.ylabel("Actual National Inventory (Units)")
        plt.legend(title='Backorder', loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_prototyping_proofs(df: pd.DataFrame, target_col: str, inv_col: str, velocity_col: str,
                                        transit_col: str) -> None:
        """Restores Chapter 5: The 2x2 Feature Signal Prototyping (ECDF & Risk Matrix)."""
        poc_df = df[[inv_col, velocity_col, transit_col, target_col]].copy()
        poc_df['velocity_safe'] = poc_df[velocity_col] + 0.01
        poc_df['inventory_runway'] = poc_df[inv_col].clip(lower=0) / poc_df['velocity_safe']
        poc_df['transit_ratio'] = poc_df[transit_col] / poc_df['velocity_safe']

        plot_df_row1 = poc_df[(poc_df[inv_col] >= 0) & (poc_df[inv_col] <= 100) & (poc_df['inventory_runway'] <= 20)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Feature Signal Prototyping: Ratios & Interactions", fontsize=14, fontweight='bold', y=1.02)

        # 1A: Raw ECDF
        sns.ecdfplot(data=plot_df_row1, x=inv_col, hue=target_col, palette={0: 'lightgrey', 1: 'red'}, linewidth=2,
                     ax=axes[0, 0], legend=False)
        axes[0, 0].set_title(f"1A. RAW: {inv_col} (Slow separation)", fontweight='bold')

        # 1B: Engineered ECDF
        sns.ecdfplot(data=plot_df_row1, x='inventory_runway', hue=target_col, palette={0: 'lightgrey', 1: 'red'},
                     linewidth=3, ax=axes[0, 1], legend=False)
        axes[0, 1].set_title("1B. ENGINEERED: Runway (The 50% Proportion Gap)", fontweight='bold', color='teal')
        axes[0, 1].set_xlim(-0.5, 5)
        axes[0, 1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].text(1.2, 0.6, "← ~50% Separation Gap", fontsize=10, fontweight='bold', color='darkred')

        # 2A: Raw 2D Scatter
        interact_df = poc_df[poc_df[transit_col] > 0].copy()
        sns.scatterplot(data=interact_df[interact_df[inv_col] < 100], x=inv_col, y=transit_col, hue=target_col,
                        palette={0: 'lightgrey', 1: 'red'}, alpha=0.5, ax=axes[1, 0], legend=False)
        axes[1, 0].set_title(f"2A. RAW 2D SPACE: Inv vs. Transit", fontweight='bold')
        axes[1, 0].set_ylim(0, 100)

        # 2B: Engineered Risk Matrix
        interact_df['runway_bin'] = pd.cut(interact_df['inventory_runway'], bins=[-np.inf, 0.5, 1.5, 3, np.inf],
                                           labels=['Critical', 'Low', 'Healthy', 'Excess'])
        interact_df['transit_bin'] = pd.cut(interact_df['transit_ratio'], bins=[-np.inf, 0.5, 1.5, np.inf],
                                            labels=['Low', 'Med', 'High'])
        pivot = interact_df.groupby(['transit_bin', 'runway_bin'], observed=False)[target_col].mean().unstack() * 100

        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=axes[1, 1], cbar_kws={'label': 'Backorder Rate (%)'})
        axes[1, 1].set_title("2B. ENGINEERED INTERACTION: The Risk Matrix", fontweight='bold', color='teal')
        axes[1, 1].invert_yaxis()

        handles = [plt.Line2D([0], [0], color='lightgrey', lw=3), plt.Line2D([0], [0], color='red', lw=3)]
        fig.legend(handles, ['Safe (0)', 'Backorder (1)'], loc='lower right', bbox_to_anchor=(0.98, 0.59),
                   title="Target")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_adversarial_validation(X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: list,
                                   seed: int = 42) -> float:
        """Detects Covariate Shift between Train and Validation sets."""
        logger.info("Running Adversarial Validation (Covariate Shift Detection)...")
        y_adv_train = np.zeros(len(X_train))
        y_adv_val = np.ones(len(X_val))
        X_adv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_adv = np.concatenate([y_adv_train, y_adv_val])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(y_adv))

        for tr_idx, va_idx in skf.split(X_adv, y_adv):
            X_tr_f, y_tr_f = X_adv.iloc[tr_idx], y_adv[tr_idx]
            X_va_f, y_va_f = X_adv.iloc[va_idx], y_adv[va_idx]

            dtrain = lgb.Dataset(X_tr_f, label=y_tr_f, categorical_feature=categorical_cols, free_raw_data=False)
            dval = lgb.Dataset(X_va_f, label=y_va_f, reference=dtrain, free_raw_data=False)

            model = lgb.train({'objective': 'binary', 'verbosity': -1, 'random_state': seed, 'max_depth': 3}, dtrain,
                              num_boost_round=50, valid_sets=[dval], callbacks=[lgb.early_stopping(10, verbose=False)])
            oof_preds[va_idx] = model.predict(X_va_f, num_iteration=model.best_iteration)

        adv_auc = roc_auc_score(y_adv, oof_preds)
        logger.info(f"Adversarial AUC: {adv_auc:.4f}")
        if adv_auc > 0.7:
            logger.warning("Strong Covariate Shift detected!")
        return adv_auc

    # --- Static Visualizations ---
    @staticmethod
    def plot_robust_distribution(df: pd.DataFrame, col: str, target: Optional[str] = None,
                                 log_scale: bool = False, cnt_log_scale: bool = False, title_suffix: str = "") -> None:
        """
        Static Histogram with Log Scale toggle and Target split.

        ARCHITECTURAL JUSTIFICATION:
        Demonstrates long-tail distributions (extreme variance) in supply chain metrics.
        This directly justifies the choice of Gradient Boosted Trees (LightGBM) over
        Deep Learning / Linear models, as trees are robust to monotonic transformations
        and do not require strict standardization of extreme outliers.
        """
        plt.figure(figsize=(10, 5))
        plot_data = df[[col, target]].dropna() if target else df[[col]].dropna()
        plot_data = plot_data[~np.isinf(plot_data[col])]
        sns.histplot(data=plot_data, x=col, hue=target, bins=50, log_scale=log_scale,
                     palette={0: 'lightgrey', 1: 'red'} if target else None, alpha=0.6, element="step")
        scale_str = "(Log Scale)" if log_scale else "(Linear Scale)"
        plt.title(f"Robust Distribution: {col} {scale_str}\n{title_suffix}", fontsize=12, fontweight='bold')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        if cnt_log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_categorical_distribution(df: pd.DataFrame, col: str, top_k: int = 10,
                                      target: Optional[str] = None) -> None:
        """
        Dual-axis Bar Chart (Volume) and Line Chart (Risk Rate).

        ARCHITECTURAL JUSTIFICATION:
        Proves heuristic validity (e.g., does 'deck_risk' actually correlate with backorders?).
        If risk variance is high across categories, it validates the use of these boolean/categorical
        flags as primary splitting nodes in our tree architecture.
        """
        counts = df[col].value_counts().nlargest(top_k).reset_index()
        counts.columns = [col, 'Count']
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        sns.barplot(data=counts, x=col, y='Count', ax=ax1, color="teal", alpha=0.6)
        ax1.set_ylabel("Volume (Count)", color="teal", fontweight='bold')

        if target:
            rates = df.groupby(col)[target].mean().reset_index()
            merged = pd.merge(counts, rates, on=col)
            ax2 = ax1.twinx()
            ax2.plot(merged.index, merged[target], color="red", marker="o", linewidth=2, markersize=8)
            ax2.set_ylabel(f"{target} Rate", color="red", fontweight='bold')
            ax2.set_yticklabels(['{:,.2%}'.format(x) for x in ax2.get_yticks()])
        plt.title(f"Volume and Risk Distribution: {col}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_health(evals_dict: Dict):
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        epochs = range(len(evals_dict['validation']['Procurement_NDCG']))
        # Plot NDCG (Primary Axis)
        ax1.plot(epochs, evals_dict['validation']['Procurement_NDCG'], color='teal', linewidth=2, label='Val NDCG')
        ax1.set_xlabel("Boosting Rounds", fontweight='bold')
        ax1.set_ylabel("NDCG Score", color='teal', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='teal')
        # Plot Cost on secondary axis
        ax2 = ax1.twinx()
        cost_raw = np.array(evals_dict['validation']['Asymmetric_Cost'])
        cost_scaled = cost_raw / max(cost_raw)
        ax2.plot(epochs, cost_scaled, color='red', linestyle='--', linewidth=2, label='Val Cost (Normalized)')
        ax2.set_ylabel("Cost (Normalized)", color='red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Training Health: NDCG Maximization vs. Cost Minimization", fontsize=12, fontweight='bold')
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(model, top_k=15):
        imp_df = pd.DataFrame(
                {'Feature': model.feature_name(), 'Gain': model.feature_importance(importance_type='gain')})
        imp_df = imp_df.sort_values(by='Gain', ascending=False).head(top_k)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df, x='Gain', y='Feature', color='darkblue', alpha=0.8)
        plt.title(f"Top {top_k} ROI-Driving Signals (LightGBM Gain)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_target_imbalance(df: pd.DataFrame, target_col: str) -> None:
        """Visualizes the extreme target imbalance (Chapter 1 from POC)."""
        target_counts = df[target_col].value_counts(normalize=True) * 100
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette=['#2ecc71', '#e74c3c'])
        plt.title("Target Distribution: The Imbalance Physics", fontsize=12, fontweight='bold')
        plt.ylabel("Percentage of SKUs (%)")
        plt.xlabel("Target Variable")
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                        va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_safety_stock_distribution(danger_zone_df: pd.DataFrame, tau_hat: float) -> None:
        """Visualizes the recommended safety stock buffers from the Conformal Engine."""
        plt.figure(figsize=(12, 6))
        sns.histplot(data=danger_zone_df, x='Recommended_Safety_Stock', hue='Actual_Backorder', bins=50,
                multiple='layer', palette={0: 'lightgrey', 1: 'red'}, edgecolor='white', alpha=0.7)
        plt.title(
                f"Distribution of Recommended Safety Stock Units\n(Conformal Threshold: {tau_hat:.4f} for 95% "
                f"Confidence)",
                fontsize=14, fontweight='bold')
        plt.xlabel("Extra Units to Order (Safety Buffer)", fontweight='bold')
        plt.ylabel("Number of SKUs", fontweight='bold')
        plt.legend(title='Did it actually Backorder?', labels=['Yes (True Risk)', 'No (False Alarm)'])
        plt.tight_layout()
        plt.show()


# ==============================================================================
# LAYER 1: AUTOMATED DATA PHYSICS PROFILING (Ingestion Router)
# ==============================================================================
@dataclass
class MLRoutingDecision:
    demand_type: str  # Continuous / Intermittent
    is_demand_censored: bool  # Stockout masking
    mnar_features: List[str]  # Features that CANNOT be imputed
    recommended_lags: List[int]  # Autoregressive lags to build
    concept_drift_detected: bool  # Requires strict time-series splits
    recommended_model: str  # GBDT vs DeepLearning
    recommended_objective: str  # Standard vs FocalLoss vs Tweedie


class SparsityAnalyzer:
    @staticmethod
    def calculate_adi(df: pd.DataFrame, velocity_col: str) -> str:
        """Calculates Average Demand Interval (ADI) to route Continuous vs Intermittent models."""
        if velocity_col not in df.columns:
            return "Unknown"

        total_periods = len(df)
        periods_with_sales = (df[velocity_col] > 0).sum()
        adi = total_periods / periods_with_sales if periods_with_sales > 0 else float('inf')

        demand_type = "Intermittent" if adi > 1.32 else "Continuous"
        logger.info(f"ADI Score: {adi:.2f} -> Demand Type: {demand_type}")
        return demand_type

    @staticmethod
    def detect_mnar(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> List[str]:
        """Identifies columns where Missingness directly correlates with the Target (MNAR)."""
        mnar_cols = []
        for col in feature_cols:
            if col in df.columns and df[col].isna().sum() > 0:
                mask = df[col].isna()
                risk_missing = df.loc[mask, target_col].mean()
                risk_present = df.loc[~mask, target_col].mean()

                # If risk jumps by more than 20% when missing, it's a structural signal
                if risk_present > 0 and (risk_missing / risk_present > 1.2 or risk_missing / risk_present < 0.8):
                    mnar_cols.append(col)

        logger.info(f"MNAR Features detected (IMPUTATION BANNED): {mnar_cols}")
        return mnar_cols


class TemporalAnalyzer:
    @staticmethod
    def find_significant_lags(df: pd.DataFrame, target_col: str, max_lags: int = 14) -> List[int]:
        """Uses PACF to automatically find memory retention (which lags to engineer)."""
        # Drop NAs and ensure variance exists
        series = df[target_col].dropna()
        if series.nunique() <= 1 or len(series) < max_lags * 2:
            return []

        pacf_vals, confint = pacf(series, nlags=max_lags, alpha=0.05)
        significant_lags = []

        # Index 0 is lag 0 (always 1.0). Check lags 1 to max_lags.
        for lag in range(1, len(pacf_vals)):
            lower_bound = confint[lag][0] - pacf_vals[lag]
            upper_bound = confint[lag][1] - pacf_vals[lag]

            if pacf_vals[lag] > upper_bound or pacf_vals[lag] < lower_bound:
                significant_lags.append(lag)

        logger.info(f"PACF Significant Lags discovered: {significant_lags}")
        return significant_lags

    @staticmethod
    def detect_concept_drift(df: pd.DataFrame, time_col: str) -> bool:
        """Adversarial Validation: Can a model distinguish between Past and Future?"""
        if time_col not in df.columns:
            return False

        df_sorted = df.sort_values(time_col).copy()
        split_idx = int(len(df_sorted) * 0.7)

        # Label 0 for Past, 1 for Future
        df_sorted['is_future'] = 0
        df_sorted.iloc[split_idx:, df_sorted.columns.get_loc('is_future')] = 1

        # Take numeric columns for adversarial check
        num_cols = df_sorted.select_dtypes(include=[np.number]).columns.drop('is_future', errors='ignore')
        if len(num_cols) == 0:
            return False

        X = df_sorted[num_cols].fillna(-999)
        y = df_sorted['is_future']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)
        drift = auc > 0.70
        logger.info(f"Adversarial Validation AUC: {auc:.3f} -> Concept Drift Detected: {drift}")
        return drift


class MLArchitectureRouter:
    """The central brain that orchestrates the profilers and issues the ML Blueprint."""

    def __init__(self, target_col: str, time_col: Optional[str] = None, velocity_col: Optional[str] = None):
        self.target_col = target_col
        self.time_col = time_col
        self.velocity_col = velocity_col

    def profile_and_route(self, df_raw: pd.DataFrame, feature_cols: List[str]) -> MLRoutingDecision:
        logger.info("=== INITIALIZING ENTERPRISE INGESTION PROFILING ===")

        # Phase A Tests
        demand_type = "Continuous"
        if self.velocity_col and self.velocity_col in df_raw.columns:
            demand_type = SparsityAnalyzer.calculate_adi(df_raw, self.velocity_col)

        mnar_feats = SparsityAnalyzer.detect_mnar(df_raw, self.target_col, feature_cols)

        # Phase B Tests
        lags = []
        drift = False
        if self.time_col:
            lags = TemporalAnalyzer.find_significant_lags(df_raw, self.target_col)
            drift = TemporalAnalyzer.detect_concept_drift(df_raw, self.time_col)

        # Phase C: Imbalance Routing (Simplified Cost Ratio)
        prevalence = df_raw[self.target_col].mean()
        objective = "FocalLoss (Asymmetric)" if prevalence < 0.05 else "Standard LogLoss"
        if demand_type == "Intermittent":
            objective = "Tweedie Variance"  # Better for zero-inflated demand modeling

        # Structure Selection
        model = "GBDT (LightGBM)"  # Default for tabular
        if self.time_col and len(df_raw) > 500000 and len(lags) > 10:
            model = "DeepLearning (LSTM/TFT)"  # High density sequential data

        decision = MLRoutingDecision(demand_type=demand_type, is_demand_censored=False,
                                     # Placeholder for Kaplan-Meier unconstraining
                                     mnar_features=mnar_feats, recommended_lags=lags, concept_drift_detected=drift,
                                     recommended_model=model, recommended_objective=objective)

        logger.info("=== PROFILING COMPLETE. ROUTING DECISION LOCKED. ===")
        logger.info(f"Routing: {model} with {objective} | Drift: {drift} | MNAR Feats: {len(mnar_feats)}")
        return decision


# ==============================================================================
# LAYER 2: GOVERNANCE & CONTRACTS (The Semantic Schema Registry)
# ==============================================================================
class ColumnRole(Enum):
    IDENTITY = "identity"
    TARGET = "target"
    FEATURE = "feature"
    METADATA = "metadata"


@dataclass
class FeatureContract:
    """
    Defines the physical and semantic expectations of a feature.
    """
    name: str
    dependencies: List[str]
    dtype_family: str  # 'numeric', 'categorical', 'boolean'
    role: ColumnRole = ColumnRole.FEATURE
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""


class SemanticSchemaRegistry:
    """
    A globally accessible registry tracking both reactive schema checkpoints
    and proactive semantic feature contracts.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._checkpoints: Dict[str, Dict[str, str]] = {}
        self._contracts: Dict[str, FeatureContract] = {}
        self.logger = logging.getLogger("SemanticRegistry")

    # --- 1. PROACTIVE CONTRACT MANAGEMENT ---

    def register_contracts(self, contracts: List[FeatureContract]) -> None:
        """Registers the expected definitions and bounds of features."""
        for contract in contracts:
            self._contracts[contract.name] = contract
        self.logger.info(f"Registered {len(contracts)} explicit feature contracts.")

    def pre_flight_dependency_check(self, df: pd.DataFrame, target_features: List[str]) -> bool:
        """
        Validates that all upstream dependencies exist BEFORE attempting feature engineering.
        """
        missing_deps = set()
        for feat in target_features:
            if feat in self._contracts:
                for dep in self._contracts[feat].dependencies:
                    if dep not in df.columns:
                        missing_deps.add(dep)

        if missing_deps:
            msg = f"PRE-FLIGHT FAILED: Missing dependencies {missing_deps} required for target features."
            if self.strict_mode:
                raise KeyError(msg)
            else:
                self.logger.warning(msg)
                return False
        return True

    def validate_semantic_bounds(self, df: pd.DataFrame) -> None:
        """
        Validates that computed features strictly adhere to their defined physical bounds.
        """
        violations = []
        for feat_name, contract in self._contracts.items():
            if feat_name in df.columns and contract.dtype_family == 'numeric':
                series = df[feat_name].dropna()
                if series.empty:
                    continue

                if contract.min_val is not None and (series < contract.min_val).any():
                    violations.append(f"{feat_name} contains values below min bound ({contract.min_val})")
                if contract.max_val is not None and (series > contract.max_val).any():
                    violations.append(f"{feat_name} contains values above max bound ({contract.max_val})")

        if violations:
            msg = f"SEMANTIC BOUND VIOLATIONS DETECTED:\n" + "\n".join(violations)
            if self.strict_mode:
                raise ValueError(msg)
            else:
                self.logger.error(msg)
        else:
            self.logger.info("All features successfully passed semantic bound validation.")

    # --- 2. REACTIVE CHECKPOINTING & DRIFT DETECTION ---

    def capture_checkpoint(self, df: pd.DataFrame, checkpoint_name: str) -> None:
        schema = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        self._checkpoints[checkpoint_name] = schema
        self.logger.info(f"Captured reactive checkpoint '{checkpoint_name}' | Features: {len(schema)}")

    def validate_checkpoint(self, df: pd.DataFrame, checkpoint_name: str, auto_align: bool = False) -> Optional[
        pd.DataFrame]:
        """
        Validates incoming data against a historical checkpoint to prevent Schema/DType Drift.
        """
        if checkpoint_name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_name}' does not exist.")

        expected_schema = self._checkpoints[checkpoint_name]
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)

        # Missing Features
        missing = expected_cols - actual_cols
        if missing and self.strict_mode:
            raise KeyError(f"[{checkpoint_name}] CRITICAL: Missing {len(missing)} features: {list(missing)[:5]}")

        # DType Drift with Safe Casting Check
        drift_count = 0
        for col in expected_cols:
            if col in df.columns:
                exp_type = expected_schema[col]
                act_type = str(df[col].dtype)

                if not self._is_safe_cast(exp_type, act_type):
                    self.logger.error(f"[{checkpoint_name}] Drift on '{col}': Expected {exp_type}, got {act_type}")
                    drift_count += 1

        if drift_count > 0 and self.strict_mode:
            raise TypeError(f"[{checkpoint_name}] Schema validation failed due to {drift_count} DType drifts.")

        if auto_align:
            ordered_cols = [c for c in expected_schema.keys() if c in df.columns]
            return df[ordered_cols]
        return None

    def _is_safe_cast(self, expected: str, actual: str) -> bool:
        """Internal heuristic to determine if a type change is a safe pandas upcast."""
        if expected == actual:
            return True
        if 'int' in expected and 'int' in actual:
            return True
        if 'float' in expected and 'float' in actual:
            return True
        return False

    # --- 3. MLOPS INTEGRATION ---

    def export_inventory(self) -> Dict[str, pd.DataFrame]:
        """
        Exports the entire tracked graph (Contracts and Checkpoints) for MLflow/Governance.
        """
        ckpt_records = [{'Checkpoint': ckpt, 'Feature': col, 'Expected_DType': dtype} for ckpt, schema in
                        self._checkpoints.items() for col, dtype in schema.items()]

        contract_records = [{'Feature': name, 'Dependencies': ", ".join(c.dependencies), 'DType_Family': c.dtype_family,
                             'Min_Val': c.min_val, 'Max_Val': c.max_val} for name, c in self._contracts.items()]

        return {'checkpoints': pd.DataFrame(ckpt_records), 'contracts': pd.DataFrame(contract_records)}


# ==============================================================================
# LAYER 3: CROSS-SECTIONAL FEATURE ENGINEERING (Domain-Generic mapped to POC logic)
# ==============================================================================
class CrossSectionalDerivativeEngineer(BaseEstimator, TransformerMixin):
    """
    Translates longitudinal temporal concepts into cross-sectional features.
    Uses tuples of (column_name, time_window_divisor) to safely normalize
    different time scales into comparable velocity metrics.
    """

    def __init__(self, inventory_col: str, velocity_col: Tuple[str, float],  # e.g., ('sales_3_month', 3.0)
                 lead_time_col: Tuple[str, float],  # e.g., ('lead_time', 30.0) -> days to months
                 transit_col: Optional[str] = None, sales_mapping: Optional[Dict[str, Tuple[str, float]]] = None,
                 forecast_mapping: Optional[Dict[str, Tuple[str, float]]] = None,
                 performance_mapping: Optional[Dict[str, Tuple[str, float]]] = None, eps: float = 1e-5):

        self.inventory_col = inventory_col
        self.velocity_col_name, self.velocity_div = velocity_col
        self.lead_time_col_name, self.lead_time_div = lead_time_col
        self.transit_col = transit_col
        self.sales_mapping = sales_mapping
        self.forecast_mapping = forecast_mapping
        self.performance_mapping = performance_mapping
        self.eps = eps
        self.learned_lead_time_median_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.lead_time_col_name in X.columns:
            self.learned_lead_time_median_ = X[self.lead_time_col_name].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. MNAR Missing Indicator
        if self.lead_time_col_name in X_out.columns:
            X_out[f'is_{self.lead_time_col_name}_missing'] = X_out[self.lead_time_col_name].isna().astype(np.int8)

        # Base Velocity (Normalized)
        if self.velocity_col_name in X_out.columns:
            base_velocity = X_out[self.velocity_col_name] / self.velocity_div
        else:
            base_velocity = pd.Series(0, index=X_out.index)

        # 2. Runway Calculation
        if self.inventory_col in X_out.columns and self.velocity_col_name in X_out.columns:
            X_out['inventory_runway_periods'] = X_out[self.inventory_col] / (base_velocity + self.eps)
            X_out['critical_runway_flag'] = (X_out['inventory_runway_periods'] < 1.0).astype(np.int8)

        # 3. Transit Context
        if self.transit_col and self.transit_col in X_out.columns:
            lt = X_out[self.lead_time_col_name].fillna(
                self.learned_lead_time_median_ if self.learned_lead_time_median_ else 1.0)
            lt_normalized = lt / self.lead_time_div
            expected_demand = base_velocity * lt_normalized
            X_out['transit_coverage_ratio'] = X_out[self.transit_col] / (expected_demand + self.eps)

        # 4. Demand Acceleration & Spikes (Safely Normalized!)
        if self.sales_mapping and all(v[0] in X_out.columns for v in self.sales_mapping.values()):
            short_col, short_div = self.sales_mapping.get('short')
            long_col, long_div = self.sales_mapping.get('long')

            short_vel = X_out[short_col] / short_div
            long_vel = X_out[long_col] / long_div

            X_out['sales_acceleration'] = short_vel - long_vel
            X_out['sales_spike_ratio'] = short_vel / (long_vel + self.eps)

        # 5. Forecast Change Points
        if self.forecast_mapping and all(v[0] in X_out.columns for v in self.forecast_mapping.values()):
            short_col, short_div = self.forecast_mapping.get('short')
            long_col, long_div = self.forecast_mapping.get('long')
            X_out['forecast_acceleration'] = (X_out[short_col] / short_div) - (X_out[long_col] / long_div)

        # 6. Supplier Degradation Signal (No time division needed for percentages)
        if self.performance_mapping and all(v[0] in X_out.columns for v in self.performance_mapping.values()):
            short_col, _ = self.performance_mapping.get('short')
            long_col, _ = self.performance_mapping.get('long')
            X_out['supplier_degradation_score'] = X_out[long_col] - X_out[short_col]
            X_out['severe_degradation_flag'] = (X_out['supplier_degradation_score'] > 0.10).astype(np.int8)

        return X_out


class SKUDensityArchetyper(BaseEstimator, TransformerMixin):
    """Reduces highly variant behavioral heuristics into stable GMM clusters."""

    def __init__(self, numeric_features: List[str], n_components: int = 8, reg_covar: float = 1e-3,
                 random_state: int = 75):
        self.numeric_features = numeric_features
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.scaler_ = RobustScaler()
        self.gmm_ = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=reg_covar,
                                    random_state=random_state)
        self.medians_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_num = X[self.numeric_features].astype(np.float64)
        self.medians_ = X_num.median()
        X_scaled = self.scaler_.fit_transform(X_num.fillna(self.medians_))
        # Subsample for EM speed if large
        idx = np.random.choice(len(X_scaled), min(100000, len(X_scaled)), replace=False)
        self.gmm_.fit(X_scaled[idx])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_scaled = self.scaler_.transform(X_out[self.numeric_features].astype(np.float64).fillna(self.medians_))
        X_out['sku_archetype_id'] = pd.Categorical(self.gmm_.predict(X_scaled))
        return X_out


# ==============================================================================
# LAYER 4: LONGITUDINAL FEATURE ENGINEERING (Time-Series Canonical Components)
# ==============================================================================
class LongitudinalRobustEWM(BaseEstimator, TransformerMixin):
    """
    Translates Phoenix 'ewm_state' and 'robust_filler' into a unified Sklearn transformer.
    Extracts underlying operational baselines (e.g., Supplier Reliability, Demand Baseline)
    while being resilient to localized systemic noise (outliers).
    """

    def __init__(self, entity_col: str, time_col: str, target_cols: List[str], span: int = 7,
                 outlier_threshold_std: float = 3.0):
        self.entity_col, self.time_col, self.target_cols, self.span, self.outlier_threshold_std = (entity_col, time_col,
                                                                                                   target_cols, span,
                                                                                                   outlier_threshold_std)
        self.learned_state_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> 'LongitudinalRobustEWM':
        """Learns the final chronological state for inference stability."""
        X_sorted = X.sort_values(by=[self.entity_col, self.time_col])
        for col in self.target_cols:
            if col in X.columns:
                ewm_series = X_sorted.groupby(self.entity_col)[col].transform(
                        lambda x: x.ewm(span=self.span, adjust=False).mean())
                state_df = pd.DataFrame(
                        {self.entity_col: X_sorted[self.entity_col], 'val': ewm_series}).drop_duplicates(
                        subset=[self.entity_col], keep='last')
                self.learned_state_[col] = state_df.set_index(self.entity_col)['val'].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies EWM smoothing, handling missing data via historical state."""
        X_out = X.copy()
        if not X_out[self.time_col].is_monotonic_increasing: X_out = X_out.sort_values(
                by=[self.entity_col, self.time_col])
        for col in self.target_cols:
            if col not in X_out.columns: continue
            smoothed_col, residual_col = f"{col}_ewma_robust", f"{col}_ewma_residual"
            X_out[smoothed_col] = X_out.groupby(self.entity_col)[col].transform(
                    lambda x: x.ewm(span=self.span, adjust=False).mean())

            global_fallback = np.nanmedian(list(self.learned_state_.get(col, {}).values())) if self.learned_state_.get(
                    col) else 0.0
            if X_out[smoothed_col].isna().any():
                X_out[smoothed_col] = X_out.apply(
                    lambda row: self.learned_state_.get(col, {}).get(row[self.entity_col], global_fallback) if pd.isna(
                            row[smoothed_col]) else row[smoothed_col], axis=1)

            X_out[residual_col] = X_out[col] - X_out[smoothed_col]
            clip_val = self.outlier_threshold_std * X_out[residual_col].std()
            X_out[residual_col] = X_out[residual_col].clip(-clip_val, clip_val)
        return X_out.loc[X.index]


class LongitudinalShockDetector(BaseEstimator, TransformerMixin):
    """
    Translates Phoenix 'change_point_detector' and 'lags_pairwise_distance'.
    Identifies sudden regime changes in supply chain flow (e.g., Inventory runs, Demand shocks).
    """

    def __init__(self, entity_col: str, time_col: str, target_cols: List[str], window_size: int = 4,
                 z_threshold: float = 2.5):
        self.entity_col, self.time_col, self.target_cols, self.window_size, self.z_threshold = (entity_col, time_col,
                                                                                                target_cols,
                                                                                                window_size,
                                                                                                z_threshold)
        self.global_stds_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> 'LongitudinalShockDetector':
        for col in self.target_cols:
            if col in X.columns: self.global_stds_[col] = X[col].std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if not X_out[self.time_col].is_monotonic_increasing: X_out = X_out.sort_values(
                by=[self.entity_col, self.time_col])
        for col in self.target_cols:
            if col not in X_out.columns: continue
            shifted_group = X_out.groupby(self.entity_col)[col].shift(1)
            rolling_mean = shifted_group.groupby(X_out[self.entity_col]).rolling(window=self.window_size,
                                                                                 min_periods=1).mean().reset_index(
                    level=0, drop=True)
            rolling_std = shifted_group.groupby(X_out[self.entity_col]).rolling(window=self.window_size,
                                                                                min_periods=1).std().reset_index(
                    level=0, drop=True)
            dist_col = f"{col}_pairwise_distance"
            X_out[dist_col] = X_out[col] - rolling_mean
            safe_std = rolling_std.fillna(self.global_stds_.get(col, 1.0)).replace(0, self.global_stds_.get(col, 1.0))
            X_out[f"{col}_rolling_zscore"] = X_out[dist_col] / safe_std
            X_out[f"{col}_shock_flag"] = (X_out[f"{col}_rolling_zscore"].abs() > self.z_threshold).astype(np.int8)
        return X_out.loc[X.index]


class LongitudinalFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Translates Phoenix 'temporal_feature_selector'.
    Filters temporal features dynamically, ensuring no target leakage by only calculating
    correlations and variances on the training fold.
    """

    def __init__(self, target_col: str, max_features: int = 50, corr_threshold: float = 0.01):
        self.target_col, self.max_features, self.corr_threshold = target_col, max_features, corr_threshold
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> 'LongitudinalFeatureSelector':
        candidate_cols = [c for c in X.columns if c != self.target_col and pd.api.types.is_numeric_dtype(X[c])]
        correlations = {col: abs(corr) for col in candidate_cols if
                        pd.notna(corr := X[col].corr(X[self.target_col])) and abs(corr) >= self.corr_threshold}
        self.selected_features_ = [feat for feat, _ in sorted(correlations.items(), key=lambda i: i[1], reverse=True)][
            :self.max_features]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[c for c in X.columns if c in self.selected_features_ or c == self.target_col]]


class WalkForwardTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Translates Phoenix 'TargetGuidedEncoder' into a Leakage-Free Temporal Encoder.
    Designed specifically for Longitudinal Event Logs.
    Instead of K-Fold (which causes future leakage in time series), it uses an
    Expanding Window. For any row at time T, its encoded risk is calculated strictly
    using the Bayesian smoothed target mean from time 0 to T-1.
    """

    def __init__(self, cat_cols: List[str], target_col: str, time_col: str, smoothing_factor: float = 10.0):
        self.cat_cols, self.target_col, self.time_col, self.smoothing_factor = (cat_cols, target_col, time_col,
                                                                                smoothing_factor)
        self.global_encodings_: Dict[str, Dict[str, float]] = {}
        self.global_target_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y=None) -> 'WalkForwardTargetEncoder':
        """Calculates the final global encodings to be used ONLY for future, unseen inference data."""
        self.global_target_mean_ = X[self.target_col].mean()
        for col in self.cat_cols:
            stats = X.groupby(col)[self.target_col].agg(['mean', 'count'])
            self.global_encodings_[col] = (
                    (stats['mean'] * stats['count'] + self.global_target_mean_ * self.smoothing_factor) / (
                    stats['count'] + self.smoothing_factor)).to_dict()
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Executes strict Walk-Forward (Expanding Window) encoding for the training timeline."""
        X_out = X.copy()
        if not X_out[self.time_col].is_monotonic_increasing: X_out = X_out.sort_values(by=self.time_col)
        self.fit(X_out, y)
        global_expanding_mean = X_out[self.target_col].expanding().mean().shift(1).fillna(self.global_target_mean_)

        for col in self.cat_cols:
            cum_sum = X_out.groupby(col)[self.target_col].cumsum()
            cum_count = X_out.groupby(col).cumcount() + 1
            hist_sum = cum_sum.groupby(X_out[col]).shift(1).fillna(0)
            hist_count = cum_count.groupby(X_out[col]).shift(1).fillna(0)
            X_out[f"{col}_risk_prior"] = (hist_sum + global_expanding_mean * self.smoothing_factor) / (
                    hist_count + self.smoothing_factor)
        return X_out.loc[X.index]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fully matured global encodings to new test data."""
        X_out = X.copy()
        for col in self.cat_cols:
            new_col_name = f"{col}_risk_prior"
            encoding_dict = self.global_encodings_[col]
            X_out[new_col_name] = X_out[col].map(encoding_dict).fillna(self.global_target_mean_)
        return X_out


# ==============================================================================
# LAYER 5: MODELING OBJECTIVES & OPTIMIZATION EVALUATION
# ==============================================================================
class SupplyChainObjectives:
    """Mathematical formulations for business-aligned model optimization."""

    @staticmethod
    def asymmetric_focal_loss(preds: np.ndarray, dtrain: any, gamma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Custom objective strictly penalizing False Negatives in extreme imbalance."""
        labels = dtrain.get_label()
        weights = dtrain.get_weight() if dtrain.get_weight() is not None else np.ones_like(labels)
        # Clip probabilities to strictly prevent 0.0 or 1.0
        eps = 1e-7  # Numerical stability constant
        p = np.clip(expit(preds), eps, 1.0 - eps)

        term1 = (1 - p) ** gamma
        term2 = p ** gamma

        # Gradient computation
        grad = (
            p * term1 * np.log(p) * gamma * (labels == 1) -
            (1 - p) * term2 * np.log(1 - p) * gamma * (labels == 0) -
            term1 * (labels == 1) +
            term2 * (labels == 0)
        )

        # Hessian approximation
        hess = np.abs(p * (1 - p) * (gamma + 1))

        # SCALE BY BUSINESS IMPORTANCE
        return grad * weights, hess * weights


class SupplyChainEvaluationMetrics:
    """Generates the dual-evaluation metrics required by LightGBM custom loss."""

    def __init__(self, fn_cost: float = 50.0, fp_cost: float = 1.0, k_ndcg: int = 500):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.k_ndcg = k_ndcg

    def cost_ndcg_dual_eval(self, preds: np.ndarray, val_data: any) -> List[Tuple[str, float, bool]]:
        labels = val_data.get_label()
        probs = np.clip(expit(preds), 1e-7, 1.0 - 1e-7)

        # 1. Procurement NDCG
        try:
            ndcg_val = ndcg_score(labels.reshape(1, -1), probs.reshape(1, -1), k=self.k_ndcg)
        except:
            ndcg_val = 0.0

        # 2. Financial Cost
        preds_binary = (probs > 0.5).astype(int)
        fn = np.sum((labels == 1) & (preds_binary == 0))
        fp = np.sum((labels == 0) & (preds_binary == 1))
        total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)

        return [('Procurement_NDCG', ndcg_val, True), ('Asymmetric_Cost', total_cost, False)]


# ==============================================================================
# LAYER 6: MODEL TRAINING & EXPERIMENTATION (The MLOps Suite)
# ==============================================================================
class SupplyChainModelTrainer:
    """
    Abstracts the entire LightGBM training lifecycle: Surrogate Selection,
    Optuna Bayesian Optimization, and Dynamic Cross-Validation (K-Fold vs Time-Series).
    """

    def __init__(self, objective_func, eval_metric_func, is_time_series: bool = False, n_splits: int = 5,
                 random_state: int = 42):
        self.objective_func = objective_func
        self.eval_metric_func = eval_metric_func
        self.is_time_series = is_time_series
        self.n_splits = n_splits
        self.random_state = random_state

        self.models_: List[lgb.Booster] = []
        self.oof_preds_: Optional[np.ndarray] = None
        self.feature_importance_df_: Optional[pd.DataFrame] = None
        self.best_params_: Optional[dict] = None
        self.eval_results_dict_: Optional[dict] = None

    def _get_cv_splitter(self):
        """Dynamically routes CV strategy based on Profiler's concept drift detection."""
        if self.is_time_series:
            logger.info("CV Routing: Using TimeSeriesSplit (Preventing Future Leakage)")
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            logger.info("CV Routing: Using StratifiedKFold (Cross-Sectional Snapshot)")
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    @staticmethod
    def compute_dynamic_weights(velocity_series: pd.Series) -> pd.Series:
        """Data-Driven Weights (Business ROI Proxy)."""
        w_array = 1.0 + np.log1p(velocity_series.fillna(0).clip(lower=0))
        return pd.Series(w_array.values, index=velocity_series.index, dtype=np.float32)

    def run_surrogate_pruning(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series, categorical_cols: List[str],
                              gain_threshold: float = 1.0) -> List[str]:
        """Trains a quick surrogate model to prune dead/uninformative features."""
        logger.info("[*] Running Surrogate Model for Feature Vitality Assessment...")
        X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
            X, y, weights, test_size=0.2,
            stratify=y if not self.is_time_series else None, random_state=self.random_state)
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=categorical_cols, free_raw_data=False)
        surrogate_params = {'objective': self.objective_func, 'verbosity': -1, 'random_state': self.random_state}

        surrogate_model = lgb.train(surrogate_params, dtrain, num_boost_round=150)

        imp_df = pd.DataFrame(
                {'Feature': X.columns, 'Gain': surrogate_model.feature_importance(importance_type='gain')}).sort_values(
            by='Gain', ascending=False)

        vital_features = imp_df[imp_df['Gain'] > gain_threshold]['Feature'].tolist()
        logger.info(
                f"[*] Pruned {len(X.columns) - len(vital_features)} dead features. Retained {len(vital_features)} "
                f"vital features.")
        return vital_features

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series, categorical_cols: List[str],
                                 n_trials: int = 20) -> dict:
        """Runs Bayesian Optimization using Optuna and dynamic CV splitting."""
        logger.info("[*] Launching Optuna Bayesian Optimization...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            param_grid = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 300),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'verbosity': -1, 'random_state': self.random_state, 'objective': self.objective_func,
            }
            cv = self._get_cv_splitter()
            cv_ndcg = []

            for train_idx, val_idx in cv.split(X, y):
                X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], weights.iloc[train_idx]
                X_va, y_va, w_va = X.iloc[val_idx], y.iloc[val_idx], weights.iloc[val_idx]

                dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=categorical_cols,
                                     free_raw_data=False)
                dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain, free_raw_data=False)

                model = lgb.train(
                    params=param_grid, train_set=dtrain, num_boost_round=250, valid_sets=[dval],
                    feval=self.eval_metric_func,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=25, verbose=False),
                        LightGBMPruningCallback(trial, 'Procurement_NDCG'),
                    ],
                )
                cv_ndcg.append(model.best_score['valid_0']['Procurement_NDCG'])

            return np.mean(cv_ndcg)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))

        with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
            def tqdm_callback(study, trial):
                """"""
                # Update the progress bar
                pbar.update(1)
            study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
            pbar.close()

        logger.info(f"[*] Optimization Complete. Best OOF NDCG: {study.best_value:.4f}")
        logger.info(f"[*] Best Params:\n{study.best_params}")
        self.best_params_ = copy.deepcopy(study.best_params)
        return self.best_params_

    def fit_cv(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series, categorical_cols: List[str],
               params: dict = None, num_boost_round: int = 1000):
        """Trains the final models using CV, collects Out-Of-Fold (OOF) predictions and feature importances."""
        logger.info("[*] Executing Final Production CV Training...")
        if params is None: params = copy.deepcopy(self.best_params_)

        # Enforce deterministic behavior for final training
        params.update({
            'verbosity': -1, 'random_state': self.random_state, 'objective': self.objective_func,
            'deterministic': True, 'force_col_wise': True,
            },
        )
        cv = self._get_cv_splitter()
        self.oof_preds_ = np.full(len(y), np.nan)
        self.eval_results_dict_ = {}
        self.models_ = []
        importances = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"--- Training Fold {fold + 1}/{self.n_splits} ---")
            X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], weights.iloc[train_idx]
            X_va, y_va, w_va = X.iloc[val_idx], y.iloc[val_idx], weights.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=categorical_cols,
                                 free_raw_data=False)
            dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain, free_raw_data=False)

            # Custom Real-Time Logger
            def real_time_logger(period=20):
                """In-Training Matrics Callback"""
                best_val_ndcg = 0.0

                def _callback(env):
                    nonlocal best_val_ndcg
                    metrics = env.evaluation_result_list
                    try:
                        val_ndcg = [m[2] for m in metrics if m[0] == 'validation' and m[1] == 'Procurement_NDCG'][0]
                        val_cost = [m[2] for m in metrics if m[0] == 'validation' and m[1] == 'Asymmetric_Cost'][0]
                        train_ndcg = [m[2] for m in metrics if m[0] == 'train' and m[1] == 'Procurement_NDCG'][0]
                        train_cost = [m[2] for m in metrics if m[0] == 'train' and m[1] == 'Asymmetric_Cost'][0]
                    except IndexError:
                        # Fallback if evaluation metrics are missing
                        return

                    delta = val_ndcg - best_val_ndcg
                    if val_ndcg > best_val_ndcg:
                        best_val_ndcg = val_ndcg
                    sign = "+" if delta >= 0 else ""

                    if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
                        print(f"[*] Tree {env.iteration:3d} | "
                              f"Train NDCG: {train_ndcg:.4f} | "
                              f"Train Cost: {train_cost:.4f} | "
                              f"Val NDCG: {val_ndcg:.4f} (Δ {sign}{delta:.4f}) | "
                              f"Val Cost: {val_cost:.0f}", flush=True)

                return _callback

            # FIX: Included train_set in valid_sets and specified valid_names for the logger
            model = lgb.train(
                params=params, train_set=dtrain, num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval], valid_names=['train', 'validation'], feval=self.eval_metric_func,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=75, verbose=False),
                    lgb.record_evaluation(self.eval_results_dict_),
                    real_time_logger(period=50),
                ],
            )
            self.models_.append(model)
            # Store raw margins (logits) in OOF array
            self.oof_preds_[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
            importances.append(model.feature_importance(importance_type='gain'))

        # Average feature importance across folds
        self.feature_importance_df_ = pd.DataFrame(
                {'Feature': X.columns, 'Gain': np.mean(importances, axis=0)},
        ).sort_values(by='Gain', ascending=False).reset_index(drop=True)

        logger.info("[*] CV Training Complete. Models and OOF predictions generated.")
        return self.oof_preds_, self.models_, self.feature_importance_df_, self.eval_results_dict_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions: Ensemble for cross-sectional, Latest-Model for Time-Series."""
        if not self.models_:
            raise ValueError("Models are not trained yet. Call fit_cv first.")

        if self.is_time_series:
            logger.info("[*] Time-Series Mode: Inferencing using ONLY the most recent temporal model...")
            latest_model = self.models_[-1]
            return latest_model.predict(X, num_iteration=latest_model.best_iteration)

        else:
            logger.info(f"[*] Cross-Sectional Mode: Inferencing via Ensemble of {len(self.models_)} models...")
            preds = np.zeros(len(X))
            for model in self.models_:
                preds += model.predict(X, num_iteration=model.best_iteration)
            return preds / len(self.models_)


# ==============================================================================
# LAYER 7: OPERATIONS & MONETIZATION (Domain-Generic Conformal Engine)
# ==============================================================================
class ConformalInventoryEngine:
    """Translates model probabilities into physical inventory actions."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.tau_hat = None

    def calibrate(self, probs_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Finds the critical threshold tau that guarantees (1-alpha) coverage
        for the positive class (True Backorders).
        """
        # Filter NaNs (relevant if TimeSeriesSplit left early OOFs as NaN)
        valid_idx = ~np.isnan(probs_val)
        probs_clean, y_clean = probs_val[valid_idx], y_val[valid_idx]
        # 1. Isolate the probabilities of actual backorders & Sort probabilities in ascending order
        backorder_probs = np.sort(probs_clean[y_clean == 1])
        # 2. Find the (alpha)-quantile.
        # For example, if alpha=0.05, we find the probability threshold where only 5%
        # of actual backorders scored lower.
        n = len(backorder_probs)
        index = int(np.ceil(self.alpha * (n + 1))) - 1
        index = max(0, min(index, n - 1))  # Bounds check
        self.tau_hat = backorder_probs[index]
        logger.info(f"Conformal Calibration [Alpha={self.alpha}]: Threshold = {self.tau_hat:.4f}")
        return self.tau_hat

    def allocate_buffers(self, df_features: pd.DataFrame, probs: np.ndarray, velocity_col: str, lead_time_col: str,
                         lead_time_scale: float = 0.25) -> pd.DataFrame:
        """
        Calculates specific inventory unit recommendations based on conformal risk.
        """
        if self.tau_hat is None: raise ValueError("Engine must be calibrated.")
        allocations = pd.DataFrame(index=df_features.index)
        allocations['Risk_Probability'] = probs
        # 1. Flag items that breached the conformal threshold
        allocations['Conformal_Flag'] = (probs >= self.tau_hat).astype(int)
        # 2. Calculate Risk Multiplier
        # How far above the threshold is this item? (Caps at 3x to prevent infinite buffers)
        allocations['Risk_Multiplier'] = np.where(allocations['Conformal_Flag'] == 1,
                                                  np.clip(probs / self.tau_hat, 1.0, 3.0), 0.0)
        # 3. Safe extraction of business metrics (fallback to 1.0 if missing) & Clean up negative or zero values
        lead_time = df_features.get(lead_time_col, pd.Series(1.0, index=df_features.index)).clip(lower=1.0)
        velocity = df_features.get(velocity_col, pd.Series(0.1, index=df_features.index)).clip(lower=0.1)

        # Buffer = Velocity (units/month) * (Lead Time scaled, e.g. weeks to months via 0.25) * Risk Multiplier
        # This converts abstract risk into exact "spare units" needed to cover the lead time.
        allocations['Recommended_Safety_Stock'] = np.ceil(
                velocity * (lead_time * lead_time_scale) * allocations['Risk_Multiplier'])
        return allocations
