#!/usr/bin/env python3
"""
cluster.py — product clustering for substitution planning under tariff risk

This script loads a CSV of scraped product records (e.g., Target RedSky-like
schema), engineers robust features (numeric, categorical, and text), searches
for a good number of clusters (K), clusters products using (MiniBatch) KMeans,
and writes an enriched CSV with a cluster_id and only the columns useful for
downstream decisioning (e.g., multi‑armed bandit for substitutions).

Key design choices:
- Scales to large datasets with sparse matrices and MiniBatchKMeans.
- Extracts and parses embedded JSON-in-strings safely.
- Text features default to TF‑IDF + SVD (no network required). Optional
  OpenAI embeddings or Sentence‑Transformers can be turned on if available.
- Determines K via a combo of Silhouette / Calinski‑Harabasz and elbow on
  inertia; you can override with --k.
- Explicitly models tariff risk using the "handling import designation"
  field (e.g., "Made in the USA or Imported") and related signals.

Usage (defaults expect products.csv in working dir):

  python cluster.py \
    --input products.csv \
    --output products_enriched.csv \
    --metrics-out cluster_metrics.csv \
    --auto-k 2 24 \
    --text-mode tfidf \
    --sample 0   # 0 = use all rows; else integer sample size

If you already know K, supply --k 12 and omit --auto-k.

Dependencies:
  pip install pandas numpy scikit-learn scipy
  (optional) pip install sentence-transformers openai tiktoken

Author: you + GPT‑5 Thinking
"""

from __future__ import annotations
import argparse
import ast
import html
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -----------------------------
# Argument parsing
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KMeans clustering of products for substitution planning")
    p.add_argument("--input", default="products.csv", help="Path to input CSV")
    p.add_argument("--output", default="products_enriched.csv", help="Path to enriched CSV with cluster_id")
    p.add_argument("--metrics-out", default="cluster_metrics.csv", help="Where to write per-K metrics when --auto-k is used")
    p.add_argument("--k", type=int, default=None, help="Number of clusters to use (overrides --auto-k)")
    p.add_argument("--auto-k", nargs=2, type=int, metavar=("K_MIN","K_MAX"), default=(2, 20), help="Range of K to evaluate when --k is not set")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--sample", type=int, default=0, help="Random subsample size for speed; 0 = all")

    # Feature controls
    p.add_argument("--text-mode", choices=["tfidf","openai","sbert"], default="tfidf", help="How to embed text fields")
    p.add_argument("--text-max-features", type=int, default=12000, help="Max vocab when using TF-IDF")
    p.add_argument("--text-svd-dim", type=int, default=256, help="Dimensionality reduction for text features; 0 = disable")
    p.add_argument("--min-df", type=float, default=3, help="Min doc freq for TF-IDF (int for count >=, float for proportion)")

    p.add_argument("--weights", nargs=3, type=float, metavar=("W_NUM","W_CAT","W_TXT"), default=(1.0, 1.0, 1.0),
                   help="Relative weights for numeric, categorical, and text feature blocks")

    p.add_argument("--save-elbow", default=None, help="Optional .png path to save inertia elbow plot")

    # Column overrides (pipe-separated lists). Defaults are suitable for the provided schema.
    p.add_argument("--numeric-cols", default=None,
                   help="Pipe-separated numeric columns to include; defaults chosen for RedSky-like schema")
    p.add_argument("--categorical-cols", default=None,
                   help="Pipe-separated categorical columns to include; defaults chosen for RedSky-like schema")
    p.add_argument("--text-cols", default=None,
                   help="Pipe-separated text columns to include; defaults chosen for RedSky-like schema")

    # Performance / memory
    p.add_argument("--batch-kmeans", action="store_true", help="Use MiniBatchKMeans instead of full KMeans")
    p.add_argument("--batch-size", type=int, default=2048, help="MiniBatchKMeans batch size")

    # 2D projection for plotting
    p.add_argument("--embed-2d", dest="embed_2d", action="store_true", default=True,
                   help="Append 2D projection columns (UMAP1, UMAP2) to the output CSV (default: on)")
    p.add_argument("--no-embed-2d", dest="embed_2d", action="store_false",
                   help="Disable 2D projection columns")
    p.add_argument("--proj-method", choices=["umap", "svd"], default="umap",
                   help="Method for 2D projection (UMAP if available, else fallback to SVD)")
    p.add_argument("--umap-n-neighbors", type=int, default=20,
                   help="UMAP: number of neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP: min_dist")
    p.add_argument("--umap-metric", type=str, default="euclidean",
                   help="UMAP: metric (e.g., euclidean, cosine)")

    return p.parse_args()

# -----------------------------
# Utilities
# -----------------------------

JSON_LIKE_RE = re.compile(r"^\s*[\[{]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
AMP_BULLET_RE = re.compile(r"&bull;|\u2022|•")
WHITESPACE_RE = re.compile(r"\s+")


def safe_json_loads(x: Any) -> Any:
    """Attempt to parse JSON or Python-literal-ish strings; return original on failure."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return None
    # Unescape backslashes common in CSV-escaped JSON
    
    # If it's obviously JSON-like, try json -> ast literal as fallback
    if JSON_LIKE_RE.match(s):
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    # Not JSON-looking; return as-is
    return x


def strip_html(s: str) -> str:
    s = html.unescape(str(s))
    s = HTML_TAG_RE.sub(" ", s)
    s = AMP_BULLET_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()


def coalesce(*vals, default=None):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if v == "":
            continue
        return v
    return default


@dataclass
class FeatureBlocks:
    X_num: Optional[sparse.csr_matrix]
    X_cat: Optional[sparse.csr_matrix]
    X_txt: Optional[sparse.csr_matrix]

    def hstack(self, w_num: float, w_cat: float, w_txt: float) -> sparse.csr_matrix:
        blocks = []
        if self.X_num is not None and self.X_num.shape[1] > 0:
            blocks.append(self.X_num.multiply(w_num))
        if self.X_cat is not None and self.X_cat.shape[1] > 0:
            blocks.append(self.X_cat.multiply(w_cat))
        if self.X_txt is not None and self.X_txt.shape[1] > 0:
            blocks.append(self.X_txt.multiply(w_txt))
        if not blocks:
            raise ValueError("No features created — check column lists or input data")
        return sparse.hstack(blocks, format="csr")

# --- diagnostics for NaN/inf in feature matrices ---
import logging
from typing import Optional, Sequence, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def get_logger(name: str = "cluster_diag", log_file: Optional[str] = "cluster_debug.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _summarize_counts(idx: np.ndarray, names: List[str], top_n: int = 25) -> List[Tuple[str, int]]:
    if idx.size == 0:
        return []
    counts = np.bincount(idx, minlength=len(names))
    order = np.argsort(-counts)
    return [(names[i], int(counts[i])) for i in order if counts[i] > 0][:top_n]


def report_nan_in_dataframe(
    df, logger: Optional[logging.Logger] = None, label: str = "features_df", show_examples: int = 5
):
    if pd is None:
        raise RuntimeError("pandas not available for report_nan_in_dataframe")
    logger = logger or get_logger()

    nan_cols = df.columns[df.isna().any()].tolist()
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        # booleans per column indicating if that column has any ±Inf
        col_has_inf = np.isinf(numeric_df.to_numpy()).any(axis=0)
        inf_cols = numeric_df.columns[col_has_inf].tolist()
    else:
        inf_cols = []

    if not nan_cols and not inf_cols:
        logger.info(f"[{label}] No NaN or ±Inf values found in DataFrame with shape {df.shape}.")
        return

    if nan_cols:
        logger.error(f"[{label}] Columns with NaN: {len(nan_cols)}")
        for c in nan_cols:
            n = int(df[c].isna().sum()); pct = (n / len(df)) * 100 if len(df) else 0.0
            logger.error(f"  - {c}: {n} NaN ({pct:.2f}%)")
            if show_examples:
                rows = df.index[df[c].isna()].tolist()[:show_examples]
                logger.error(f"      e.g., rows: {rows}")

    if inf_cols:
        logger.error(f"[{label}] Columns with ±Inf: {len(inf_cols)}")
        for c in inf_cols:
            m = df[c].to_numpy()
            n = int(np.isinf(m).sum()); pct = (n / len(df)) * 100 if len(df) else 0.0
            logger.error(f"  - {c}: {n} ±Inf ({pct:.2f}%)")
            if show_examples:
                rows = df.index[np.isinf(m)].tolist()[:show_examples]
                logger.error(f"      e.g., rows: {rows}")


def report_nan_in_sparse(
    X,
    feature_names: Optional[Sequence[str]] = None,
    row_index: Optional[Sequence] = None,
    logger: Optional[logging.Logger] = None,
    label: str = "X_sparse",
    show_examples: int = 5,
    top_n_features: int = 25,
):
    if sp is None:
        raise RuntimeError("scipy.sparse not available for report_nan_in_sparse")
    logger = logger or get_logger()

    if not sp.issparse(X):
        logger.warning(f"[{label}] Input is not sparse; falling back to dense check.")
        _report_nan_in_dense(np.asarray(X), feature_names, row_index, logger, label, show_examples, top_n_features)
        return

    X = X.tocsr(copy=False)
    n_rows, n_cols = X.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_cols)]
    if len(names) != n_cols:
        logger.warning(f"[{label}] feature_names length {len(names)} != n_cols {n_cols}; using generic names.")
        names = [f"f{i}" for i in range(n_cols)]

    data = X.data
    if data.size == 0:
        logger.info(f"[{label}] Sparse matrix has 0 nnz; nothing to check.")
        return

    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)

    nnz_per_row = np.diff(X.indptr)
    row_ids_per_nnz = np.repeat(np.arange(n_rows), nnz_per_row)

    def _log(mask: np.ndarray, kind: str):
        if not mask.any():
            logger.info(f"[{label}] No {kind} values found in sparse data.")
            return
        bad_col_idx = X.indices[mask]
        tops = _summarize_counts(bad_col_idx, names, top_n=top_n_features)
        logger.error(f"[{label}] Found {mask.sum()} {kind} entries affecting {len(np.unique(bad_col_idx))} feature(s). Top offenders:")
        for feat, cnt in tops:
            logger.error(f"  - {feat}: {cnt} {kind} entries")

        if show_examples:
            bad_rows = row_ids_per_nnz[mask]
            mapped = [row_index[int(r)] for r in bad_rows[:show_examples]] if row_index is not None else bad_rows[:show_examples].tolist()
            logger.error(f"[{label}] Example rows with {kind}: {mapped}")

    _log(nan_mask, "NaN")
    _log(inf_mask, "±Inf")


def _report_nan_in_dense(
    A: np.ndarray,
    feature_names: Optional[Sequence[str]],
    row_index: Optional[Sequence],
    logger: logging.Logger,
    label: str,
    show_examples: int,
    top_n_features: int,
):
    n_rows, n_cols = A.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_cols)]
    if len(names) != n_cols:
        names = [f"f{i}" for i in range(n_cols)]

    nan_cols = np.where(np.isnan(A).any(axis=0))[0]
    inf_cols = np.where(np.isinf(A).any(axis=0))[0]

    def _summ(cols, kind):
        if cols.size == 0:
            logger.info(f"[{label}] No {kind} values found in dense matrix.")
            return
        counts = [(names[i], int(np.isnan(A[:, i]).sum() if kind == "NaN" else np.isinf(A[:, i]).sum())) for i in cols]
        counts.sort(key=lambda x: x[1], reverse=True)
        logger.error(f"[{label}] Columns with {kind}: {len(cols)} (top {top_n_features})")
        for name, cnt in counts[:top_n_features]:
            logger.error(f"  - {name}: {cnt} {kind}")
        if show_examples:
            i0 = int(cols[0])
            mask = np.isnan(A[:, i0]) if kind == "NaN" else np.isinf(A[:, i0])
            rows = np.where(mask)[0][:show_examples]
            rows = [row_index[int(r)] for r in rows] if row_index is not None else rows.tolist()
            logger.error(f"[{label}] Example rows with {kind} in '{names[i0]}': {rows}")

 
# -----------------------------
# Column defaults tailored to shared example
# -----------------------------

DEFAULT_NUMERIC = [
    # Price
    "data-product-price-current_retail_min",
    "data-product-price-reg_retail_max",
    # Ratings
    "data-product-ratings_and_reviews-statistics-rating-average",
    "data-product-ratings_and_reviews-statistics-rating-count",
    "data-product-ratings_and_reviews-statistics-review_count",
    "data-product-ratings_and_reviews-statistics-recommended_percentage",
    # Package dims / weight
    "data-product-item-package_dimensions-weight",
    "data-product-item-package_dimensions-depth",
    "data-product-item-package_dimensions-height",
    "data-product-item-package_dimensions-width",
]

DEFAULT_CATEGORICAL = [
    "data-product-category-name",
    "data-product-item-primary_brand-name",
    "data-product-item-product_classification-item_type-name",
    "data-product-item-product_classification-purchase_behavior",
    "data-product-item-merchandise_classification-department_name",
    # Tariff risk signal (handling import designation)
    "data-product-item-handling-import_designation_description",
    # Relationship / return method sometimes indicate pack vs variant
    "data-product-item-relationship_type_code",
    "data-product-item-return_method",
]

DEFAULT_TEXT = [
    # Titles / bullets / descriptions
    "data-product-item-product_description-title",
    "data-product-item-product_description-bullet_descriptions",
    "data-product-item-product_description-soft_bullet_description",
    # Long form description if present
    "data-product-item-product_description-downstream_description",
    # Category breadcrumbs string (names will be extracted)
    "data-product-category-breadcrumbs",
]

# -----------------------------
# Data Loading & Cleaning
# -----------------------------

def load_products_csv(path: str, sample_n: int = 0, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if sample_n and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return df


def parse_embedded_json_columns(df: pd.DataFrame, json_candidate_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Parse obviously JSON-like text columns in-place, and normalize a few known ones."""
    if json_candidate_cols is None:
        json_candidate_cols = [c for c in df.columns if df[c].astype(str).str.strip().str.startswith(("[","{")).any()]
    for c in json_candidate_cols:
        try:
            df[c + "__parsed"] = df[c].apply(safe_json_loads)
        except Exception:
            # Best effort; keep going
            pass

    # Normalize breadcrumbs (extract names only, joined by ">")
    bc_col = "data-product-category-breadcrumbs__parsed"
    if bc_col in df:
        def _bc_names(o):
            if isinstance(o, list):
                return " > ".join([str(x.get("name")) for x in o if isinstance(x, dict) and x.get("name")])
            return None
        df["category_path"] = df[bc_col].apply(_bc_names)

    # Normalize ratings secondary averages JSON (quality/value)
    sec_col = "data-product-ratings_and_reviews-statistics-secondary_averages__parsed"
    if sec_col in df:
        def _extract_sec_avg(o, key):
            if isinstance(o, list):
                for d in o:
                    if d.get("id") == key:
                        return d.get("value")
            return None
        df["rating_quality"] = df[sec_col].apply(lambda o: _extract_sec_avg(o, "quality"))
        df["rating_value"] = df[sec_col].apply(lambda o: _extract_sec_avg(o, "value"))

    # Children normalization: compute variant-level price range if available
    kids_col = "data-product-children__parsed"
    if kids_col in df:
        def _price_stats(lst):
            prs = []
            brands = []
            handling = []
            pkg_qty = []
            if isinstance(lst, list):
                for it in lst:
                    try:
                        p = it.get("price", {}).get("current_retail")
                        if p is None:
                            p = it.get("price", {}).get("reg_retail")
                        if p is not None:
                            prs.append(float(p))
                    except Exception:
                        pass
                    try:
                        br = it.get("item", {}).get("primary_brand", {}).get("name")
                        if br:
                            brands.append(br)
                    except Exception:
                        pass
                    try:
                        h = it.get("item", {}).get("handling", {}).get("import_designation_description")
                        if h:
                            handling.append(h)
                    except Exception:
                        pass
                    try:
                        # Pull package quantity if obvious
                        bullets = it.get("item", {}).get("product_description", {}).get("bullet_descriptions")
                        if isinstance(bullets, list):
                            for b in bullets:
                                m = re.search(r"Package Quantity:\s*(\d+)", str(b), flags=re.I)
                                if m:
                                    pkg_qty.append(int(m.group(1)))
                                    break
                    except Exception:
                        pass
            return {
                "children_price_min": np.min(prs) if prs else None,
                "children_price_max": np.max(prs) if prs else None,
                "children_brand_mode": pd.Series(brands).mode().iloc[0] if brands else None,
                "children_handling_mode": pd.Series(handling).mode().iloc[0] if handling else None,
                "children_pkg_qty_mode": pd.Series(pkg_qty).mode().iloc[0] if pkg_qty else None,
            }
        stats = df[kids_col].apply(_price_stats).apply(pd.Series)
        df = pd.concat([df, stats], axis=1)
    return df


def derive_tariff_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create tariff_risk flag and simple heuristics based on handling/import designation & origin text when available."""
    candidates = [
        "data-product-item-handling-import_designation_description",
        "handling-import_designation_description",
        "children_handling_mode",
    ]
    def _risk_from_text(s: Any) -> Optional[int]:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        t = str(s).lower()
        # Heuristic: Imported or Assem USA w/ foreign/dom. parts => higher tariff exposure
        if "import" in t or "foreign" in t or "assembled" in t:
            return 1
        if "made in the usa" in t or "usa" in t and "import" not in t:
            return 0
        return None
    risk = None
    for c in candidates:
        if c in df.columns:
            if risk is None:
                risk = df[c].apply(_risk_from_text)
            else:
                r2 = df[c].apply(_risk_from_text)
                risk = risk.fillna(r2)
    df["tariff_risk"] = risk.fillna(0).astype(int)  # default to 0 if unknown
    return df

def _to_num(s: Optional[pd.Series], index: pd.Index) -> pd.Series:
    """Coerce to float Series; if missing, return NaN series with same index."""
    if s is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(s, errors="coerce")

def prepare_numeric_derivations(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Coalesce price features to reduce missingness
    - Build package volume (L*W*H)
    - Add explicit missingness flags for prices, volume, and weight
    """
    # --- price coalescing ---
    # current_retail_min := current_retail_min or current_retail or children_price_min
    if "data-product-price-current_retail_min" in df:
        df["data-product-price-current_retail_min"] = (
            _to_num(df.get("data-product-price-current_retail_min"), df.index)
            .fillna(_to_num(df.get("data-product-price-current_retail"), df.index))
            .fillna(_to_num(df.get("children_price_min"), df.index))
        )
        df["data-product-price-current_retail_min__missing"] = df["data-product-price-current_retail_min"].isna().astype(int)

    # reg_retail_max := reg_retail_max or reg_retail or children_price_max
    if "data-product-price-reg_retail_max" in df:
        df["data-product-price-reg_retail_max"] = (
            _to_num(df.get("data-product-price-reg_retail_max"), df.index)
            .fillna(_to_num(df.get("data-product-price-reg_retail"), df.index))
            .fillna(_to_num(df.get("children_price_max"), df.index))
        )
        df["data-product-price-reg_retail_max__missing"] = df["data-product-price-reg_retail_max"].isna().astype(int)

    # --- package volume & weight flags ---
    depth  = _to_num(df.get("data-product-item-package_dimensions-depth"),  df.index)
    height = _to_num(df.get("data-product-item-package_dimensions-height"), df.index)
    width  = _to_num(df.get("data-product-item-package_dimensions-width"),  df.index)
    df["pkg_volume_lwh"] = depth * height * width
    df["pkg_volume_lwh__missing"] = df["pkg_volume_lwh"].isna().astype(int)

    if "data-product-item-package_dimensions-weight" in df:
        df["data-product-item-package_dimensions-weight"] = _to_num(
            df.get("data-product-item-package_dimensions-weight"), df.index
        )
        df["data-product-item-package_dimensions-weight__missing"] = df["data-product-item-package_dimensions-weight"].isna().astype(int)

    return df


def tune_numeric_cols(numeric_cols: Sequence[str], df: pd.DataFrame) -> List[str]:
    """
    Remove raw L/W/H from numeric features, add derived volume and missingness flags.
    Keep weight (with its missingness flag).
    Also include price missingness flags so 'unknown price' becomes a feature.
    """
    num = list(numeric_cols)

    # Drop the raw dims
    dims = [
        "data-product-item-package_dimensions-depth",
        "data-product-item-package_dimensions-height",
        "data-product-item-package_dimensions-width",
    ]
    num = [c for c in num if c not in dims]

    # Add derived volume if available
    if "pkg_volume_lwh" in df.columns:
        if "pkg_volume_lwh" not in num:
            num.append("pkg_volume_lwh")
        if "pkg_volume_lwh__missing" in df.columns and "pkg_volume_lwh__missing" not in num:
            num.append("pkg_volume_lwh__missing")

    # Keep weight (already in defaults) and add its missingness flag if present
    if "data-product-item-package_dimensions-weight__missing" in df.columns and \
       "data-product-item-package_dimensions-weight__missing" not in num:
        num.append("data-product-item-package_dimensions-weight__missing")

    # Add price missingness flags
    for col in [
        "data-product-price-current_retail_min__missing",
        "data-product-price-reg_retail_max__missing",
    ]:
        if col in df.columns and col not in num:
            num.append(col)

    return num

def assemble_text_corpus(df: pd.DataFrame, text_cols: Sequence[str]) -> List[str]:
    corpus = []
    for _, row in df[text_cols].iterrows():
        parts = []
        for c in text_cols:
            val = row.get(c)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            # If this is a parsed JSON list/dict from earlier, convert gracefully
            if c.endswith("__parsed") or isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            parts.append(strip_html(str(val)))
        corpus.append(" \n ".join(parts))
    return corpus

# -----------------------------
# Text Embedding
# -----------------------------

class TextEmbedder:
    def __init__(self, mode: str = "tfidf", max_features: int = 12000, min_df=3, svd_dim: int = 256,
                 random_state: int = 42):
        self.mode = mode
        self.max_features = max_features
        self.min_df = min_df
        self.svd_dim = svd_dim
        self.random_state = random_state
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._openai = None
        self._sbert_model = None
        self.out_feature_names_: Optional[List[str]] = None  # <-- new

    def fit_transform(self, texts: List[str]) -> sparse.csr_matrix:
        if self.mode == "tfidf":
            return self._fit_transform_tfidf(texts)
        elif self.mode == "openai":
            X = self._fit_transform_openai(texts)
            self.out_feature_names_ = [f"embed_{i}" for i in range(X.shape[1])]
            return X
        elif self.mode == "sbert":
            X = self._fit_transform_sbert(texts)
            self.out_feature_names_ = [f"sbert_{i}" for i in range(X.shape[1])]
            return X
        else:
            raise ValueError(f"Unknown text mode: {self.mode}")

    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        if self.mode == "tfidf":
            X = self._tfidf.transform(texts)
            if self._svd is not None:
                X = self._svd.transform(X)
                X = sparse.csr_matrix(X)
            return X
        elif self.mode == "openai":
            return self._embed_openai(texts)
        elif self.mode == "sbert":
            return self._embed_sbert(texts)
        else:
            raise ValueError(f"Unknown text mode: {self.mode}")

    def get_feature_names_out(self) -> List[str]:
        if self.out_feature_names_ is None:
            return []
        return list(self.out_feature_names_)

    # ---- TF-IDF + SVD ----
    def _fit_transform_tfidf(self, texts: List[str]) -> sparse.csr_matrix:
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        X = self._tfidf.fit_transform(texts)
        if self.svd_dim and self.svd_dim > 0 and X.shape[1] > self.svd_dim:
            self._svd = TruncatedSVD(n_components=self.svd_dim, random_state=self.random_state)
            X = self._svd.fit_transform(X)
            X = sparse.csr_matrix(X)
            self.out_feature_names_ = [f"svd_{i}" for i in range(X.shape[1])]
        else:
            self.out_feature_names_ = list(map(str, self._tfidf.get_feature_names_out()))
        return X

    # ---- OpenAI embeddings (optional) ----
    def _lazy_openai(self):
        if self._openai is not None:
            return self._openai
        try:
            import openai  # type: ignore
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY env var not set")
            openai.api_key = api_key
            self._openai = openai
            return self._openai
        except Exception as e:
            raise RuntimeError(f"OpenAI mode requested but failed to initialize: {e}")

    def _embed_openai(self, texts: List[str]) -> sparse.csr_matrix:
        openai = self._lazy_openai()
        EMB_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        vectors = []
        BATCH = 64
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            try:
                resp = openai.embeddings.create(model=EMB_MODEL, input=batch)  # type: ignore
                vecs = [np.array(d["embedding"], dtype=np.float32) for d in resp.data]
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding error at batch {i}: {e}")
            vectors.extend(vecs)
        X = np.vstack(vectors)
        return sparse.csr_matrix(X)

    def _fit_transform_openai(self, texts: List[str]) -> sparse.csr_matrix:
        return self._embed_openai(texts)

    # ---- Sentence-Transformers ----
    def _lazy_sbert(self):
        if self._sbert_model is not None:
            return self._sbert_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
            self._sbert_model = SentenceTransformer(model_name)
            return self._sbert_model
        except Exception as e:
            raise RuntimeError(f"SBERT mode requested but failed to initialize: {e}")

    def _embed_sbert(self, texts: List[str]) -> sparse.csr_matrix:
        model = self._lazy_sbert()
        X = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=256)
        return sparse.csr_matrix(X)

    def _fit_transform_sbert(self, texts: List[str]) -> sparse.csr_matrix:
        return self._embed_sbert(texts)


# -----------------------------
# Feature engineering pipeline
# -----------------------------

def build_feature_blocks(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    text_cols: Sequence[str],
    text_mode: str = "tfidf",
    text_max_features: int = 12000,
    min_df: float | int = 3,
    text_svd_dim: int = 256,
    random_state: int = 42,
) -> Tuple[FeatureBlocks, List[str]]:
    # ----- Numeric -----
    num_exist = [c for c in numeric_cols if c in df.columns]
    X_num = None
    num_names: List[str] = []
    if num_exist:
        # 1) Median-impute (robust, avoids NaNs)
        imputer = SimpleImputer(strategy="median")
        Xn = imputer.fit_transform(df[num_exist].astype(float))
        # 2) Scale (no centering to preserve sparsity)
        scaler = StandardScaler(with_mean=False)
        Xn = scaler.fit_transform(Xn)
        # 3) To CSR
        X_num = sparse.csr_matrix(Xn)
        num_names = [f"num__{c}" for c in num_exist]

    # ----- Categorical -----
    cat_exist = [c for c in categorical_cols if c in df.columns]
    X_cat = None
    cat_names: List[str] = []
    if cat_exist:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
        Xc = enc.fit_transform(df[cat_exist].astype(str))
        X_cat = sparse.csr_matrix(Xc)
        cat_names = [f"cat__{n}" for n in enc.get_feature_names_out(cat_exist).astype(str).tolist()]

    # ----- Text (raw + parsed helpers) -----
    txt_exist, txt_aug_cols = [], []
    for c in text_cols:
        if c in df.columns:
            txt_exist.append(c)
        if c + "__parsed" in df.columns:
            txt_aug_cols.append(c + "__parsed")
    if "category_path" in df.columns:
        txt_aug_cols.append("category_path")

    X_txt = None
    txt_names: List[str] = []
    if txt_exist or txt_aug_cols:
        cols = list(dict.fromkeys(txt_exist + txt_aug_cols))  # de-dup preserve order
        texts = assemble_text_corpus(df, cols)
        embedder = TextEmbedder(mode=text_mode, max_features=text_max_features, min_df=min_df,
                                svd_dim=text_svd_dim, random_state=random_state)
        Xt = embedder.fit_transform(texts)
        X_txt = sparse.csr_matrix(Xt)
        txt_names = [f"txt__{n}" for n in embedder.get_feature_names_out()]

    blocks = FeatureBlocks(X_num=X_num, X_cat=X_cat, X_txt=X_txt)
    feature_names = num_names + cat_names + txt_names
    return blocks, feature_names

# -----------------------------
# K selection utilities
# -----------------------------

def inertia_elbow_k(inertias: List[float], k_values: List[int]) -> int:
    """Simple elbow via max distance to line connecting first and last inertia points."""
    if len(inertias) < 3:
        return k_values[int(np.argmax(inertias))]
    x = np.array(k_values)
    y = np.array(inertias)
    # Normalize x and y
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
    # Line from first to last
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    v = p2 - p1
    v = v / (np.linalg.norm(v) + 1e-9)
    # Distances
    dists = []
    for xi, yi in zip(x_n, y_n):
        p = np.array([xi, yi])
        # distance from point to line (p1->p2)
        proj_len = np.dot(p - p1, v)
        proj = p1 + proj_len * v
        dist = np.linalg.norm(p - proj)
        dists.append(dist)
    idx = int(np.argmax(dists))
    return k_values[idx]


def select_k_via_metrics(X: sparse.csr_matrix, k_min: int, k_max: int, random_state: int,
                         batch: bool = False, batch_size: int = 2048,
                         metrics_out: Optional[str] = None) -> Tuple[int, pd.DataFrame]:
    """Evaluate K range and return chosen K along with metrics DF."""
    rows = []
    inertias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        if batch:
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size, n_init=10)
        else:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        inertias.append(inertia)
        # Silhouette on a sample for speed when > 20k rows
        sample_sz = 20000 if X.shape[0] > 20000 else None
        try:
            sil = silhouette_score(X, labels, metric="euclidean", sample_size=sample_sz, random_state=random_state)
        except Exception:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(X.toarray() if sparse.issparse(X) and X.shape[1] <= 5000 else X, labels)
        except Exception:
            ch = np.nan
        rows.append({"k": k, "inertia": inertia, "silhouette": sil, "calinski_harabasz": ch})
        print(f"[auto-k] k={k:>3} inertia={inertia:.2e} silhouette={sil:.4f} calinski={ch:.2f}")
    dfm = pd.DataFrame(rows)
    # Normalize metrics to [0,1] for combining
    def _norm(col):
        v = dfm[col].values.astype(float)
        m, M = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-9:
            return np.zeros_like(v)
        return (v - m) / (M - m + 1e-9)
    dfm["inertia_elbow_k"] = inertia_elbow_k(inertias, ks)
    dfm["sil_norm"] = _norm("silhouette")
    dfm["ch_norm"] = _norm("calinski_harabasz")
    # Composite score prioritizes silhouette and CH; elbow influences via choosing between near-ties
    dfm["score"] = 0.55*dfm["sil_norm"] + 0.45*dfm["ch_norm"]
    best_idx = int(dfm["score"].idxmax())
    best_k = int(dfm.loc[best_idx, "k"])
    # If elbow suggests a different K and it's close in score, pick elbow
    elbow_k = int(dfm["inertia_elbow_k"].iloc[0])
    if abs(best_k - elbow_k) >= 2:
        # choose the one with higher score between best_k and elbow_k
        s_best = float(dfm.loc[dfm.k==best_k, "score"].iloc[0])
        s_elbow = float(dfm.loc[dfm.k==elbow_k, "score"].iloc[0]) if (dfm.k==elbow_k).any() else -1
        if s_elbow > 0 and s_elbow >= s_best * 0.98:
            best_k = elbow_k
    if metrics_out:
        dfm.to_csv(metrics_out, index=False)
    return best_k, dfm

# -----------------------------
# Clustering
# -----------------------------

def fit_cluster(X: sparse.csr_matrix, k: int, random_state: int, batch: bool, batch_size: int) -> Tuple[Any, np.ndarray]:
    if batch:
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size, n_init=20)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(X)
    return km, labels

# -----------------------------
# Output assembly
# -----------------------------

KEEP_FOR_DOWNSTREAM = [
    # Identifiers
    "_store_id", "_tcin", "data-product-tcin",
    "data-product-item-primary_barcode", "data-product-item-dpci",
    # Links / media
    "data-product-item-enrichment-buy_url", "data-product-item-enrichment-image_info-primary_image-url",
    # Category / brand / type
    "data-product-category-name", "category_path",
    "data-product-item-primary_brand-name", "data-product-item-product_classification-item_type-name",
    # Price range & current
    "data-product-price-current_retail_min", "data-product-price-reg_retail_max",
    "children_price_min", "children_price_max",
    # Package (derived + raw if present)
    "pkg_volume_lwh",
    "data-product-item-package_dimensions-weight", "data-product-item-package_dimensions-depth",
    "data-product-item-package_dimensions-height", "data-product-item-package_dimensions-width",
    # Ratings
    "data-product-ratings_and_reviews-statistics-rating-average",
    "data-product-ratings_and_reviews-statistics-rating-count",
    "data-product-ratings_and_reviews-statistics-review_count",
    # Tariff risk & handling
    "data-product-item-handling-import_designation_description", "children_handling_mode", "tariff_risk",
    "UMAP1", "UMAP2"
]

def clean_enums(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Normalize inconsistent enum strings (case/wording variants) to canonical values.

    Canonical outputs:
      - "Imported"
      - "Made in USA"
      - "Made in USA or Imported"
      - "Made in USA and Imported"
      - "Assembled in USA (foreign/domestic parts)"

    Applies in-place to the given columns (defaults to common handling/origin fields).
    Returns the same DataFrame for convenience.
    """
    if columns is None:
        columns = [
            "data-product-item-handling-import_designation_description",
            "handling-import_designation_description",
            "children_handling_mode",
        ]

    def _canon_origin(val: Any) -> Any:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        s = str(val).strip()
        if not s:
            return np.nan

        # normalize: lower, collapse whitespace, unify "&" -> "and"
        t = s.lower()
        t = t.replace("&", "and")
        t = re.sub(r"\s+", " ", t)
        t = t.replace(" the usa", " usa")  # drop "the" for consistency

        # rules
        if "assem" in t and "usa" in t:
            return "Assembled in USA (foreign/domestic parts)"
        if "made" in t and "usa" in t and "import" in t:
            return "Made in USA and Imported"
        if "made" in t and "usa" in t:
            return "Made in USA"
        if "import" in t:
            return "Imported"

        # fallback: title-case the original string to avoid inventing a value
        return s.strip().title()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_canon_origin)

    return df


def write_enriched_csv(df: pd.DataFrame, labels: np.ndarray, out_path: str) -> None:
    df_out = df.copy()
    df_out["cluster_id"] = labels.astype(int)

    keep = [c for c in KEEP_FOR_DOWNSTREAM if c in df_out.columns]
    ordered_cols = ["cluster_id"] + keep
    # Append any obvious aisle / node ids if present
    for c in [
        "data-product-sales_classification_nodes-node_ids",
    ]:
        if c in df_out.columns and c not in ordered_cols:
            ordered_cols.append(c)

    df_out = df_out[ordered_cols]
    df_out.to_csv(out_path, index=False)

def project_2d(
    X: sparse.csr_matrix,
    method: str,
    random_state: int,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, str]:
    """
    Returns a (n_samples, 2) projection and the method actually used.
    Tries UMAP; if unavailable or it errors, falls back to SVD(2).
    """
    if method == "umap":
        try:
            import umap  # pip install umap-learn
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                verbose=False,
            )
            coords = reducer.fit_transform(X)
            return coords, "umap"
        except Exception as e:
            print(f"[warn] UMAP failed or not installed ({e}); falling back to TruncatedSVD(2).")
            method = "svd"

    # Fallback / explicit SVD
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    coords = svd.fit_transform(X)
    return coords, "svd"

# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    logger = get_logger()

    print(f"Loading {args.input} ...")
    df = load_products_csv(args.input, sample_n=args.sample, random_state=args.random_state)

    print("Parsing embedded JSON-like columns ...")
    df = parse_embedded_json_columns(df)

    # Raw check
    report_nan_in_dataframe(df, logger=logger, label="raw_df")

    print("Deriving tariff risk flags ...")
    df = clean_enums(df)
    df = derive_tariff_risk_flags(df)

    # business-aware numeric derivations (prices, volume, flags)
    df = prepare_numeric_derivations(df)

    # Columns
    numeric_cols = args.numeric_cols.split("|") if args.numeric_cols else DEFAULT_NUMERIC
    for c in ["rating_quality", "rating_value"]:
        if c in df.columns and c not in numeric_cols:
            numeric_cols.append(c)

    # >>> NEW: drop L/W/H, add volume & missingness flags
    numeric_cols = tune_numeric_cols(numeric_cols, df)

    categorical_cols = args.categorical_cols.split("|") if args.categorical_cols else DEFAULT_CATEGORICAL
    text_cols = args.text_cols.split("|") if args.text_cols else DEFAULT_TEXT

    # Pre-transform check on the actual feature set (numeric + categorical)
    cols_for_check = [c for c in numeric_cols + categorical_cols if c in df.columns]
    if cols_for_check:
        report_nan_in_dataframe(df[cols_for_check], logger=logger, label="features_df")

    print("Building feature blocks ...")
    blocks, feature_names = build_feature_blocks(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        text_mode=args.text_mode,
        text_max_features=args.text_max_features,
        min_df=args.min_df,
        text_svd_dim=args.text_svd_dim,
        random_state=args.random_state,
    )

    X = blocks.hstack(*args.weights)
    print(f"Final feature matrix: {X.shape[0]} rows x {X.shape[1]} cols (sparse={sparse.issparse(X)})")

    # Post-transform sparse diagnostics (this will pinpoint exact feature columns)
    report_nan_in_sparse(X, feature_names=feature_names, row_index=df.index, logger=logger, label="X_sparse")

    # Choose K
    if args.k is not None and args.k > 1:
        k = args.k
        print(f"Using provided k = {k}")
    else:
        k_min, k_max = args.auto_k
        print(f"Selecting k in [{k_min}, {k_max}] ...")
        try:
            k, dfm = select_k_via_metrics(
                X=X,
                k_min=k_min,
                k_max=k_max,
                random_state=args.random_state,
                batch=args.batch_kmeans,
                batch_size=args.batch_size,
                metrics_out=args.metrics_out,
            )
        except ValueError as e:
            # If MiniBatchKMeans/KMeans complain about NaN/Inf, dump more context right away
            logger.error("Auto-K failed due to invalid values; running diagnostics before re-raising...")
            report_nan_in_dataframe(df[cols_for_check], logger=logger, label="features_df@auto_k")
            report_nan_in_sparse(X, feature_names=feature_names, row_index=df.index, logger=logger, label="X_sparse@auto_k", show_examples=10)
            raise
        print(f"Chosen k = {k}")

    # Fit clusters (with the same protective diagnostics)
    print("Fitting clusters ...")
    try:
        km, labels = fit_cluster(X, k=k, random_state=args.random_state, batch=args.batch_kmeans, batch_size=args.batch_size)
    except ValueError as e:
        logger.error("Clustering failed due to invalid values; running diagnostics before re-raising...")
        report_nan_in_dataframe(df[cols_for_check], logger=logger, label="features_df@fit")
        report_nan_in_sparse(X, feature_names=feature_names, row_index=df.index, logger=logger, label="X_sparse@fit", show_examples=10)
        raise

    # 2D projection for plotting
    if args.embed_2d:
        coords, used = project_2d(
            X,
            method=args.proj_method,
            random_state=args.random_state,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
        )
        df["UMAP1"] = coords[:, 0]
        df["UMAP2"] = coords[:, 1]
        print(f"Added 2D projection columns: UMAP1, UMAP2 (method={used})")

    print(f"Writing enriched CSV to {args.output} ...")
    write_enriched_csv(df, labels, args.output)
    print("Done.")


if __name__ == "__main__":
    main()


# ```bash
# # default: TF-IDF text, auto K in [2,20]
# python target_cluster.py --input data/product.csv --output data/product_enriched.csv --metrics-out cluster_metrics.csv --auto-k 2 24 --batch-kmeans

# # if you already know K:
# python target_cluster.py --k 12

# # optional: sentence-transformers or OpenAI embeddings
# python target_cluster.py --text-mode sbert         # needs sentence-transformers installed
# python target_cluster.py --text-mode openai        # needs OPENAI_API_KEY set
# ```

