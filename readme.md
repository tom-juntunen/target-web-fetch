# Target MN Scraper & Product Clustering

End-to-end pipeline to (1) collect Target (RedSky) **stores & product** JSON for Minnesota, (2) flatten to CSV, and (3) **cluster** products for substitution planning under **tariff risk**.

* **Phase A — Scrape** (`scraper.py`):
  Discovers MN stores, enumerates category PLPs, fetches PDP JSON per (store × TCIN), and exports tidy CSVs.
* **Phase B — Cluster** (`cluster.py`):
  Cleans & featurizes numeric/categorical/text fields, auto-tunes *K*, fits KMeans, writes enriched CSV with cluster diagnostics, labels, neighbors, 2D embeddings, and a per-cluster summary.

---

## Contents

* [Features](#features)
* [Requirements](#requirements)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [Data Inputs & Outputs](#data-inputs--outputs)
* [Scraper Usage](#scraper-usage)
* [Clustering Usage](#clustering-usage)
* [Performance & Resilience](#performance--resilience)
* [Optional: Postgres setup](#optional-postgres-setup)
* [Troubleshooting](#troubleshooting)
* [Legal & Ethics](#legal--ethics)

---

## Features

**Scraper**

* Store discovery via `nearby_stores_v1`, PLP pagination (offset cap respected), PDP per TCIN.
* **Resume-safe** JSONL appenders with (store\_id, tcin) de-duplication.
* HTTP/2 with retries + jitter; optional **Selenium** transport to bypass soft-blocks.
* Robust JSON flattener → dash-named CSV columns.

**Clustering**

* Numeric impute/scale, categorical OHE, text **TF-IDF(+SVD)** by default; optional **SBERT** or **OpenAI** embeddings.
* Auto-K with **Silhouette + Calinski-Harabasz + elbow**.
* Per-item: `cluster_id`, distance to centroid, **outlier flag (95th pct)**, and **top-K in-cluster neighbors**.
* Per-cluster summary CSV (size, label mode, median price, imported share, outlier rate).
* 2D projection (**UMAP** w/ SVD(2) fallback) for plotting.

---

## Requirements

* **Python** 3.12+
* Install dependencies per phase:

```bash
# Scraper
pip install httpx pandas tenacity python-dateutil
# Optional (for --use-selenium)
pip install selenium webdriver-manager

# Clustering
pip install pandas numpy scikit-learn scipy umap-learn
# Optional text embedders
pip install sentence-transformers
# Optional OpenAI path
pip install openai tiktoken python-dotenv
```

> Create a virtualenv first if you like:
> `python -m venv .venv && source .venv/bin/activate`

---

## Quickstart

```bash
# 1) Put inputs in place:
#   - mn_zip_codes.csv
#   - category.csv    (# columns: name,slug,url_path,full_url,n_code,data_id,number_of_children)

# 2) (Optional) sanity check: duplicate ZIPs in column 1
tail -n +2 mn_zip_codes.csv | cut -d',' -f1 | sort | uniq -d

# 3) Run scraper — discover stores then products
python target_scrape.py --mn-zip-csv mn_zip_codes.csv --category-csv category.csv --phase all
# Tip: start small while testing:
# python target_scrape.py --phase stores --max-zip-codes 50
# python target_scrape.py --phase products --max-per-category 200 --concurrency 4 --use-selenium

# 4) Run clustering on exported products CSV
python target_cluster.py \
  --input data/product.csv \
  --output data/product_enriched.csv \
  --metrics-out cluster_metrics.csv \
  --summary-out cluster_summary.csv \
  --auto-k 2 24 \
  --batch-kmeans
```

---

## Configuration

### Environment variables

| Variable                 | Used by | Purpose                      | Default                  |
| ------------------------ | ------- | ---------------------------- | ------------------------ |
| `TARGET_API_KEY`         | scraper | RedSky API key               | public web key baked in  |
| `TARGET_VISITOR_ID`      | scraper | RedSky visitor id            | stable uuid-like         |
| `TARGET_PLP_ENDPOINT`    | scraper | Override PLP base endpoint   | auto-try v2→v1→legacy    |
| `OPENAI_API_KEY`         | cluster | Enables `--text-mode openai` | none                     |
| `OPENAI_EMBEDDING_MODEL` | cluster | Embedding model name         | `text-embedding-3-small` |
| `SBERT_MODEL`            | cluster | Sentence-Transformers model  | `all-MiniLM-L6-v2`       |

Supports a local `.env` (loaded via `python-dotenv`).

---

## Data Inputs & Outputs

**Input files**

* `mn_zip_codes.csv` — MN ZCTAs. Must include column **`ZIP Code`**.
* `category.csv` — category rows (uses `data_id`, `url_path`; can auto-derive `data_id` from `n_code` like `N-5xt4g`).

**Output files (./data)**

* `data/store_raw.jsonl` — raw `nearby_stores_v1` results (deduped by `store_id`)
* `data/store.csv` — flattened stores
* `data/product_raw.jsonl` — one raw PDP JSON per **(store\_id × tcin)**
* `data/product.csv` — flattened products (used by clustering)
* `data/product_enriched.csv` — clustered output
* `cluster_metrics.csv` — Auto-K metrics (when used)
* `cluster_summary.csv` — per-cluster summary (when `--summary-out` is set)

---

## Scraper Usage

```bash
python target_scrape.py --mn-zip-csv mn_zip_codes.csv --category-csv category.csv --phase all
```

**Phases**

| Phase                               | What happens                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `stores`                            | Discover stores for each MN ZIP; write JSONL → CSV                              |
| `products`                          | For each (store × category): PLP enumerate TCINs, fetch PDPs, write JSONL → CSV |
| `all`                               | `stores` then `products`                                                        |
| `export-stores` / `export-products` | Re-flatten JSONL to CSV without network calls                                   |

**Important flags**

| Flag                 | Type | Default | Notes                                      |
| -------------------- | ---- | ------: | ------------------------------------------ |
| `--concurrency`      | int  |       4 | Max parallel HTTP/2 connections            |
| `--max-zip-codes`    | int  |    None | Cap ZIPs for testing                       |
| `--max-per-category` | int  |    None | Cap TCINs per (store × category)           |
| `--use-selenium`     | flag |     off | Fetch PLP/PDP via headless Chrome          |
| `--no-headless`      | flag |     off | Show browser window (useful if challenged) |

**Soft-block handling**

* Randomized page sizes, backoff + VID/UA rotation, and optional **Selenium** transport.
* RedSky **PLP offset ≤ 1199** is enforced; shard large categories if needed.

---

## Clustering Usage

```bash
python target_cluster.py \
  --input data/product.csv \
  --output data/product_enriched.csv \
  --metrics-out cluster_metrics.csv \
  --summary-out cluster_summary.csv \
  --auto-k 2 24 \
  --batch-kmeans
```

**Key options**

| Flag                                                  | Purpose                                         |
| ----------------------------------------------------- | ----------------------------------------------- |
| `--k` or `--auto-k K_MIN K_MAX`                       | Fix K or auto-select within range               |
| `--text-mode {tfidf,sbert,openai}`                    | Text embeddings backend                         |
| `--text-max-features`, `--text-svd-dim`, `--min-df`   | TF-IDF/SVD controls                             |
| `--weights W_NUM W_CAT W_TXT`                         | Block weights when hstacking                    |
| `--numeric-cols`, `--categorical-cols`, `--text-cols` | Pipe-sep column overrides                       |
| `--batch-kmeans`, `--batch-size`                      | MiniBatchKMeans for large datasets              |
| `--embed-2d`, `--proj-method {umap,svd}`              | 2D projection; UMAP if available                |
| `--topk-neighbors`                                    | Top-K in-cluster neighbors per item (default 3) |

**What the enriched CSV includes**

* `cluster_id`, `cluster_label` (mode of category name), `price_band`, `tariff_risk`
* Distances & diagnostics: `cluster_dist`, `cluster_outlier`
* Neighbors: `nn1_id`, `nn1_dist`, `nn2_id`, `nn2_dist`, `nn3_id`, `nn3_dist`
* 2D projection: `UMAP1`, `UMAP2`
* Plus curated product fields (ids, brand, type, prices, ratings, package dims)

**Per-cluster summary (`--summary-out`)**

* `cluster_id`, `n`, `label`, `median_price`, `imported_share`, `outlier_rate`

---

## Performance & Resilience

* Use `--concurrency` to tune HTTP parallelism; start low and scale.
* Prefer **MiniBatchKMeans** (`--batch-kmeans`) on large matrices.
* Use **Selenium** (`--use-selenium`) if you encounter frequent PLP/PDP soft-blocks.
* Auto-K writes `cluster_metrics.csv` so you can inspect Silhouette/CH and inertia curves.

---

## Optional: Postgres setup

> The default pipeline writes files to `./data/`. DB usage is **optional**; include only if you’ll ingest the CSVs.

**Create DB & role**

```sql
-- psql -U postgres
CREATE DATABASE target;
CREATE USER target_admin WITH PASSWORD 'ChangeMe01!' LOGIN;
GRANT ALL ON DATABASE target TO target_admin;
GRANT ALL ON SCHEMA public TO target_admin;
```

Connect:

```bash
psql -U target_admin -d target
```

**Migrations (if you add a DB sink):**

* Create Alembic scaffolding and models for `store`, `product`, etc.
* Then: `alembic upgrade head`

---

## Troubleshooting

* **No products exported**: Confirm `category.csv` has `data_id` or `n_code` (like `N-5xt4g`).
* **PLP stops early**: You’ve likely hit offset 1199; shard the category or tighten facets.
* **Soft-blocks (404 near success)**: Use `--use-selenium`, reduce `--concurrency`, or rerun later.
* **UMAP import error**: Install `umap-learn` or switch `--proj-method svd`.
* **OpenAI/SBERT path failing**: Ensure `OPENAI_API_KEY` or `sentence-transformers` is installed; otherwise stick to `--text-mode tfidf`.

---

## Legal & Ethics

This project is for research/analysis. **Respect the Target Terms of Use** and robots guidelines, apply reasonable rate limits, and avoid disrupting services. Only collect and store data you are permitted to handle.

---

### Appendix: Handy one-liners

* **Check duplicate ZIPs** (first CSV column):

  ```bash
  tail -n +2 mn_zip_codes.csv | cut -d',' -f1 | sort | uniq -d
  ```

* **Example: products phase with guardrails**

  ```bash
  python target_scrape.py --category-csv category.csv --phase products --use-selenium --concurrency 4 --max-per-category 175
  ```

* **Example: clustering with fixed K**

  ```bash
  python target_cluster.py --input data/product.csv --output data/product_enriched.csv --k 34 --summary-out cluster_summary.csv
  ```
