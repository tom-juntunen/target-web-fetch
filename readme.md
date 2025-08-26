readme.md


# Run these SQL commands ahead of time

```
psql -U postgres
```

```
CREATE DATABASE target;
CREATE USER target_admin WITH PASSWORD 'ChangeMe01!' LOGIN;
GRANT ALL ON DATABASE target TO target_admin;
GRANT ALL ON SCHEMA public TO target_admin;
```

```
psql -U target_admin -d target
```

# Create tables using alembic migration

# Check for duplicates in the first column of a csv file 
`tail -n +2 mn_zip_codes.csv | cut -d',' -f1 | sort | uniq -d`


# Run target_scrape.py (Phase 1 - stores)
`python target_scrape.py --max_zip_codes 1 --phase stores`

# Run target_scrape.py (Phase 2 - products)
`python target_scrape.py --max_zip_codes 1 --phase products`

# Run target_cluster.py
`python target_cluster.py --input data/product.csv --output data/product_enriched.csv --metrics-out cluster_metrics.csv --auto-k 34 34  --batch-kmeans`

