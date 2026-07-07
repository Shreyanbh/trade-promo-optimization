# Trade Promo Optimization — Customer Segmentation & Recommendation Engine

A production-grade, cloud-agnostic AI data team that ingests your organisation's trade and customer data from **any source**, runs a 5-phase segmentation and recommendation pipeline, and presents results through a real-time dashboard — with every step reviewed and promoted through a formal DEV → PROD governance chain.

---

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start — Local](#3-quick-start--local)
4. [Connecting Your Organisation's Data](#4-connecting-your-organisations-data)
   - 4.1 [Local Files](#41-local-files-csv-excel-parquet-json)
   - 4.2 [Multiple Files / Glob Patterns](#42-multiple-files--glob-patterns)
   - 4.3 [SQL Databases](#43-sql-databases)
   - 4.4 [REST APIs](#44-rest-apis)
   - 4.5 [Cloud Storage — S3](#45-cloud-storage--amazon-s3)
   - 4.6 [Cloud Storage — Azure ADLS Gen2](#46-cloud-storage--azure-adls-gen2)
   - 4.7 [Cloud Storage — Google Cloud Storage](#47-cloud-storage--google-cloud-storage)
   - 4.8 [Multiple Sources for the Same Table](#48-multiple-sources-for-the-same-table)
   - 4.9 [Upload via Dashboard UI](#49-upload-via-dashboard-ui)
5. [Cloud Environments](#5-cloud-environments)
   - 5.1 [AWS (S3 + EMR + Bedrock)](#51-aws--s3--emr--bedrock)
   - 5.2 [Azure (ADLS + Databricks + Azure OpenAI)](#52-azure--adls--databricks--azure-openai)
   - 5.3 [GCP (GCS + Dataproc)](#53-gcp--gcs--dataproc)
   - 5.4 [Databricks (any cloud)](#54-databricks-any-cloud)
   - 5.5 [Local-only (no cloud account needed)](#55-local-only-no-cloud-account-needed)
6. [Pipeline Phases](#6-pipeline-phases)
7. [DEV → PROD Governance](#7-dev--prod-governance)
8. [Dashboard Walkthrough](#8-dashboard-walkthrough)
9. [Configuration Reference](#9-configuration-reference)
   - 9.1 [config.yaml](#91-configyaml)
   - 9.2 [sources.yaml](#92-sourcesyaml)
   - 9.3 [.env variables](#93-env-variables)
10. [Docker Deployment](#10-docker-deployment)
11. [Agent Team Structure](#11-agent-team-structure)
12. [Project Structure](#12-project-structure)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What This Tool Does

You bring your organisation's data (CRM, ERP, POS, e-commerce, promotions calendar — from any system, in any format). This tool:

- **Ingests** your data from files, databases, cloud buckets, or REST APIs — automatically detects column names even when they differ from the expected schema.
- **Segments** your customers using KMeans clustering on RFM + CLV + promo-sensitivity features.
- **Recommends** the best promotion for each customer segment using collaborative filtering (ALS/NMF).
- **Generates** boardroom-ready HTML reports and dashboard views.
- **Governs** every step through a review chain: no artefact moves from DEV to PROD without sign-off from a senior team member.

All of this happens on whatever infrastructure you already have — local laptop, AWS, Azure, GCP, or Databricks — with a single config change.

---

## 2. Architecture Overview

```
Your Data                Pipeline                    Governance              Output
─────────                ────────                    ──────────              ──────
CRM database  ─┐
ERP tables    ─┤  sources.yaml     Phase 1 Ingest    Work Reviews           Dashboard
S3 bucket     ─┼──────────────►   Phase 2 EDA    ──────────────►  DEV ──► PROD Report
REST API      ─┤  auto column      Phase 3 Features  Senior sign-offs       Models
Excel files   ─┘  detection        Phase 4 Models    DEV→PROD Promotion     Segments
                                   Phase 5 Reports   CEO final approval
                     │                                                │
                     ▼                                                ▼
              config.yaml                                      Streamlit Dashboard
              ─────────────                                    ─────────────────────
              storage:  s3 | azure | gcs | local               Upload Data page
              compute: emr | databricks | dataproc | local      Configuration page
              llm: bedrock | azure_openai | anthropic            Environments page
```

---

## 3. Quick Start — Local

### Prerequisites

- Python 3.11+
- Java 17+ (for PySpark — only needed if running the pipeline, not just the dashboard)

```bash
# 1. Clone / enter the project directory
cd customer-recommendation-segmentation

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Copy and fill in your API key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the pipeline (generates synthetic data automatically if no sources.yaml is set up)
python -m src.run_all.pipeline_runner

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at **http://localhost:8501**.

---

## 4. Connecting Your Organisation's Data

All data source configuration lives in **`sources.yaml`** at the project root.
The pipeline needs four logical tables. You can supply each from a different system — as many source files as you like.

| Table | Required | Minimum columns |
|-------|----------|-----------------|
| `customers` | Yes | `customer_id` |
| `transactions` | Yes | `customer_id`, `date`, `amount` |
| `products` | No | `product_id` |
| `promotions` | No | `promo_id` |

Column names are **auto-detected** — you don't need to rename your files. A fuzzy matcher maps `ACCT_NUMBER` → `customer_id`, `INVOICE_DATE` → `date`, etc. You confirm or override the mapping in the Upload page of the dashboard.

---

### 4.1 Local Files (CSV, Excel, Parquet, JSON)

```yaml
# sources.yaml
sources:
  - name: customer_master
    type: file
    path: data/crm/customers.csv      # relative to project root, or absolute
    format: csv                        # csv | parquet | excel | json | xml | avro | delta
    maps_to: customers
    column_map:                        # optional — leave empty {} for auto-detection
      CUST_ID: customer_id
      REGION:  region

  - name: transaction_history
    type: file
    path: data/erp/sales_2024.xlsx
    format: excel
    sheet_name: Transactions           # Excel only
    maps_to: transactions
    column_map: {}                     # auto-detect
```

**Supported formats:**

| Format | Extension(s) | Notes |
|--------|-------------|-------|
| CSV | `.csv` | `pd.read_csv`, handles encoding |
| Parquet | `.parquet` | Fast columnar, recommended for large tables |
| Excel | `.xlsx`, `.xls` | Set `sheet_name:` if data is not on the first sheet |
| JSON | `.json` | Records array or newline-delimited |
| XML | `.xml` | Set `xpath:` in `options:` to target a specific element |
| Avro | `.avro` | Requires `pip install fastavro` |
| Delta | — | Set path to Delta table directory; requires `pip install deltalake` |

---

### 4.2 Multiple Files / Glob Patterns

All matched files are automatically **unioned** into one DataFrame.

```yaml
sources:
  - name: regional_pos_files
    type: file
    path: data/pos/2024_Q*.parquet      # glob — matches Q1, Q2, Q3, Q4
    format: parquet
    maps_to: transactions
    column_map:
      SALES_ORDER_ID: transaction_id
      ACCT_NUMBER:    customer_id
      SKU:            product_id
      INVOICE_DATE:   date
      NET_AMOUNT:     amount
```

Works for local paths. For S3/Azure/GCS, use the cloud prefix (see §4.5–4.7).

---

### 4.3 SQL Databases

Requires `pip install sqlalchemy` plus the driver for your database.

```yaml
sources:
  - name: erp_transactions
    type: database
    connection: postgresql://analyst:${DB_PASSWORD}@db.internal:5432/erp_prod
    query: >
      SELECT order_id, customer_id, product_code, order_date,
             net_revenue, sales_channel, promo_applied
      FROM   fact_sales
      WHERE  order_date >= '2024-01-01'
    maps_to: transactions
    column_map:
      order_id:      transaction_id
      product_code:  product_id
      order_date:    date
      net_revenue:   amount
      sales_channel: channel
      promo_applied: promo_code
```

**Supported databases and their drivers:**

| Database | Connection prefix | `pip install` |
|----------|------------------|---------------|
| PostgreSQL | `postgresql://` | `psycopg2-binary` |
| MySQL | `mysql+pymysql://` | `pymysql` |
| SQLite | `sqlite:///path/to/file.db` | *(built-in)* |
| Snowflake | `snowflake://user:pass@account/db/schema` | `snowflake-sqlalchemy` |
| BigQuery | `bigquery://project/dataset` | `sqlalchemy-bigquery` |
| Redshift | `redshift+redshift_connector://` | `redshift-connector sqlalchemy-redshift` |
| SQL Server | `mssql+pyodbc://` | `pyodbc` |

**Snowflake example:**

```yaml
sources:
  - name: snowflake_customers
    type: database
    connection: snowflake://${SNOWFLAKE_USER}:${SNOWFLAKE_PASSWORD}@${SNOWFLAKE_ACCOUNT}/PROD_DB/PUBLIC
    query: SELECT * FROM DIM_CUSTOMER WHERE IS_ACTIVE = TRUE
    maps_to: customers
    column_map:
      CUSTOMER_KEY:    customer_id
      SALES_TERRITORY: region
      BIRTH_YEAR:      age
      LOYALTY_STATUS:  loyalty_tier
```

**BigQuery example:**

```yaml
sources:
  - name: bigquery_transactions
    type: database
    connection: bigquery://${GCP_PROJECT}/analytics
    query: >
      SELECT transaction_id, customer_id, product_id,
             DATE(created_at) AS date, amount
      FROM   `analytics.fact_transactions`
      WHERE  DATE(created_at) >= '2024-01-01'
    maps_to: transactions
    column_map: {}
```

---

### 4.4 REST APIs

```yaml
sources:
  - name: product_catalog_api
    type: api
    url: https://api.company.com/v2/products
    method: GET
    headers:
      Authorization: Bearer ${PRODUCT_API_TOKEN}
      Accept: application/json
    pagination:
      type: page           # page | offset | cursor | link | none
      page_param: page
      size_param: per_page
      size: 200
      max_pages: 50
    data_path: data.products   # dot-path into response JSON to find the records array
    maps_to: products
    column_map:
      sku:          product_id
      display_name: product_name
      division:     category
      retail_price: price
```

**Pagination types:**

| Type | How it works |
|------|-------------|
| `page` | Increments `page` query param until empty response |
| `offset` | Increments `offset` by `size` until empty response |
| `cursor` | Reads `cursor_field` from response, sends as `cursor_param` in next request |
| `link` | Follows `Link: <url>; rel="next"` response header |
| `none` | Single request, no pagination |

**Auth types:**

```yaml
# Bearer token
auth:
  type: bearer
  token: ${MY_TOKEN}

# Basic auth
auth:
  type: basic
  username: ${API_USER}
  password: ${API_PASS}

# API key in header
auth:
  type: api_key
  header: X-API-Key
  key: ${API_KEY}
```

---

### 4.5 Cloud Storage — Amazon S3

```yaml
sources:
  - name: s3_promo_calendar
    type: file
    path: s3://${S3_BUCKET}/promo/calendar/*.csv
    format: csv
    maps_to: promotions
    column_map:
      PromoID:      promo_id
      ItemSKU:      product_id
      DiscountRate: discount_pct
      StartDt:      start_date
      EndDt:        end_date
      MechanicType: promo_type
```

Set in `.env`:

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=my-tpo-bucket
```

Install: `pip install boto3`

---

### 4.6 Cloud Storage — Azure ADLS Gen2

```yaml
sources:
  - name: adls_customer_master
    type: file
    path: abfs://${AZURE_CONTAINER}@${AZURE_STORAGE_ACCOUNT}.dfs.core.windows.net/crm/customers.parquet
    format: parquet
    maps_to: customers
    column_map:
      CustID:  customer_id
      RegCode: region
```

Set in `.env`:

```
AZURE_STORAGE_ACCOUNT=mystorageaccount
AZURE_CONTAINER=tpo-data
AZURE_STORAGE_CREDENTIAL=        # leave blank for managed identity / az login
```

Install: `pip install azure-storage-file-datalake azure-identity`

---

### 4.7 Cloud Storage — Google Cloud Storage

```yaml
sources:
  - name: gcs_transactions
    type: file
    path: gs://${GCS_BUCKET}/transactions/2024/*.parquet
    format: parquet
    maps_to: transactions
    column_map: {}
```

Set in `.env`:

```
GCS_BUCKET=my-tpo-bucket
GCP_PROJECT=my-gcp-project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Install: `pip install google-cloud-storage google-auth`

---

### 4.8 Multiple Sources for the Same Table

Sources that share the same `maps_to` target are **automatically unioned** after column mapping. This lets you pull from several systems into one consolidated table without writing any code.

```yaml
sources:
  - name: online_transactions
    type: file
    path: data/ecomm/orders_*.parquet
    maps_to: transactions
    column_map:
      order_uuid: transaction_id
      buyer_id:   customer_id

  - name: instore_pos_transactions
    type: database
    connection: postgresql://${POS_DB_USER}:${POS_DB_PASS}@pos-db:5432/pos
    query: SELECT * FROM pos_receipts WHERE receipt_date >= '2024-01-01'
    maps_to: transactions
    column_map:
      receipt_no:  transaction_id
      loyalty_id:  customer_id

  - name: wholesale_transactions
    type: api
    url: https://wholesale.internal/api/orders
    headers:
      X-API-Key: ${WHOLESALE_API_KEY}
    pagination:
      type: cursor
      cursor_field: next_cursor
      cursor_param: cursor
    maps_to: transactions
    column_map:
      order_number: transaction_id
      account_id:   customer_id
```

All three get loaded, column-mapped, then concatenated into a single `transactions` DataFrame before the pipeline runs.

---

### 4.9 Upload via Dashboard UI

If you don't want to edit `sources.yaml`, use the **Upload Data** page in the dashboard:

1. Open the dashboard: `streamlit run dashboard/app.py`
2. Navigate to **Upload Data** in the sidebar.
3. Drop your CSV / Excel / Parquet files into the uploaders.
4. Review the auto-detected column mappings and adjust any that are wrong.
5. Click **Save Mapped Data & Run Pipeline**.

Uploaded files are saved to `data/uploads/` and can be used immediately.

---

## 5. Cloud Environments

Choose where to store data, where to run Spark, and which LLM to use by editing `config.yaml` (or overriding via `.env`).

---

### 5.1 AWS (S3 + EMR + Bedrock)

**`config.yaml`:**

```yaml
storage:
  provider: s3
  s3:
    bucket: ${S3_BUCKET}
    region: ${AWS_REGION:-us-east-1}

compute:
  provider: emr
  emr:
    master_url: spark://emr-master:7077
    region: ${AWS_REGION:-us-east-1}

llm:
  provider: bedrock
  bedrock:
    region: ${AWS_REGION:-us-east-1}
    model_id: anthropic.claude-3-5-sonnet-20241022-v2:0
```

**`.env`:**

```
STORAGE_PROVIDER=s3
COMPUTE_PROVIDER=emr
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=my-tpo-bucket
```

**Install:** `pip install boto3 s3fs`

---

### 5.2 Azure (ADLS + Databricks + Azure OpenAI)

**`config.yaml`:**

```yaml
storage:
  provider: azure
  azure:
    account_name: ${AZURE_STORAGE_ACCOUNT}
    container: ${AZURE_CONTAINER}
    credential: ${AZURE_STORAGE_CREDENTIAL:-}

compute:
  provider: databricks
  databricks:
    host: ${DATABRICKS_HOST}
    token: ${DATABRICKS_TOKEN}
    cluster_id: ${DATABRICKS_CLUSTER_ID}

llm:
  provider: azure
  azure:
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_KEY}
    deployment: ${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}
```

**`.env`:**

```
STORAGE_PROVIDER=azure
COMPUTE_PROVIDER=databricks
LLM_PROVIDER=azure
AZURE_STORAGE_ACCOUNT=mystorageaccount
AZURE_CONTAINER=tpo-data
AZURE_STORAGE_CREDENTIAL=
DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
DATABRICKS_CLUSTER_ID=0101-...
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

**Install:** `pip install azure-storage-file-datalake azure-identity openai databricks-connect`

---

### 5.3 GCP (GCS + Dataproc)

**`config.yaml`:**

```yaml
storage:
  provider: gcs
  gcs:
    bucket: ${GCS_BUCKET}
    project: ${GCP_PROJECT}

compute:
  provider: dataproc
  dataproc:
    master_url: spark://${DATAPROC_MASTER}:7077
    project: ${GCP_PROJECT}
    region: ${GCP_REGION:-us-central1}
    cluster: ${DATAPROC_CLUSTER}
```

**`.env`:**

```
STORAGE_PROVIDER=gcs
COMPUTE_PROVIDER=dataproc
GCS_BUCKET=my-tpo-bucket
GCP_PROJECT=my-gcp-project
GCP_REGION=us-central1
DATAPROC_MASTER=my-cluster-m
DATAPROC_CLUSTER=my-cluster
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Install:** `pip install google-cloud-storage google-auth`

---

### 5.4 Databricks (any cloud)

Databricks works with S3, ADLS, or GCS underneath. Set `compute.provider: databricks` in `config.yaml` and point `DATABRICKS_HOST` + `DATABRICKS_TOKEN` at your workspace. Storage stays as whatever cloud you're on.

```bash
pip install databricks-connect
databricks-connect configure     # follow prompts to link to your cluster
```

---

### 5.5 Local-only (no cloud account needed)

This is the default. The pipeline runs PySpark in local mode on your laptop.

```yaml
storage:
  provider: local

compute:
  provider: local

llm:
  provider: anthropic
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

No extra packages needed beyond `requirements.txt`.

---

## 6. Pipeline Phases

The pipeline runs end-to-end with `python -m src.run_all.pipeline_runner`.

| Phase | What happens | Key outputs |
|-------|-------------|-------------|
| **Phase 1** Data Ingestion | Loads from `sources.yaml` (or synthetic data), validates schema, profiles quality | `customers.parquet`, `transactions.parquet` |
| **Phase 2** EDA & Cleaning | Deduplication, outlier capping, date parsing, spend distributions | `eda_customers.md`, `eda_transactions.md` |
| **Phase 3** Feature Engineering | RFM scores, CLV projection (BG/NBD), promo-sensitivity elasticity, category affinity | `feature_matrix.parquet` |
| **Phase 4** Segmentation & Recommendations | KMeans (silhouette-optimal k), ALS/NMF collaborative filtering, precision@k / NDCG@k evaluation | `segment_report.md`, model `.pkl` files |
| **Phase 5** Reporting | Jinja2 HTML report, segment summaries, board-level KPIs | `final_report.html`, `project_state.json` |

Each phase has a **senior review gate** before its artefacts are promoted to PROD (see §7).

---

## 7. DEV → PROD Governance

Every artefact produced in DEV must pass a review chain before it is promoted to PROD. The review chain mirrors a real data team hierarchy:

```
Data Engineer 1/2
      │
      ▼
  DE Lead
      │
      ▼
Code Reviewer + PM
      │
      ▼
     CEO  ──► PROD promotion

Data Scientist 1/2
      │
      ▼
Senior Data Scientist
      │
      ▼
   DS Lead
      │
      ▼
     CEO  ──► PROD promotion

Business Analyst 1/2
      │
      ▼
Business Lead
      │
      ▼
      PM
      │
      ▼
     CEO  ──► PROD promotion
```

Each review is logged to `outputs/reports/work_reviews.json`. Each promotion is logged to `outputs/reports/env_promotions.json`. The **Environments** page of the dashboard shows the full history.

---

## 8. Dashboard Walkthrough

Launch with `streamlit run dashboard/app.py`.

| Page | What you'll find |
|------|-----------------|
| **Project Overview** | Phase completion, agent activity timeline, key metrics |
| **Customer Segments** | Cluster profiles, PCA scatter, segment size breakdown |
| **Recommendations** | Per-customer top-N promo recommendations, segment heatmap |
| **Model Performance** | Silhouette scores, elbow curve, precision@k / NDCG@k |
| **Agent Monitor** | Full message bus log from the AI team |
| **Team Communications** | Slack-style activity feed per channel |
| **Work Reviews** | Every senior sign-off with findings and verdict |
| **CEO Approvals** | Final gate decisions per phase |
| **Environments** | DEV workspace, PROD artefacts, promotion history, review chain |
| **Data Sources** | Live data lineage from source to pipeline output |
| **Upload Data** | Upload your own files, confirm column mappings, run pipeline |
| **Configuration** | Active cloud provider, env vars, inline config.yaml editor |

---

## 9. Configuration Reference

### 9.1 config.yaml

```yaml
storage:
  provider: local            # local | s3 | azure | gcs
  local:
    base_path: ./outputs
  s3:
    bucket: ${S3_BUCKET}
    region: ${AWS_REGION:-us-east-1}
  azure:
    account_name: ${AZURE_STORAGE_ACCOUNT}
    container: ${AZURE_CONTAINER}
    credential: ${AZURE_STORAGE_CREDENTIAL:-}
  gcs:
    bucket: ${GCS_BUCKET}
    project: ${GCP_PROJECT}

compute:
  provider: local            # local | databricks | emr | dataproc
  local:
    spark_cores: "*"
    memory: 4g
  databricks:
    host: ${DATABRICKS_HOST}
    token: ${DATABRICKS_TOKEN}
    cluster_id: ${DATABRICKS_CLUSTER_ID}
  emr:
    master_url: ${EMR_MASTER_URL}
    region: ${AWS_REGION:-us-east-1}
  dataproc:
    master_url: spark://${DATAPROC_MASTER}:7077

llm:
  provider: anthropic        # anthropic | bedrock | azure
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-6
  bedrock:
    region: ${AWS_REGION:-us-east-1}
    model_id: anthropic.claude-3-5-sonnet-20241022-v2:0
  azure:
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_KEY}
    deployment: ${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}

pipeline:
  n_customers: 50000
  n_transactions: 500000
  n_products: 500
  n_promotions: 50
  random_seed: 42
```

`${VAR}` — required env var (fails if not set).
`${VAR:-default}` — optional, falls back to `default`.

---

### 9.2 sources.yaml

See [§4](#4-connecting-your-organisations-data) above for the full reference with examples.

Key fields per source entry:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for this source |
| `type` | Yes | `file` \| `database` \| `api` |
| `maps_to` | Yes | Target table: `customers` \| `transactions` \| `products` \| `promotions` |
| `column_map` | No | Dict of `your_column: pipeline_column`. Empty `{}` triggers auto-detection |
| `path` | file only | Local path, glob pattern, or cloud URI |
| `format` | file only | `csv` \| `parquet` \| `excel` \| `json` \| `xml` \| `avro` \| `delta` |
| `connection` | database only | SQLAlchemy URI with `${ENV_VAR}` interpolation |
| `query` | database only | SQL SELECT string (or use `table:` for full table) |
| `url` | api only | Base endpoint URL |
| `pagination` | api only | Pagination config dict (see §4.4) |
| `data_path` | api only | Dot-path into response JSON to find records array |

---

### 9.3 .env variables

Copy `.env.example` to `.env` and fill in what your environment needs.

```bash
cp .env.example .env
```

**Core:**

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for the AI agent team (local LLM mode) |
| `STORAGE_PROVIDER` | Overrides `config.yaml storage.provider` |
| `COMPUTE_PROVIDER` | Overrides `config.yaml compute.provider` |
| `LLM_PROVIDER` | Overrides `config.yaml llm.provider` |

**AWS:**

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM credentials |
| `AWS_REGION` | Default `us-east-1` |
| `S3_BUCKET` | Bucket for data and artefacts |

**Azure:**

| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_ACCOUNT` | Storage account name |
| `AZURE_CONTAINER` | Container for data and artefacts |
| `AZURE_STORAGE_CREDENTIAL` | SAS token or connection string (blank = managed identity) |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` | Azure OpenAI credentials |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `DATABRICKS_CLUSTER_ID` | Databricks workspace |

**GCP:**

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | GCP project ID |
| `GCS_BUCKET` | Bucket for data and artefacts |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON (blank = ADC) |

**Database sources** (add as many as you need in `.env` and reference with `${MY_VAR}` in `sources.yaml`):

```
DB_PASSWORD=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=...
PRODUCT_API_TOKEN=...
WHOLESALE_API_KEY=...
```

---

## 10. Docker Deployment

### Dashboard only

```bash
docker compose up dashboard
# Opens at http://localhost:8501
```

### Full pipeline + dashboard

```bash
docker compose --profile pipeline up
```

### With local S3 (MinIO — useful for testing without an AWS account)

```bash
docker compose --profile s3-local up
# MinIO console → http://localhost:9001  (admin / adminpassword)
# Set S3_BUCKET=tpo-local-bucket and AWS_ENDPOINT_URL=http://minio:9000 in .env
```

### Build with cloud extras

```bash
docker build \
  --build-arg CLOUD_EXTRAS="boto3 s3fs" \
  -t tpo-tool:aws .
```

Available extras (space-separated): `boto3 s3fs` (AWS), `azure-storage-file-datalake azure-identity openai` (Azure), `google-cloud-storage google-auth` (GCP), `databricks-connect` (Databricks), `sqlalchemy psycopg2-binary` (PostgreSQL), `snowflake-sqlalchemy` (Snowflake).

---

## 11. Agent Team Structure

The AI team has 17 agents. Each agent is an autonomous Python process backed by Claude. They communicate through an async message bus and hand off work phase by phase.

| Agent | Role | Review authority |
|-------|------|-----------------|
| `project_manager` | Orchestrates all phases, assigns tasks, tracks milestones | Approves phase completions |
| `ceo` | Final gate for all DEV→PROD promotions | Signs off every phase |
| `de_lead` | Reviews and approves DE work | Promotes DE artefacts |
| `data_engineer_1` | Data ingestion, schema validation, ETL | — |
| `data_engineer_2` | Quality checks, storage optimisation, archiving | — |
| `ds_lead` | Reviews and approves all DS work | Promotes DS artefacts |
| `senior_data_scientist` | Reviews DS1/DS2 outputs before DS Lead | — |
| `data_scientist_1` | EDA, CLV, promo-sensitivity, feature engineering | — |
| `data_scientist_2` | Segmentation, recommender, evaluation | — |
| `business_lead` | Defines KPIs, approves segments, signs off reports | Promotes business artefacts |
| `marketing_analyst` | Reviews segment applicability for marketing | — |
| `finance_analyst` | Reviews CLV and promo-ROI assumptions | Approves with conditions |
| `ml_engineer` | Reviews model deployment readiness | — |
| `business_analyst_1` | Visualisations, segment reports, dashboard data | — |
| `business_analyst_2` | Stakeholder updates, requirements documentation | — |
| `code_reviewer` | Reviews pipeline code quality | — |

---

## 12. Project Structure

```
customer-recommendation-segmentation/
│
├── config.yaml                   <- Cloud provider + pipeline config
├── sources.yaml                  <- Your data source registry  <- START HERE
├── .env.example                  <- Copy to .env, fill in credentials
├── requirements.txt              <- Core dependencies
├── requirements-cloud.txt        <- Optional cloud deps (install what you need)
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/structured/           <- Default local source paths
│   └── uploads/                  <- Files saved from the dashboard Upload page
│
├── outputs/
│   ├── models/                   <- Trained model artefacts (.joblib, .pkl)
│   ├── reports/                  <- JSON logs, HTML reports, Markdown summaries
│   └── visualizations/           <- PNG / HTML chart exports
│
├── dashboard/
│   └── app.py                    <- Streamlit dashboard (12 pages)
│
├── src/
│   ├── config/
│   │   ├── settings.py           <- Paths, model params, constants
│   │   └── config_loader.py      <- Loads config.yaml with ${VAR} interpolation
│   │
│   ├── cloud/
│   │   ├── storage.py            <- StorageAdapter: local / S3 / Azure / GCS
│   │   ├── spark.py              <- SparkSessionFactory: local / Databricks / EMR / Dataproc
│   │   └── llm.py                <- LLMAdapter: Anthropic / Bedrock / Azure OpenAI
│   │
│   ├── ingestion/
│   │   ├── schema_mapper.py      <- Fuzzy column name detection + mapping UI data
│   │   ├── data_validator.py     <- Row-level validation + referential integrity
│   │   ├── source_loader.py      <- Orchestrates sources.yaml -> 4 pipeline tables
│   │   └── connectors/
│   │       ├── file_connector.py       <- CSV / Parquet / Excel / JSON / XML / Avro / Delta
│   │       ├── database_connector.py   <- SQLAlchemy (Postgres, MySQL, Snowflake, BigQuery ...)
│   │       └── api_connector.py        <- REST API with all pagination types
│   │
│   ├── phase1/  <- Ingestion + schema validation
│   ├── phase2/  <- EDA + cleaning
│   ├── phase3/  <- Feature engineering (RFM, CLV, promo sensitivity)
│   ├── phase4/  <- Segmentation (KMeans) + recommendations (ALS/NMF)
│   ├── phase5/  <- Reporting (HTML) + dashboard data export
│   └── run_all/
│       └── pipeline_runner.py    <- Entry point: runs phases 1-5 with review gates
│
└── agentic_ai/
    ├── agents/                   <- 17 autonomous agent classes
    ├── communication/            <- Async message bus
    ├── state/                    <- Thread-safe shared project state
    └── tools/                    <- Data, model, report, pipeline tool wrappers
```

---

## 13. Troubleshooting

**`ModuleNotFoundError: No module named 'pyspark'`**

```bash
pip install pyspark==3.5.0
# Also requires Java 17+ on PATH
```

**`JAVA_HOME is not set` on Windows**

Download [Eclipse Temurin 17](https://adoptium.net/) and set:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17..."
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
```

**`Error loading source 'my_source': No module named 'sqlalchemy'`**

Install the driver for your database (see §4.3 table).

**`EnvironmentError: Required env var ${DB_PASSWORD} is not set`**

Add the variable to your `.env` file. The source loader substitutes `${VAR}` at runtime.

**Dashboard loads slowly on page switch**

All JSON and Parquet files are cached for 5 minutes. If still slow, the Agent Monitor paginates at 20 events per page by default — that's the most common cause.

**`streamlit: error: File does not exist`**

Run from the project root:

```bash
cd customer-recommendation-segmentation
streamlit run dashboard/app.py
```

**S3 / Azure / GCS auth errors**

- AWS: ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set, or configure an IAM role / instance profile.
- Azure: leave `AZURE_STORAGE_CREDENTIAL` blank to use `az login` (DefaultAzureCredential), or paste a SAS token.
- GCP: set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON, or run `gcloud auth application-default login`.

**`fastavro` / `deltalake` not found**

These are optional. Install only if you need those formats:

```bash
pip install fastavro        # Avro files
pip install deltalake       # Delta Lake tables
```
