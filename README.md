# University Student Success Analytics Platform

A modular analytics platform that helps educational institutions monitor, predict, and improve student outcomes using data pipelines, machine learning, and interactive dashboards.

## Features

- **Synthetic student data generation**
- **ETL pipeline** with data processing & model training
- **Predictive ML model** to flag at-risk students
- **DBT-powered SQL transformations**
- **Airflow DAG automation** for orchestration
- **Streamlit dashboard** with filters, tabs, and KPIs
- **Dockerized setup** for reproducibility

## Architecture

The platform architecture consists of the following components:

```mermaid
flowchart TD
    A[Synthetic Data Generator] --> B[ETL Pipeline (Python, Pandas)]
    B --> C[Data Warehouse (SQLite/Snowflake/Redshift)]
    C --> D[DBT Transformations]
    D --> E[Analytics Reports]
    C --> F[ML Model Training & Prediction]
    F --> G[At-risk Student Flags]
    C --> H[Streamlit Dashboard]
    B -.-> I[Airflow DAG Orchestration]
    D -.-> I
    F -.-> I
    I -.-> B
    I -.-> D
    I -.-> F
```

**Description:**
- **Synthetic Data Generator:** Creates sample student data for development and testing.
- **ETL Pipeline:** Processes and loads data into the warehouse.
- **Data Warehouse:** Stores raw and processed data (SQLite for dev, Snowflake/Redshift for prod).
- **DBT Transformations:** Performs SQL-based data modeling and analytics.
- **ML Model:** Predicts at-risk students using historical data.
- **Streamlit Dashboard:** Visualizes KPIs, trends, and risk predictions.
- **Airflow DAG:** Orchestrates and automates the entire workflow.

## Setup & Installation

### Prerequisites

- Docker 20.10+
- Docker Compose 1.29+
- Python 3.10+

### Clone & Launch

```bash
git clone https://github.com/sLikitha7/StudentSucessAnalyticsPlatform.git
cd student-success-analytics
```

### Environment Setup

Create a `.env` file in the root directory:

```env
# Required
AIRFLOW_UID=50000
AIRFLOW_GID=0
DBT_PROFILES_DIR=/opt/airflow/dbt

# Optional
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
```

### Start the Platform

```bash
docker-compose up --build
```

## Usage Instructions

### Manual Execution

```bash
# Generate synthetic data
python student_analytics_platform.py

# Generate reports
python generate_reports.py
```

### Airflow Orchestration

- Access UI: [http://localhost:8080](http://localhost:8080) (login: `airflow` / `airflow`)
- Trigger DAG:
    ```
    airflow dags trigger student_success_analytics
    ```

## 📂 Outputs

- **Reports:** `./reports/*.csv`
- **Dashboard:** [http://localhost:8501](http://localhost:8501)

## Technology Stack

| Component        | Technology                                 |
|------------------|--------------------------------------------|
| Data Pipeline    | Python, Pandas, SQLAlchemy                 |
| Orchestration    | Apache Airflow 2.7+                        |
| Transformations  | DBT Core                                   |
| Dashboard        | Streamlit                                  |
| Data Warehouse   | SQLite (Dev), Snowflake/Redshift (Prod)    |
| Infrastructure   | Docker, docker-compose                     |

## Deployment Modes

### Local Development

- Built-in with SQLite and local volumes
- Lightweight, quick iterations

### Production Setup

For cloud deployment (e.g., AWS/GCP):

```bash
docker-compose -f docker-compose.prod.yml up
```

## 📈 Performance Metrics

Below are key performance metrics achieved during a typical Airflow pipeline run (with 1,000 synthetic student records):

- **ML Model (Risk Prediction):**
    - **Accuracy:** 94%
    - **Precision:** 0.97 (not at-risk), 0.90 (at-risk)
    - **Recall:** 0.95 (not at-risk), 0.94 (at-risk)
    - **F1-score:** 0.96 (not at-risk), 0.92 (at-risk)
    - **Macro avg F1-score:** 0.94

- **DBT Transformations:**
    - **6 models built** (staging, core, analytics)
    - **Total runtime:** ~3.4 seconds
    - **All models completed successfully (PASS=6, ERROR=0)**

- **Report Generation:**
    - **3 analytics reports** generated (risk, department, course)
    - **Total runtime:** <2 seconds

These results demonstrate the platform's ability to efficiently process, transform, and analyze student data at scale.

## Known Issues & Roadmap

### Current Limitations

- SQLite performance degrades beyond 50,000 records
- Streamlit may lag with >1,000 concurrent users

### Roadmap

- Integrate Snowflake for scalable data warehouse
- Add real-time data streaming pipeline
- Build faculty-facing early intervention alert system

---

**Likitha Shatdarsanam**  
MS in Information Systems and Operations Management – Data Science  
University of Florida

- **Email:** shatdars.likitha@ufl.edu
- **LinkedIn:** [linkedin.com/in/likitha-shatdarsanam-395362194](https://linkedin.com/in/likitha-shatdarsanam-395362194)