import sys
import os
from datetime import timedelta
import pendulum
# Add the parent folder of `airflow/` to the Python path
sys.path.append('/opt/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from student_analytics_platform import main as run_pipeline

default_args = {
    'owner': 'university',
    'depends_on_past': False,
    'start_date': pendulum.now().subtract(days=1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

dag = DAG(
    'student_success_analytics',
    default_args=default_args,
    description='ETL and analytics pipeline for student success',
    schedule='@weekly',
    catchup=False,
    tags=['education', 'analytics'],
)

# Start and end tasks for better visualization
start = EmptyOperator(task_id='start', dag=dag)
end = EmptyOperator(task_id='end', dag=dag)

# Task 1: Run the ETL and ML pipeline
run_pipeline_task = PythonOperator(
    task_id='run_student_analytics_pipeline',
    python_callable=run_pipeline,
    dag=dag,
    execution_timeout=timedelta(minutes=30),
)

# Task 2: Run DBT transformations
run_dbt_task = BashOperator(
    task_id='run_dbt_transformations',
    bash_command='cd /opt/airflow/dbt && dbt run --profiles-dir .',
    dag=dag,
)

# Task 3: Generate reports
generate_reports_task = BashOperator(
    task_id='generate_analytics_reports',
    bash_command='cd /opt/airflow && python student_analytics_platform.py',
    dag=dag,
)

# Set task dependencies
start >> run_pipeline_task >> run_dbt_task >> generate_reports_task >> end
