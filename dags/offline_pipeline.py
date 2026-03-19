from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, "/opt/airflow/scripts")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

def node0():
    from dvc_pull import dvc_pull_or_download
    return dvc_pull_or_download()
def node1(): from download_datasets import download_datasets; download_datasets()
def node2(): from preprocess import preprocess_datasets; preprocess_datasets()
def node3(): from split_dataset import split_dataset; split_dataset()
def node4(): from ge_stats import generate_ge_stats; generate_ge_stats()
def node5(): from schema_inference import infer_schema; infer_schema()
def node6(): from anomaly_detection import detect_anomalies; detect_anomalies()
def node6b(): from bias_detection import detect_bias; detect_bias()
def node7(): from dataset_slicing import create_slices; create_slices()
def node8(): from dataset_approval import approve_dataset; approve_dataset()
def node9(): from upload_gcs import upload_to_gcs; upload_to_gcs()
def node_dvc_push(): from dvc_push import dvc_push_datasets; dvc_push_datasets()
def node10(): from trigger_online import trigger_online_pipeline; trigger_online_pipeline()

with DAG(
    dag_id="offline_translation_pipeline",
    default_args=default_args,
    description="Offline data pipeline for bilingual translation fine-tuning",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "translation", "offline"],
) as dag:

    t0 = PythonOperator(task_id="node0_dvc_pull_or_download", python_callable=node0)

    def branch_after_node0(**context):
        # True means raw data is already present in /opt/airflow/data/raw (either local or restored from DVC).
        dvc_restored = context["ti"].xcom_pull(task_ids="node0_dvc_pull_or_download")
        if dvc_restored:
            return "start_from_node2"
        return "start_from_node1"

    branch = BranchPythonOperator(task_id="branch_after_node0", python_callable=branch_after_node0)

    start_from_node1 = EmptyOperator(task_id="start_from_node1")
    start_from_node2 = EmptyOperator(task_id="start_from_node2")

    t1 = PythonOperator(task_id="node1_download_datasets", python_callable=node1)
    # Node 2 needs to run whether we came from Node 1 or direct-from-Node-2 branch.
    t2 = PythonOperator(
        task_id="node2_preprocess",
        python_callable=node2,
        trigger_rule="none_failed_min_one_success",
    )
    t3  = PythonOperator(task_id="node3_train_val_test_split",     python_callable=node3)
    t4  = PythonOperator(task_id="node4_ge_statistics",            python_callable=node4)
    t5  = PythonOperator(task_id="node5_schema_inference",         python_callable=node5)
    t6  = PythonOperator(task_id="node6_anomaly_detection",        python_callable=node6)
    t6b = PythonOperator(task_id="node6b_bias_detection", python_callable=node6b)
    t7  = PythonOperator(task_id="node7_dataset_slicing",          python_callable=node7)
    t8  = PythonOperator(task_id="node8_dataset_approval",         python_callable=node8)
    t9  = PythonOperator(task_id="node9_upload_to_gcs",            python_callable=node9)
    t_dvc = PythonOperator(task_id="node9b_dvc_push",              python_callable=node_dvc_push)
    t10 = PythonOperator(task_id="node10_trigger_online_pipeline", python_callable=node10)

    t0 >> branch
    branch >> start_from_node1
    branch >> start_from_node2

    start_from_node1 >> t1 >> t2
    start_from_node2 >> t2

    t2 >> t3 >> t4 >> t5 >> t6 >> t6b >> t7 >> t8 >> t9 >> t_dvc >> t10