# Import necessary libraries and modules
import os
import sys
import subprocess
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Set base directory for scripts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def run_script(script_path):
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.stdout:
        logger.info(f"Output:\n{result.stdout}")
    if result.returncode != 0:
        raise Exception(f"Script failed: {script_path}\n{result.stderr}")
    return result.stdout
def acquire_opus_callable():
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_opus.py"))

def acquire_librispeech_callable():
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_librispeech.py"))

def acquire_domain_data_callable():
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_domain_data.py"))

def preprocess_audio_callable():
    # Path logic for your existing audio DSP script
    return run_script(os.path.join(BASE_DIR, "pre_process", "preprocess_audio.py"))

def preprocess_nmt_callable():
    # Path logic for your existing NMT cleaning script
    return run_script(os.path.join(BASE_DIR, "pre_process", "preprocess_nmt.py"))

def evaluation_callable():
    logger.warning("Evaluation not implemented yet. Add your evaluation logic to complete this step.")

def save_results_callable():
    logger.warning("Save results not implemented yet. Add your save logic to complete this step.")

default_args = {
    'owner': 'Henil',
    'start_date': datetime(2026, 2, 19),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['your_email@example.com'],
}

# DAG
with DAG(
    'live_speech_interpreter_pipeline',
    default_args=default_args,
    description='End-to-end data pipeline for ASR and NMT components',
    catchup=False,
) as dag:

    # Task to pull MD5 hashed data from GCP using DVC
    dvc_pull_task = BashOperator(
        task_id='dvc_pull_task',
        bash_command='dvc pull',
    )

    # Tasks to acquire data
    acquire_opus_task = PythonOperator(
        task_id='acquire_opus_task',
        python_callable=acquire_opus_callable,
    )

    acquire_librispeech_task = PythonOperator(
        task_id='acquire_librispeech_task',
        python_callable=acquire_librispeech_callable,
    )

    acquire_domain_data_task = PythonOperator(
        task_id='acquire_domain_data_task',
        python_callable=acquire_domain_data_callable,
    )

    # Tasks to perform data preprocessing
    preprocess_audio_task = PythonOperator(
        task_id='preprocess_audio_task',
        python_callable=preprocess_audio_callable,
    )

    preprocess_nmt_task = PythonOperator(
        task_id='preprocess_nmt_task',
        python_callable=preprocess_nmt_callable,
    )

    # Task for evaluation
    evaluation_task = PythonOperator(
        task_id='evaluation_task',
        python_callable=evaluation_callable,
    )

    # Task to save metadata
    save_results_task = PythonOperator(
        task_id='save_results_task',
        python_callable=save_results_callable,
    )

    # Set task dependencies
    # Start with DVC Sync
    dvc_pull_task >> [acquire_opus_task, acquire_librispeech_task]

    # Parallel Pipeline Branches
    acquire_librispeech_task >> preprocess_audio_task
    acquire_opus_task >> acquire_domain_data_task >> preprocess_nmt_task

    # Join branches for final steps
    [preprocess_audio_task, preprocess_nmt_task] >> evaluation_task >> save_results_task

if __name__ == "__main__":
    dag.test()