import os
import sys
import time
import subprocess
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DIR = os.path.join(PROJECT_ROOT, 'scripts')


def run_script(script_path):
    """Runs a Python script as a subprocess with full debug output."""
    logger.info(f"[DEBUG] Starting script: {script_path}")
    logger.info(f"[DEBUG] Project root: {PROJECT_ROOT}")
    logger.info(f"[DEBUG] Python executable: {sys.executable}")
    logger.info(f"[DEBUG] Script exists: {os.path.exists(script_path)}")

    start_time = time.time()

    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    elapsed = time.time() - start_time

    if result.stdout:
        logger.info(f"[STDOUT]\n{result.stdout}")
    if result.stderr:
        logger.warning(f"[STDERR]\n{result.stderr}")

    logger.info(f"[DEBUG] Script finished in {elapsed:.1f}s with exit code {result.returncode}")

    if result.returncode != 0:
        raise Exception(
            f"Script failed: {script_path}\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )
    return result.stdout


def dvc_pull_callable():
    """Attempts DVC pull; skips gracefully if DVC is not initialized."""
    logger.info("[DEBUG] Checking DVC status...")
    dvc_dir = os.path.join(PROJECT_ROOT, '.dvc')
    if not os.path.isdir(dvc_dir):
        logger.warning(
            "[DEBUG] .dvc directory not found -- DVC is not initialized. "
            "Skipping dvc pull. Data will be acquired fresh by downstream tasks."
        )
        return "SKIPPED: DVC not initialized"

    logger.info("[DEBUG] DVC is initialized. Running dvc pull...")
    result = subprocess.run(
        ['dvc', 'pull'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.stdout:
        logger.info(f"[STDOUT]\n{result.stdout}")
    if result.stderr:
        logger.warning(f"[STDERR]\n{result.stderr}")
    if result.returncode != 0:
        logger.warning(f"[DEBUG] dvc pull returned non-zero ({result.returncode}), continuing anyway.")
    return result.stdout


def acquire_opus_callable():
    logger.info("[DEBUG] >>> Task: acquire_opus")
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_opus.py"))


def acquire_librispeech_callable():
    logger.info("[DEBUG] >>> Task: acquire_librispeech")
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_librispeech.py"))


def acquire_domain_data_callable():
    logger.info("[DEBUG] >>> Task: acquire_domain_data")
    return run_script(os.path.join(BASE_DIR, "acquire", "acquire_domain_data.py"))


def preprocess_audio_callable():
    logger.info("[DEBUG] >>> Task: preprocess_audio")
    input_file = os.path.join(PROJECT_ROOT, "data", "raw", "librispeech", "train_clean_100.parquet")
    logger.info(f"[DEBUG] Expected input: {input_file}")
    logger.info(f"[DEBUG] Input exists: {os.path.exists(input_file)}")
    return run_script(os.path.join(BASE_DIR, "pre_process", "preprocess_audio.py"))


def preprocess_nmt_callable():
    logger.info("[DEBUG] >>> Task: preprocess_nmt")
    opus_file = os.path.join(PROJECT_ROOT, "data", "raw", "opus", "opus_100k.parquet")
    domain_file = os.path.join(PROJECT_ROOT, "data", "raw", "domain_data", "domain_specific_raw.parquet")
    logger.info(f"[DEBUG] OPUS file exists: {os.path.exists(opus_file)}")
    logger.info(f"[DEBUG] Domain file exists: {os.path.exists(domain_file)}")
    return run_script(os.path.join(BASE_DIR, "pre_process", "preprocess_nmt.py"))


def validate_nmt_callable():
    logger.info("[DEBUG] >>> Task: validate_nmt (TFDV)")
    return run_script(os.path.join(BASE_DIR, "validation", "validate_nmt.py"))


def validate_asr_callable():
    logger.info("[DEBUG] >>> Task: validate_asr (TFDV)")
    return run_script(os.path.join(BASE_DIR, "validation", "validate_asr.py"))


def bias_analysis_callable():
    logger.info("[DEBUG] >>> Task: bias_analysis (Data Slicing)")
    return run_script(os.path.join(BASE_DIR, "bias", "run_bias_analysis.py"))


def evaluation_callable():
    logger.info("[DEBUG] >>> Task: evaluation")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    if os.path.isdir(processed_dir):
        files = os.listdir(processed_dir)
        logger.info(f"[DEBUG] Processed files available: {files}")

    validation_dir = os.path.join(PROJECT_ROOT, "data", "validation")
    if os.path.isdir(validation_dir):
        for root, dirs, files in os.walk(validation_dir):
            for f in files:
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, PROJECT_ROOT)
                logger.info(f"[DEBUG] Validation artifact: {rel}")
                if f.endswith(".json"):
                    import json
                    with open(fpath) as jf:
                        data = json.load(jf)
                    logger.info(f"[DEBUG]   Content: {json.dumps(data, indent=2)}")

    logger.info("[DEBUG] Evaluation complete -- TFDV validation artifacts reviewed.")


def save_results_callable():
    logger.info("[DEBUG] >>> Task: save_results")
    logger.info("[DEBUG] All pipeline artifacts saved under data/validation/.")


default_args = {
    'owner': 'Team5',
    'start_date': datetime(2026, 2, 19),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['henilpatel2436@gmail.com'],
}

with DAG(
    'live_speech_interpreter_pipeline',
    default_args=default_args,
    description='End-to-end data pipeline for ASR and NMT components',
    schedule=None,
    catchup=False,
    tags=['mlops', 'speech', 'translation'],
) as dag:

    dvc_pull_task = PythonOperator(
        task_id='dvc_pull_task',
        python_callable=dvc_pull_callable,
    )

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

    preprocess_audio_task = PythonOperator(
        task_id='preprocess_audio_task',
        python_callable=preprocess_audio_callable,
    )

    preprocess_nmt_task = PythonOperator(
        task_id='preprocess_nmt_task',
        python_callable=preprocess_nmt_callable,
    )

    validate_nmt_task = PythonOperator(
        task_id='validate_nmt_task',
        python_callable=validate_nmt_callable,
    )

    validate_asr_task = PythonOperator(
        task_id='validate_asr_task',
        python_callable=validate_asr_callable,
    )

    bias_analysis_task = PythonOperator(
        task_id='bias_analysis_task',
        python_callable=bias_analysis_callable,
    )

    evaluation_task = PythonOperator(
        task_id='evaluation_task',
        python_callable=evaluation_callable,
    )

    save_results_task = PythonOperator(
        task_id='save_results_task',
        python_callable=save_results_callable,
    )

    dvc_pull_task >> [acquire_opus_task, acquire_librispeech_task]
    acquire_librispeech_task >> preprocess_audio_task
    acquire_opus_task >> acquire_domain_data_task >> preprocess_nmt_task

    preprocess_nmt_task >> validate_nmt_task
    preprocess_audio_task >> validate_asr_task

    [validate_nmt_task, validate_asr_task] >> bias_analysis_task >> evaluation_task >> save_results_task

if __name__ == "__main__":
    dag.test()
