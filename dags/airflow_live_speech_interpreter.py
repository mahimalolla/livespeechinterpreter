from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

# Set base directory for scripts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def run_script(script_path):
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Script failed: {script_path}\n{result.stderr}")
    return result.stdout

def acquire_opus():
    script = os.path.join(BASE_DIR, "acquire", "acquire_opus.py")
    return run_script(script)

def acquire_librispeech():
    script = os.path.join(BASE_DIR, "acquire", "acquire_librispeech.py")
    return run_script(script)

def acquire_domain_data():
    script = os.path.join(BASE_DIR, "acquire", "acquire_domain_data.py")
    return run_script(script)

def preprocess_audio():
    script = os.path.join(BASE_DIR, "pre_process", "preprocess_audio.py")
    return run_script(script)

def preprocess_nmt():
    script = os.path.join(BASE_DIR, "pre_process", "preprocess_nmt.py")
    return run_script(script)

def run_evaluation():
    # Placeholder: Add evaluation script path if available
    return "Evaluation step (add script if needed)"

def save_results():
    # Placeholder: Add saving logic if needed
    return "Results saved (add script if needed)"

default_args = {
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'live_speech_interpreter_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

acquire_opus_task = PythonOperator(
    task_id='acquire_opus',
    python_callable=acquire_opus,
    dag=dag
)
acquire_librispeech_task = PythonOperator(
    task_id='acquire_librispeech',
    python_callable=acquire_librispeech,
    dag=dag
)
acquire_domain_data_task = PythonOperator(
    task_id='acquire_domain_data',
    python_callable=acquire_domain_data,
    dag=dag
)
preprocess_audio_task = PythonOperator(
    task_id='preprocess_audio',
    python_callable=preprocess_audio,
    dag=dag
)
preprocess_nmt_task = PythonOperator(
    task_id='preprocess_nmt',
    python_callable=preprocess_nmt,
    dag=dag
)
evaluation_task = PythonOperator(
    task_id='run_evaluation',
    python_callable=run_evaluation,
    dag=dag
)
save_results_task = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    dag=dag
)

# Set task dependencies
acquire_opus_task >> acquire_domain_data_task
acquire_librispeech_task >> preprocess_audio_task
acquire_domain_data_task >> preprocess_nmt_task
preprocess_audio_task >> evaluation_task
preprocess_nmt_task >> evaluation_task
evaluation_task >> save_results_task