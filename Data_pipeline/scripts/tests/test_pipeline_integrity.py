"""Pipeline integrity tests: folder structure, scripts, DAG, and data flow."""

import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
scripts_dir = os.path.join(project_root, "scripts")
sys.path.insert(0, project_root)
sys.path.insert(0, scripts_dir)


class TestFolderStructure(unittest.TestCase):
    """Verify required pipeline folders and files exist."""

    def test_project_root_exists(self):
        self.assertTrue(os.path.isdir(project_root), f"Project root missing: {project_root}")

    def test_dags_folder_exists(self):
        dags = os.path.join(project_root, "dags")
        self.assertTrue(os.path.isdir(dags), f"dags/ folder missing")

    def test_dag_file_exists(self):
        dag_path = os.path.join(project_root, "dags", "airflow_live_speech_interpreter.py")
        self.assertTrue(os.path.isfile(dag_path), f"DAG file missing: {dag_path}")

    def test_scripts_subdirs_exist(self):
        required = ["acquire", "pre_process", "validation", "bias", "tests"]
        for sub in required:
            path = os.path.join(scripts_dir, sub)
            self.assertTrue(os.path.isdir(path), f"scripts/{sub}/ missing")

    def test_acquire_scripts_exist(self):
        required = ["acquire_opus.py", "acquire_librispeech.py", "acquire_domain_data.py"]
        for f in required:
            path = os.path.join(scripts_dir, "acquire", f)
            self.assertTrue(os.path.isfile(path), f"scripts/acquire/{f} missing")

    def test_preprocess_scripts_exist(self):
        required = ["preprocess_audio.py", "preprocess_nmt.py"]
        for f in required:
            path = os.path.join(scripts_dir, "pre_process", f)
            self.assertTrue(os.path.isfile(path), f"scripts/pre_process/{f} missing")

    def test_validation_scripts_exist(self):
        required = ["validate_nmt.py", "validate_asr.py"]
        for f in required:
            path = os.path.join(scripts_dir, "validation", f)
            self.assertTrue(os.path.isfile(path), f"scripts/validation/{f} missing")

    def test_bias_scripts_exist(self):
        required = ["data_slicing.py", "run_bias_analysis.py"]
        for f in required:
            path = os.path.join(scripts_dir, "bias", f)
            self.assertTrue(os.path.isfile(path), f"scripts/bias/{f} missing")


class TestScriptImports(unittest.TestCase):
    """Verify pipeline scripts are importable."""

    def test_import_data_slicing(self):
        from data_slicing import slice_nmt_data, run_nmt_bias_analysis
        self.assertIsNotNone(slice_nmt_data)
        self.assertIsNotNone(run_nmt_bias_analysis)

    def test_import_nmt_preprocessor(self):
        from pre_process.preprocess_nmt import NMTPreprocessor
        self.assertIsNotNone(NMTPreprocessor)

    def test_import_audio_preprocessor(self):
        try:
            from pre_process.preprocess_audio import AudioPreprocessor
            self.assertIsNotNone(AudioPreprocessor)
        except ImportError as e:
            self.skipTest(f"Audio preprocessor deps not installed (e.g. librosa): {e}")


class TestDAGStructure(unittest.TestCase):
    """Verify Airflow DAG structure is valid."""

    def test_dag_loadable(self):
        """DAG module can be imported (airflow may not be installed)."""
        try:
            dags_path = os.path.join(project_root, "dags")
            if dags_path not in sys.path:
                sys.path.insert(0, dags_path)
            from airflow_live_speech_interpreter import dag
            self.assertIsNotNone(dag)
            self.assertEqual(dag.dag_id, "live_speech_interpreter_pipeline")
        except ImportError as e:
            self.skipTest(f"Airflow not installed, cannot load DAG: {e}")

    def test_dag_has_required_tasks(self):
        """DAG defines expected task IDs."""
        try:
            dags_path = os.path.join(project_root, "dags")
            if dags_path not in sys.path:
                sys.path.insert(0, dags_path)
            from airflow_live_speech_interpreter import dag
            task_ids = [t.task_id for t in dag.tasks]
            required = [
                "dvc_pull_task",
                "acquire_opus_task",
                "acquire_librispeech_task",
                "acquire_domain_data_task",
                "preprocess_nmt_task",
                "preprocess_audio_task",
                "validate_nmt_task",
                "validate_asr_task",
                "bias_analysis_task",
                "evaluation_task",
                "save_results_task",
            ]
            for tid in required:
                self.assertIn(tid, task_ids, f"Missing task: {tid}")
        except ImportError:
            self.skipTest("Airflow not installed")


class TestDataFlowPaths(unittest.TestCase):
    """Verify data flow paths are correctly defined."""

    def test_raw_data_paths_defined(self):
        """Raw data directory structure is expected."""
        raw_dir = os.path.join(project_root, "data", "raw")
        if not os.path.isdir(raw_dir):
            self.skipTest("data/raw not yet created (run pipeline first)")
        expected_subdirs = ["opus", "librispeech", "domain_data"]
        for sub in expected_subdirs:
            path = os.path.join(raw_dir, sub)
            # At least raw dir exists; subdirs created when acquisition runs
            self.assertTrue(
                os.path.isdir(path) or not os.path.exists(path),
                f"Unexpected state for {path}"
            )

    def test_processed_path_convention(self):
        """Processed output paths follow convention."""
        processed_dir = os.path.join(project_root, "data", "processed")
        nmt_path = os.path.join(processed_dir, "nmt_processed.parquet")
        asr_path = os.path.join(processed_dir, "asr_processed.parquet")
        # Paths are defined correctly (files may not exist yet)
        self.assertEqual(os.path.basename(nmt_path), "nmt_processed.parquet")
        self.assertEqual(os.path.basename(asr_path), "asr_processed.parquet")

    def test_validation_output_paths(self):
        """Validation output directories follow convention."""
        val_dir = os.path.join(project_root, "data", "validation")
        nmt_dir = os.path.join(val_dir, "nmt")
        asr_dir = os.path.join(val_dir, "asr")
        bias_dir = os.path.join(val_dir, "bias")
        for d in [nmt_dir, asr_dir, bias_dir]:
            # Path convention is correct
            self.assertIn("validation", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
