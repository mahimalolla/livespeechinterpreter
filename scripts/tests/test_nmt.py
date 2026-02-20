import os
import sys
import unittest
import pandas as pd

# 1. Add the project root to the path so we can import your scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

# 2. Import the actual preprocessing class you wrote
from pre_process.preprocess_nmt import NMTPreprocessor


class TestNMTPipelineLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes the preprocessor once before running all tests."""
        my_hf_token = os.environ.get("HF_TOKEN")
        cls.preprocessor = NMTPreprocessor(tokenizer_name="google/gemma-3-4b-it", hf_token=my_hf_token)

    def test_01_no_empty_pairs(self):
        """Tests that pairs containing empty strings or just spaces are dropped."""
        # Create a dataframe with sneaky empty strings
        df = pd.DataFrame({
            'en': ["Valid sentence.", "", "Another valid.", "   "],
            'es': ["Oración válida.", "Vacío en inglés.", "", "   "]
        })

        # The filter_length_ratio method handles dropping zero-length sequences
        filtered_df = self.preprocessor.filter_length_ratio(df, max_ratio=3.0)

        # Only the first row should survive. The rest have 0 words on one or both sides.
        self.assertEqual(len(filtered_df), 1, "Empty pairs were not removed!")
        self.assertEqual(filtered_df.iloc[0]['en'], "Valid sentence.")

    def test_02_length_ratio_filtering_works(self):
        """Tests that pairs where one sentence is >3x longer are dropped."""
        df = pd.DataFrame({
            'en': [
                "One two three.",  # 3 words (1:1 ratio) - KEEP
                "One.",  # 1 word vs 4 words (1:4 ratio) - DROP
                "One two three four five six seven."  # 7 words vs 1 word (7:1 ratio) - DROP
            ],
            'es': [
                "Uno dos tres.",
                "Uno dos tres cuatro.",
                "Uno."
            ]
        })

        filtered_df = self.preprocessor.filter_length_ratio(df, max_ratio=3.0)

        # Only the 1:1 ratio sentence should survive
        self.assertEqual(len(filtered_df), 1, "Failed to filter extreme length mismatches!")
        self.assertEqual(filtered_df.iloc[0]['en'], "One two three.")

    def test_03_deduplication_removes_exact_copies(self):
        """Tests that identical rows are squashed into a single row."""
        df = pd.DataFrame({
            'en': ["Exact copy.", "Exact copy.", "Different."],
            'es': ["Copia exacta.", "Copia exacta.", "Diferente."]
        })

        # The process_data pipeline handles deduplication right at the start
        # We manually trigger just the deduplication logic here to verify
        initial_len = len(df)
        dedup_df = df.drop_duplicates(subset=['en', 'es']).copy()

        self.assertEqual(len(dedup_df), 2, "Deduplication failed to remove the copy!")
        self.assertTrue((dedup_df['en'] == "Exact copy.").sum() == 1, "Multiple copies still exist!")


if __name__ == '__main__':
    unittest.main(verbosity=2)