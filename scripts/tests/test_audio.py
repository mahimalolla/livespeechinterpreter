import os
import sys
import unittest
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)


from pre_process.preprocess_nmt import NMTPreprocessor


class TestNMTPipelineLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        my_hf_token = os.environ.get("HF_TOKEN")
        cls.preprocessor = NMTPreprocessor(tokenizer_name="google/gemma-3-4b-it", hf_token=my_hf_token)

    def test_01_remove_html(self):
        """Test to check if the html tags are getting tokenized."""
        dirty_text = "<p>This is a <b>medical</b> patient record <br/> with tags.</p>"
        clean_text = self.preprocessor.remove_html(dirty_text)

        self.assertEqual(clean_text, "This is a medical patient record  with tags.")
        # Test handling of non-strings (should return empty string)
        self.assertEqual(self.preprocessor.remove_html(None), "")

    def test_02_filter_length_ratio(self):
        """Tests greater than 3x is actually getting dropped"""
        # Create a tiny fake dataframe
        df = pd.DataFrame({
            'en': [
                "This is a normal sentence.",  # 5 words (1:1 ratio)
                "This English sentence is incredibly long and just keeps going and going and going and going.",
                # 16 words
                "Short."  #Just a word.
            ],
            'es': [
                "Esta es una oración normal.",  # 5 words (1:1 ratio)
                "Corta.",  # 1 word (16:1 ratio - should drop!)
                "Esta oración en español es increíblemente larga y sigue y sigue."
                # 11 words (1:11 ratio - should drop!)
            ]
        })

        filtered_df = self.preprocessor.filter_length_ratio(df, max_ratio=3.0)

        # Only the first row should survive
        self.assertEqual(len(filtered_df), 1, "Failed to drop mismatched sentences!")
        self.assertEqual(filtered_df.iloc[0]['en'], "This is a normal sentence.")

    def test_03_full_pipeline_and_tokenization(self):
        """Tests deduplication, cleaning, and Gemma 3 tokenization all at once."""
        df = pd.DataFrame({
            'en': ["Hello <b>world</b>", "Hello <b>world</b>"],  #Handling Duplicates
            'es': ["Hola mundo", "Hola mundo"]
        })

        processed_df = self.preprocessor.process_data(df)


        self.assertEqual(len(processed_df), 1, "Failed to drop duplicate rows!")

        # 2. HTML Removal
        self.assertEqual(processed_df.iloc[0]['en'], "Hello world", "HTML was not stripped in the pipeline!")

        # 3. If getting tokenized
        self.assertIn('en_tokens', processed_df.columns, "English tokens column is missing!")
        self.assertIn('es_tokens', processed_df.columns, "Spanish tokens column is missing!")

      
        en_tokens = processed_df.iloc[0]['en_tokens']
        self.assertIsInstance(en_tokens, list)
        self.assertIsInstance(en_tokens[0], int)


if __name__ == '__main__':
    unittest.main(verbosity=2)