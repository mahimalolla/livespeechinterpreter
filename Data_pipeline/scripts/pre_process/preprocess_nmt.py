import os
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from transformers import AutoTokenizer

from dotenv import load_dotenv
import os

load_dotenv()
my_hf_token = os.environ.get("HF_TOKEN")

# Set up logging for PEP 8 compliance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NMTPreprocessor:
    def __init__(self, tokenizer_name="google/gemma-3-4b-it", hf_token=None):
        logging.info(f"Loading tokenizer: {tokenizer_name}")
        # We must pass the token here to bypass Google's gated repo wall!
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

    def remove_html(self, text):
        """Strips out rogue HTML tags from web-scraped data."""
        if not isinstance(text, str):
            return ""
        return re.sub(r'<[^>]+>', '', text)

    def filter_length_ratio(self, df, max_ratio=3.0):
        """Removes pairs where one sentence is more than 3x longer than the other."""
        logging.info(f"Filtering length ratios > {max_ratio}x...")

        # Calculate word counts for both sides
        en_len = df['en'].str.split().str.len()
        es_len = df['es'].str.split().str.len()

        # Avoid division by zero by dropping empty strings first
        valid_lengths = (en_len > 0) & (es_len > 0)
        df = df[valid_lengths].copy()
        en_len, es_len = en_len[valid_lengths], es_len[valid_lengths]

        # Calculate ratio
        ratio = en_len / es_len

        # Keep rows where the ratio is between 1/3 and 3
        mask = (ratio <= max_ratio) & (ratio >= (1.0 / max_ratio))
        filtered_df = df[mask].copy()

        logging.info(f"Dropped {len(df) - len(filtered_df)} pairs due to extreme length mismatches.")
        return filtered_df

    def process_data(self, df):
        """Runs the complete NMT cleaning and tokenization pipeline."""
        # 1. Deduplicate
        initial_len = len(df)
        df = df.drop_duplicates(subset=['en', 'es']).copy()
        logging.info(f"Deduplicated: Dropped {initial_len - len(df)} redundant rows.")

        # 2. Remove HTML
        logging.info("Scrubbing HTML tags...")
        df['en'] = df['en'].apply(self.remove_html)
        df['es'] = df['es'].apply(self.remove_html)

        # 3. Filter length ratio
        df = self.filter_length_ratio(df, max_ratio=3.0)

        # 4. Tokenize using Gemma 3
        logging.info("Tokenizing English (source) text...")
        # We use max_length to ensure no sequence blows up the GPU later
        df['en_tokens'] = self.tokenizer(df['en'].tolist(), truncation=True, max_length=128)['input_ids']

        logging.info("Tokenizing Spanish (target) text...")
        df['es_tokens'] = self.tokenizer(df['es'].tolist(), truncation=True, max_length=128)['input_ids']

        return df


def main():
    # Path logic to ensure it runs from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Pointing to the specific files we successfully downloaded earlier
    opus_file = os.path.join(project_root, "data", "raw", "opus", "opus_100k.parquet")
    domain_file = os.path.join(project_root, "data", "raw", "domain_data", "domain_specific_raw.parquet")

    output_dir = os.path.join(project_root, "data", "processed")
    output_file = os.path.join(output_dir, "nmt_processed.parquet")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(opus_file) or not os.path.exists(domain_file):
        logging.error(
            "Missing raw data files! Please ensure your acquire_opus.py and acquire_domain_data.py scripts finished successfully.")
        return

    # CRITICAL: Since you are using Gemma 3, you MUST provide your Hugging Face token here
    # Replace "YOUR_ACTUAL_TOKEN_HERE" with your copied token!

    try:
        preprocessor = NMTPreprocessor(tokenizer_name="google/gemma-3-4b-it", hf_token=my_hf_token)
    except Exception as e:
        logging.error(f"Tokenizer failed to load. Did you paste your token? Error: {e}")
        return

    logging.info("Loading and merging raw Parquet files...")
    df_opus = pd.read_parquet(opus_file)
    df_domain = pd.read_parquet(domain_file)

    # Merge the OPUS and Domain data together into one big dataset
    df = pd.concat([df_opus, df_domain], ignore_index=True)
    logging.info(f"Total raw pairs combined: {len(df)}")

    logging.info("Starting NMT preprocessing pipeline...")
    processed_df = preprocessor.process_data(df)

    logging.info(f"Saving processed data to {output_file}...")
    table = pa.Table.from_pandas(processed_df)
    pq.write_table(table, output_file)
    logging.info("SUCCESS! NMT preprocessing complete.")


if __name__ == "__main__":
    main()