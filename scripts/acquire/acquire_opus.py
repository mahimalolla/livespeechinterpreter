import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import logging

# Set up logging for PEP 8 compliance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def acquire_opus_subset(output_dir="data/raw/opus"):
    """Fetches a 100k subset of OPUS-100 en-es and saves it locally."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "opus_100k.parquet")

    logging.info("Downloading OPUS-100 (en-es) 100k subset...")

    # We use the specific slice 'train[:100000]' to pull exactly 100k pairs
    try:
        dataset = load_dataset("opus100", "en-es", split="train[:100000]")

        logging.info("Converting to standardized DataFrame...")
        # OPUS-100 uses the {'translation': {'en': '...', 'es': '...'}} format
        df = pd.DataFrame({
            'en': [x['en'] for x in dataset['translation']],
            'es': [x['es'] for x in dataset['translation']]
        })

        # Save to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)

        logging.info(f"SUCCESS! Saved {len(df)} pairs to {output_file}")

    except Exception as e:
        logging.error(f"Failed to acquire OPUS data: {e}")


if __name__ == "__main__":
    acquire_opus_subset()