import os
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import logging
import requests
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def acquire_medical_data():
    """Fetches OPUS-100 and isolates medical-leaning segments."""
    logging.info("Fetching OPUS-100 for Medical domain isolation...")
    ds = load_dataset("opus100", "en-es", split="train[:50000]")

    medical_keywords = ['patient', 'treatment', 'doctor', 'hospital', 'clinical', 'medicine', 'médico']

    df = pd.DataFrame({
        'en': [x['en'] for x in ds['translation']],
        'es': [x['es'] for x in ds['translation']]
    })

    mask = df['en'].str.contains('|'.join(medical_keywords), case=False)
    medical_df = df[mask].copy()
    medical_df['domain'] = 'medical'
    return medical_df


def acquire_federal_data():
    """Bypasses OPUS tools and website redirects by downloading directly from the OPUS backend storage."""
    logging.info("Downloading Federal data directly from OPUS backend storage...")

    # OPUS actually hosts files on CSC Pouta object storage.
    # Bypassing the opus.nlpl.eu redirects prevents all 404 and library errors.
    direct_url = "https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/en-es.txt.zip"

    try:
        r = requests.get(direct_url, stream=True)
        r.raise_for_status()

        logging.info("Unzipping directly in memory to protect your disk...")
        # Unzip entirely in memory to avoid bad file saves or permission errors
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            en_filename = "JRC-Acquis.en-es.en"
            es_filename = "JRC-Acquis.en-es.es"

            with z.open(en_filename) as f_en, z.open(es_filename) as f_es:
                # Read exactly 50,000 lines
                en_lines = [next(f_en).decode('utf-8').strip() for _ in range(50000)]
                es_lines = [next(f_es).decode('utf-8').strip() for _ in range(50000)]

        federal_df = pd.DataFrame({'en': en_lines, 'es': es_lines})
        federal_df['domain'] = 'federal'

        logging.info(f"Successfully acquired {len(federal_df)} Federal pairs.")
        return federal_df

    except Exception as e:
        logging.error(f"Federal backend acquisition failed: {e}")
        return pd.DataFrame(columns=['en', 'es', 'domain'])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    raw_dir = os.path.join(project_root, "data", "raw", "domain_data")
    os.makedirs(raw_dir, exist_ok=True)

    med_df = acquire_medical_data()
    fed_df = acquire_federal_data()

    if not fed_df.empty or not med_df.empty:
        domain_df = pd.concat([med_df, fed_df], ignore_index=True)
        output_file = os.path.join(raw_dir, "domain_specific_raw.parquet")

        table = pa.Table.from_pandas(domain_df)
        pq.write_table(table, output_file)
        logging.info(f"SUCCESS! {len(domain_df)} total pairs securely saved to {output_file}")
    else:
        logging.error("No data was acquired.")