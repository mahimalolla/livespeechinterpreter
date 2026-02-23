import os
import requests
import tarfile
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import logging
import soundfile as sf  # Uses your working soundfile library, NOT torchcodec
from tqdm import tqdm



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_and_extract_librispeech(base_dir="data/raw/librispeech"):
    os.makedirs(base_dir, exist_ok=True)

    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    tar_path = os.path.join(base_dir, "train-clean-100.tar.gz")
    extract_dir = os.path.join(base_dir, "extracted")

    # 1. Download with Progress Bar
    if not os.path.exists(tar_path):
        logging.info("Downloading LibriSpeech directly from OpenSLR (6.3 GB)...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total file size from the headers
        total_size = int(response.headers.get('content-length', 0))

        # Set up the tqdm progress bar
        with open(tar_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    else:
        logging.info("Tar file already exists, skipping download.")

    # 2. Extract
    if not os.path.exists(extract_dir):
        logging.info("Extracting the archive (this will also take a few minutes)...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    else:
        logging.info("Archive already extracted, skipping extraction.")

    return extract_dir


def process_to_parquet(extract_dir, base_dir):
    logging.info("Converting raw FLAC files to memory-safe Parquet chunks...")
    output_file = os.path.join(base_dir, "train_clean_100.parquet")

    # The actual data is nested inside LibriSpeech/train-clean-100
    data_path = os.path.join(extract_dir, "LibriSpeech", "train-clean-100")

    batch_size = 200  # Reduced for Docker memory limits; 1000 caused OOM
    batch = []
    writer = None
    total_processed = 0

    # Walk through the directory structure: Speaker -> Chapter -> [audio files + transcripts]
    for root, dirs, files in os.walk(data_path):
        transcripts = {}
        # First, find the transcript file in this directory (usually .txt)
        for f in files:
            if f.endswith(".txt"):
                with open(os.path.join(root, f), 'r') as txt_file:
                    for line in txt_file:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

        # Next, process all FLAC audio files in this directory
        for f in files:
            if f.endswith(".flac"):
                file_id = f.replace(".flac", "")
                if file_id in transcripts:
                    audio_path = os.path.join(root, f)

                    # Read the audio using soundfile (which actually works on your Mac)
                    audio_data, sample_rate = sf.read(audio_path)

                    parts = file_id.split('-')
                    record = {
                        "id": file_id,
                        "text": transcripts[file_id],
                        "speaker_id": parts[0],
                        "chapter_id": parts[1],
                        "audio_array": audio_data.tolist(),
                        "sampling_rate": sample_rate
                    }
                    batch.append(record)

                    if len(batch) >= batch_size:
                        df = pd.DataFrame(batch)
                        table = pa.Table.from_pandas(df)
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema)
                        writer.write_table(table)
                        total_processed += len(batch)
                        logging.info(f"Saved {total_processed} audio records...")
                        batch = []

    if batch:
        df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        total_processed += len(batch)

    if writer:
        writer.close()

    logging.info(f"SUCCESS! {total_processed} audio records securely saved to Parquet.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    base = os.path.join(project_root, "data", "raw", "librispeech")
    extracted_path = download_and_extract_librispeech(base_dir=base)
    process_to_parquet(extracted_path, base_dir=base)