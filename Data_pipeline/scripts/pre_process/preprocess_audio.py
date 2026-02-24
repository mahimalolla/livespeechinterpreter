import os
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import librosa


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AudioPreprocessor:
    """

   This is the classic preprocessor
   We take a audio in hand set the minimum decible to 20 and trim that part.
   Next we resample it to 16Khz
   Normalize the amplitutude between -1 to 1
   Then we convert back to db
   We then store it in a paraquet file which is nothing but a Column storage files Unlike, csv which is row based.

    """

    def __init__(self, target_sr: int = 16000, n_mels: int = 80):
        self.target_sr = target_sr
        self.n_mels = n_mels
        # Standard parameters for 16kHz audio: 25ms window, 10ms step
        self.n_fft = 400
        self.hop_length = 160

    def process_audio_array(self, audio_list: list, orig_sr: int) -> list:
        """Applies the full DSP pipeline to a single audio array."""

        # Convert the stored Python list back to a numpy array for librosa
        audio = np.array(audio_list, dtype=np.float32)

        # 1. Trim Silence (removes quiet parts at the beginning and end)
        # top_db=20 means any sound 20 decibels quieter than the reference is cut
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # 2. Resample to 16kHz
        if orig_sr != self.target_sr:
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=self.target_sr)

        # 3. Normalize Amplitude (scales all values to be between -1.0 and 1.0)
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        # 4. Extract Log-Mel Spectrograms (80 bins)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert power (amplitude squared) to decibel (log) scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert the 2D numpy array to a nested Python list so Parquet can store it
        return log_mel_spec.tolist()

    def process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes a chunk of the dataframe."""
        logging.info(f"Applying DSP pipeline to {len(df)} audio records...")

        # Apply the processing function to each row
        # We use a lambda to pass both the audio array and its original sample rate
        df['log_mel_spec'] = df.apply(
            lambda row: self.process_audio_array(row['audio_array'], row['sampling_rate']),
            axis=1
        )

        # Drop the raw audio array to save massive amounts of disk space
        df = df.drop(columns=['audio_array'])

        # Update the sampling rate column to reflect the new uniform rate
        df['sampling_rate'] = self.target_sr

        return df


def main():
    # 1. Find exactly where this script is: (.../scripts/preprocess/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go UP two levels to the main LiveInterpretPython folder
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # 3. Now build the paths safely from the main project folder
    input_file = os.path.join(project_root, "data", "raw", "librispeech", "train_clean_100.parquet")
    output_dir = os.path.join(project_root, "data", "processed")
    output_file = os.path.join(output_dir, "asr_processed.parquet")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        logging.error(f"Input file not found. I am looking exactly here: {input_file}")
        return



    preprocessor = AudioPreprocessor(target_sr=16000, n_mels=80)

    # Read metadata to enable safe chunking
    parquet_file = pq.ParquetFile(input_file)
    total_row_groups = parquet_file.num_row_groups

    writer = None
    total_processed = 0

    logging.info(f"Starting ASR preprocessing. Total chunks: {total_row_groups}")

    for i in range(total_row_groups):
        logging.info(f"--- Processing Chunk {i + 1} of {total_row_groups} ---")

        # Load a single chunk
        chunk_df = parquet_file.read_row_group(i).to_pandas()

        # Process the chunk
        processed_chunk = preprocessor.process_chunk(chunk_df)

        # Write to disk
        table = pa.Table.from_pandas(processed_chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)
        total_processed += len(processed_chunk)

    if writer:
        writer.close()

    logging.info(
        f"SUCCESS! {total_processed} audio files converted to log-mel spectrograms and saved to {output_file}.")


if __name__ == "__main__":
    main()