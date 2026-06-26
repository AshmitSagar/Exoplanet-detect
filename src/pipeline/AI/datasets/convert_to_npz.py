from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_DIR = Path("data/kepler/tfrecord")
OUTPUT_DIR = Path("data/ai_ready/npz")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DESCRIPTION = {
    "global_view": tf.io.FixedLenFeature([2001], tf.float32),
    "local_view": tf.io.FixedLenFeature([201], tf.float32),

    "av_training_set": tf.io.FixedLenFeature([], tf.string),

    "kepid": tf.io.FixedLenFeature([], tf.int64),
    "tce_period": tf.io.FixedLenFeature([], tf.float32),
    "tce_plnt_num": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(example_proto):
    return tf.io.parse_single_example(
        example_proto,
        FEATURE_DESCRIPTION
    )

files = sorted(DATA_DIR.glob("train-*"))

dataset = tf.data.TFRecordDataset(
    [str(f) for f in files]
)

dataset = dataset.map(parse_example)

print(f"Loaded {len(files)} TFRecord shards")

MAX_SAMPLES = 100      # Change to None later

count = 0

for sample in tqdm(dataset):

    label = sample["av_training_set"].numpy().decode()

    kepid = int(sample["kepid"].numpy())

    planet_number = int(sample["tce_plnt_num"].numpy())

    period = float(sample["tce_period"].numpy())

    global_view = sample["global_view"].numpy()

    local_view = sample["local_view"].numpy()

    filename = OUTPUT_DIR / f"{kepid}_{planet_number}.npz"

    np.savez_compressed(
        filename,

        global_view=global_view,

        local_view=local_view,

        label=label,

        kepid=kepid,

        period=period,

        planet_number=planet_number,
    )

    count += 1

    if MAX_SAMPLES and count >= MAX_SAMPLES:
        break

print(f"\nSaved {count} NPZ files.")

files = sorted(OUTPUT_DIR.glob("*.npz"))

print(files[0])

sample = np.load(files[0])

print(sample.files)