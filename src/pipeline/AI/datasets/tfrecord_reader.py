from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/kepler/tfrecord")

# AstroNet TFRecord schema
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


def inspect_dataset():

    files = sorted(DATA_DIR.glob("train-*"))

    print(f"Found {len(files)} TFRecord shards")

    dataset = tf.data.TFRecordDataset([str(f) for f in files])

    dataset = dataset.map(parse_example)

    sample = next(iter(dataset))

    print("\n========== SAMPLE ==========\n")

    for key, value in sample.items():

        print(f"{key}")

        print(f"Shape : {value.shape}")

        print(f"Dtype : {value.dtype}")

        if len(value.shape) == 0:
            print(f"Value : {value.numpy()}")

        print()


if __name__ == "__main__":

    inspect_dataset()

