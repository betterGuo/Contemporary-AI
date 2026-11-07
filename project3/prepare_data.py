"""Convert the official CIFAR-10 Python archive to NHWC NumPy arrays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ROOT = Path(__file__).resolve().parent
PREFIX = "cifar-10-batches-py/"


def digest(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing official archive instead of downloading")
    args = parser.parse_args()
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    archive = args.archive or directory / "cifar-10-python.tar.gz"
    if not args.archive and not archive.exists():
        pending = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, pending)
        if digest(pending, "md5") != MD5:
            raise ValueError("Incomplete or unexpected download")
        pending.replace(archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("Archive does not match the official CIFAR-10 Python distribution")
    with tarfile.open(archive, "r:gz") as tar:
        def load_member(name):
            with tar.extractfile(PREFIX + name) as member:
                return pickle.load(io.BytesIO(member.read()), encoding="bytes")

        def as_split(names):
            batches = [load_member(name) for name in names]
            x = np.concatenate([b[b"data"] for b in batches]).reshape(-1, 3, 32, 32)
            y = np.asarray([v for b in batches for v in b[b"labels"]], dtype=np.int64)
            assert x.dtype == np.uint8 and y.shape == (len(x),) and len(x) == len(names) * 10000
            assert np.array_equal(np.bincount(y, minlength=10), np.full(10, len(x) // 10))
            return np.ascontiguousarray(x.transpose(0, 2, 3, 1)), y

        assert [s.decode() for s in load_member("batches.meta")[b"label_names"]] == CLASSES
        x_train, y_train = as_split([f"data_batch_{i}" for i in range(1, 6)])
        x_test, y_test = as_split(["test_batch"])
    dest = directory / "cifar10.npz"
    pending = dest.with_suffix(".partial.npz")
    np.savez(pending, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    pending.replace(dest)
    metadata = {
        "source_url": URL, "archive_name": archive.name,
        "acquisition": "verified existing archive" if args.archive else "official archive in data directory",
        "archive_md5": MD5, "archive_sha256": digest(archive, "sha256"),
        "npz_sha256": digest(dest, "sha256"), "image_layout": "NHWC RGB uint8",
        "label_dtype": "int64", "class_names": CLASSES,
        "train_shape": list(x_train.shape), "test_shape": list(x_test.shape),
        "train_counts": np.bincount(y_train).tolist(), "test_counts": np.bincount(y_test).tolist(),
        "order": "data_batch_1 to data_batch_5 and test_batch in their original order",
    }
    (directory / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Ready: {dest.name} ({len(y_train)} training, {len(y_test)} test).")


if __name__ == "__main__":
    main()
