"""Example of converting I3-files to SQLite and Parquet."""

import logging
import os
import argparse

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
    I3GenericExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter

logger = get_logger(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indir", type=str, help="Input i3 files dir")
parser.add_argument("-o", "--outdir", type=str, help="Output databases dir")
parser.add_argument(
    "-convert",
    type=str,
    help="How to convert data, either SQLite or Parquet",
    choices=["sqlite", "parquet"],
    required=True,
)
parser.add_argument(
    "-array",
    type=str,
    help="Choose between IceCube86 and Upgrade/Gen2",
    choices=["Icecube", "Upgrade", "Gen2"],
    required=True,
)
parser.add_argument(
    "-ic_keys",
    type=str,
    help="Choose keys for extraction when IceCube86 array",
    nargs="*",
)
parser.add_argument(
    "-upgrade_keys",
    type=str,
    help="Choose pulsemaps for extraction when Upgrade/Gen2 array",
    nargs="*",
)

args = parser.parse_args()

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_icecube86(backend: str) -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = [args.indir]
    outdir = args.outdir

    converter: DataConverter = CONVERTER_CLASS[backend](
        [
            I3GenericExtractor(keys=args.ic_keys),
            I3TruthExtractor(),
        ],
        outdir,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


def main_icecube_upgrade(backend: str) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = [args.indir]
    outdir = args.outdir
    workers = 1

    extractors = [I3TruthExtractor()]
    if args.array == "Upgrade":
        extractors.append(I3RetroExtractor())
    for pulses in args.upgrade_keys:
        extractors.append(I3FeatureExtractorIceCubeUpgrade(pulses))

    converter: DataConverter = CONVERTER_CLASS[backend](
        extractors,
        outdir,
        workers=workers,
        # nb_files_to_batch=10,
        # sequential_batch_pattern="temp_{:03d}",
        # input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        icetray_verbose=1,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


if __name__ == "__main__":
    backend = args.convert
    if args.array == "Icecube":
        main_icecube86(backend)
    elif args.array == "Upgrade" or args.array == "Gen2":
        main_icecube_upgrade(backend)
