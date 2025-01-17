"""Example of building and running PISA Pipeline with SQLite inputs."""

import torch
from typing import Dict, List, Union

from graphnet.data.pipeline import InSQLitePipeline
from graphnet.data.constants import TRUTH, FEATURES

torch.multiprocessing.set_sharing_strategy("file_system")


def get_output_column_names(target: str) -> List[str]:
    """Return the relevant set of output colummn names for `target`."""
    if target in ["azimuth", "zenith"]:
        output_column_names = [target + "_pred", target + "_kappa"]
    if target in ["track", "neutrino", "energy"]:
        output_column_names = [target + "_pred"]
    if target == "XYZ":
        output_column_names = [
            "position_x_pred",
            "position_y_pred",
            "position_z_pred",
        ]
    return output_column_names


def build_module_dictionary(targets: List[str]) -> Dict[str, Dict]:
    """Build a dictionary of output paths and column names for `targets`."""
    module_dict: Dict[str, Dict] = {}
    for target in targets:
        module_dict[target] = {}
        module_dict[target][
            "path"
        ] = f"/home/iwsatlas1/oersoe/phd/oscillations/models/final/dynedge_oscillation_final_{target}.pth"
        module_dict[target]["output_column_names"] = get_output_column_names(
            target
        )
    return module_dict


def main() -> None:
    """Run example."""
    # Configuration
    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86
    pulsemap = "SRTTWOfflinePulsesDC"
    batch_size = 1024 * 4
    num_workers = 40
    device = "cuda:1"
    targets = ["track", "energy", "zenith"]
    pipeline_name = "pipeline_oscillation"
    database = "/mnt/scratch/rasmus_orsoe/databases/oscillations/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db"

    # Remove `interaction_time` if it exists
    try:
        del truth[truth.index("interaction_time")]
    except ValueError:
        # not found in list
        pass

    # Build Pipeline
    pipeline = InSQLitePipeline(
        module_dict=build_module_dictionary(targets),
        features=features,
        truth=truth,
        device=device,
        batch_size=batch_size,
        n_workers=num_workers,
        pipeline_name=pipeline_name,
    )

    # Run Pipeline
    pipeline(database, pulsemap)


if __name__ == "__main__":
    main()
