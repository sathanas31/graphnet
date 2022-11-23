"""Example of training Model for direction reconstruction."""


import os
import argparse
from typing import Any, Dict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import StandardModel
from graphnet.models.coarsening import (
    DOMCoarsening,
    CustomDOMCoarsening,
    DOMAndTimeWindowCoarsening,
)
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    ZenithReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
)
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
)
from graphnet.utilities.logging import get_logger

logger = get_logger()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Parse args
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--infile", type=str, help="Input database")
parser.add_argument("-o", "--outfile", type=str, help="Output path")
parser.add_argument(
    "-wb",
    "--wandb_path",
    type=str,
    help="Enable W&B by giving the dest path",
    default=None,
)
parser.add_argument(
    "-node_pooling",
    type=str,
    help="Choose which node pooling mode",
    default=None,
    choices=["dom_coarsening", "custom_dom_coarsening", "dom_twd_coarsening"],
)
# TODO default value for time window?
# parser.add_argument(
#     "-time_window",
#     type=float,
#     help="Cluster pulses on the same DOM within time window",
#     default=None,
# )
parser.add_argument(
    "-pulses", type=str, help="Choose pulsemap", default="TWSRTOfflinePulses"
)
parser.add_argument(
    "-batch_size", type=int, help="Choose batch size", default=512
)
parser.add_argument(
    "-num_workers", type=int, help="Choose number of workers", default=10
)
parser.add_argument(
    "-num_devices", type=int, help="Choose how many gpus", default=0
)
parser.add_argument(
    "-epochs", type=int, help="Choose how many epochs", default=1
)

args = parser.parse_args()

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Make sure W&B output directory exists
if args.wandb_path is not None:
    WANDB_DIR = "./wandb/"
    os.makedirs(WANDB_DIR, exist_ok=True)

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project="example-script",
        entity="graphnet-team",
        save_dir=WANDB_DIR,
        log_model=True,
    )


def train(config: Dict[str, Any]) -> None:
    """Train model with configuration given by `config`."""
    # Log configuration to W&B
    if args.wandb_path is not None:
        wandb_logger.experiment.config.update(config)

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    train_selection = train_selection[0:50000]

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        config["db"],
        train_selection,
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    if config["node_pooling"] == "dom_coarsening":
        coarsening = DOMCoarsening()
    elif config["node_pooling"] == "custom_dom_coarsening":
        coarsening = CustomDOMCoarsening()
    elif config["node_pooling"] == "dom_twd_coarsening":
        coarsening = DOMAndTimeWindowCoarsening()
    else:
        coarsening = None
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    if config["target"] == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )
    elif config["target"] == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )

    model = StandardModel(
        detector=detector,
        coarsening=coarsening,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["n_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger if args.wandb_path is not None else True,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [config["target"] + "_pred"],
        additional_attributes=[config["target"], "event_no"],
    )

    save_results(
        config["db"], config["run_name"], results, config["archive"], model
    )


def main() -> None:
    """Run example."""
    for target in ["zenith", "azimuth"]:
        archive = args.outfile
        run_name = "dynedge_{}_example".format(target)

        # Configuration
        config = {
            "db": args.infile,
            "pulsemap": args.pulses,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "accelerator": "gpu",
            "devices": [args.num_devices],
            "target": target,
            "n_epochs": args.epochs,
            "patience": 5,
            "archive": archive,
            "run_name": run_name,
            "max_events": 50000,
            "node_pooling": args.node_pooling,
        }
        train(config)


if __name__ == "__main__":
    main()
