import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

import pandas as pd
from graphnet.components.loss_functions import VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn.dynedge import DynEdge, DynEdge_V2
from graphnet.models.coarsening import DOMCoarsening, CustomDOMCoarsening
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    ZenithReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
)
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
)
from graphnet.utilities.logging import get_logger

logger = get_logger()

# Parse args for convenience
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
parser.add_argument("-batch_size", type=int, help="Choose batch size", default=512)
parser.add_argument(
    "-num_workers", type=int, help="Choose number of workers", default=10
)
parser.add_argument("-num_devices", type=int, help="Choose how many gpus", default=1)
parser.add_argument("-epochs", type=int, help="Choose how many epochs", default=1)

args = parser.parse_args()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

if args.wandb_path is not None:
    # Make sure W&B output directory exists
    WANDB_DIR = args.wandb_path + "/wandb/"
    os.makedirs(WANDB_DIR, exist_ok=True)

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project="example-script",
        entity="graphnet-team",
        save_dir=WANDB_DIR,
        log_model=True,
    )


def save_results(db, tag, results, archive):
    db_name = db.split("/")[-1].split(".")[0]
    path = archive + "/" + db_name + "/" + tag
    os.makedirs(path, exist_ok=True)
    results.to_csv(path + "/results.csv")
    print("Results saved at: \n %s" % path)


def train(config):

    if args.wandb_path is not None:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Common variables
    selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    selection = selection[0 : config["max_events"]]

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    (training_dataloader, validation_dataloader,) = make_train_validation_dataloader(
        config["db"],
        selection,
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

    if config["node_pooling"] == "custom_dom_coarsening":
        gnn = DynEdge_V2(
            nb_inputs=detector.nb_outputs, node_pooling=CustomDOMCoarsening()
        )
    if config["node_pooling"] == "dom_coarsening":
        gnn = DynEdge_V2(nb_inputs=detector.nb_outputs, node_pooling=DOMCoarsening())
    else:
        gnn = DynEdge_V2(nb_inputs=detector.nb_outputs)

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

    model = Model(
        detector=detector,
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
        # resume_from_checkpoint =  '/home/iwsatlas1/oersoe/phd/northern_tracks/checkpoints/epoch=39-step=83040.ckpt'
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer=trainer,
        model=model,
        dataloader=validation_dataloader,
        prediction_columns=[
            config["target"] + "_pred",
            config["target"] + "_kappa",
        ],
        additional_attributes=[config["target"], "event_no"],
    )

    save_results(config["db"], config["run_name"], results, config["archive"])


# Main function definition
def main():
    for target in ["zenith", "azimuth"]:
        archive = args.outfile
        run_name = f"dynedgev2_{node_pooling}_nt_{target}_{pulsemap}"

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
            "node_pooling": args.node_pooling,
            "max_events": 50000,
        }
        train(config)


# Main function call
if __name__ == "__main__":
    main()
