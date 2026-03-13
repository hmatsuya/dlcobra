"""Entry point for exp016: uses FocalModel instead of the base Model."""
from dlshogi.ptl import CustomLightningCLI, DataModule
from dlshogi.experiments.exp016_focal_loss.ptl_focal import FocalModel


def main():
    CustomLightningCLI(FocalModel, DataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
