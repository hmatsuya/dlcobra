"""Entry point that uses HcpeDataModule instead of the default DataModule."""
from dlshogi.experiments.exp024_deep512_hao_data.data_module import HcpeDataModule
from dlshogi.ptl import CustomLightningCLI, Model


def main():
    CustomLightningCLI(
        Model, HcpeDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
