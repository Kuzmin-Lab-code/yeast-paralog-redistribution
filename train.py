import hydra
from modules.tools.types import *
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='config', config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    return


if __name__ == '__main__':
    main()
