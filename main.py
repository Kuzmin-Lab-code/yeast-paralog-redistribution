import hydra
from tools.typing import *


@hydra.main(config_path="config/default.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    return


if __name__ == '__main__':
    main()
