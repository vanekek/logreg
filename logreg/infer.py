import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from mnist import load_test_data
from model import LogReg
from train_test import test

@hydra.main(config_path="conf", config_name="test_config", version_base="1.3")
def main(cfg: DictConfig):

    model_params = OmegaConf.to_container(cfg["model"])
    model = LogReg(**model_params)
    model.load_state_dict(torch.load(cfg["test"]["save_path"]))
    model.eval()

    test_loader = load_test_data(cfg["test"]["batch_size"])

    # Test the model
    test(model, test_loader, cfg["model"]["input_size"], cfg["test"]["pred_path"])


if __name__ == "__main__":
    main()
