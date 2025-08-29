"""
train_on_task.py
----------------
Train a model on a downstream task.
Supports both regular DataParallel and Distributed Data Parallel (DDP) training.
"""

import os
import sys
import multiprocessing as mp

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from bend.models.downstream import CustomDataParallel
from bend.utils.task_trainer import (
    BaseTrainer,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MSELoss,
    PoissonLoss,
)
from bend.utils.ddp_task_trainer import launch_ddp_training
from bend.utils.seed_utils import set_seed

# Set multiprocessing start method early to avoid conflicts
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set
    pass

os.environ["WDS_VERBOSE_CACHE"] = "1"


# load config
@hydra.main(
    config_path=f"../conf/supervised_tasks/", config_name=None, version_base=None
)  #
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    # Set random seed early for reproducibility
    random_seed = getattr(cfg.params, "random_seed", 0)
    set_seed(random_seed)
    
    # Check if we should use DDP training (default for single GPU)
    use_ddp = getattr(cfg.params, "use_ddp", False) and torch.cuda.device_count() > 1

    # Only initialize wandb on main process for DDP
    if (
        not use_ddp
        or not hasattr(cfg.params, "use_ddp")
        or cfg.params.get("use_ddp", False)
    ):
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # mkdir output_dir
        os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
        print("output_dir", cfg.output_dir)
        # init wandb
        run = wandb.init(
            **cfg.wandb,
            dir=cfg.output_dir,
            config=wandb_config,
        )

        OmegaConf.save(
            cfg, f"{cfg.output_dir}/config.yaml"
        )  # save the config to the experiment dir

    # For DDP training, we need to prepare data loaders first
    if "supervised" in cfg.embedder:
        cfg.data.data_dir = cfg.data.data_dir.replace(cfg.embedder, "onehot")

    print("Data loaders instantiating")
    train_loader, val_loader, test_loader = hydra.utils.instantiate(
        cfg.data
    )  # instantiate dataloaders

    for loader in [train_loader, val_loader, test_loader]:
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
            # Regular PyTorch DataLoader
            print(
                f" - {loader.dataset.__class__.__name__} with {len(loader.dataset)} samples"
            )
        elif hasattr(loader, "dataset"):
            # WebDataset or other dataset without __len__
            dataset = loader.dataset
            if hasattr(dataset, "length"):
                print(f" - {dataset.__class__.__name__} with {dataset.length} samples")
            elif hasattr(dataset, "with_length") and hasattr(dataset, "_length"):
                print(f" - {dataset.__class__.__name__} with {dataset._length} samples")
            else:
                print(
                    f" - {dataset.__class__.__name__} (streaming dataset, length unknown)"
                )
        else:
            # WebLoader or other loader type
            print(f" - {loader.__class__.__name__} (streaming loader, length unknown)")

    # Use DDP training if multiple GPUs available and not explicitly disabled
    if use_ddp:
        print(f"Using DDP training with {torch.cuda.device_count()} GPUs")
        launch_ddp_training(cfg, train_loader, val_loader, test_loader)
        return

    # Fall back to single-GPU or DataParallel training
    print("Using single-GPU or DataParallel training")

    # set device with configurable device_id
    if hasattr(cfg.params, "device_id") and cfg.params.device_id is not None:
        if torch.cuda.is_available():
            try:
                device = torch.device(f"cuda:{cfg.params.device_id}")
                # Test if the device is valid by creating a small tensor
                torch.tensor([1.0]).to(device)
                print(f"Using specified GPU device: {device}")
            except (RuntimeError, AssertionError) as e:
                print(
                    f"Warning: Invalid device ID {cfg.params.device_id}, falling back to auto-selection. Error: {e}"
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using fallback device: {device}")
        else:
            device = torch.device("cpu")
            print(
                f"Warning: CUDA not available, using CPU instead of cuda:{cfg.params.device_id}"
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using auto-selected device:", device)

    # instantiate model
    # initialization for supervised models
    if cfg.embedder == "resnet-supervised":
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.update(cfg.supervised_encoder[cfg.embedder])
    if cfg.embedder == "basset-supervised":
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.update(cfg.supervised_encoder[cfg.embedder])
    model = hydra.utils.instantiate(cfg.model).to(device).float()

    # put model on dataparallel
    if torch.cuda.device_count() > 1:
        from bend.models.downstream import CustomDataParallel

        print("Let's use", torch.cuda.device_count(), "GPUs with DataParallel!")
        model = CustomDataParallel(model)
    print(model)

    # instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # define criterion
    print(f"Use {cfg.params.criterion} loss function")
    criterion = None
    if cfg.params.criterion == "cross_entropy":
        criterion = CrossEntropyLoss(
            ignore_index=cfg.data.padding_value,
            weight=(
                torch.tensor(cfg.params.class_weights).to(device)
                if cfg.params.class_weights is not None
                else None
            ),
        )
    elif cfg.params.criterion == "poisson_nll":
        criterion = PoissonLoss()
    elif cfg.params.criterion == "mse":
        criterion = MSELoss()
    elif cfg.params.criterion == "bce":
        criterion = BCEWithLogitsLoss(
            class_weights=(
                torch.tensor(cfg.params.class_weights).to(device)
                if cfg.params.class_weights is not None
                else None
            )
        )

    if criterion is None:
        raise ValueError(f"Unknown criterion: {cfg.params.criterion}")

    # init dataloaders - this was already done above, so we can remove the duplicate

    # Configure Automatic Mixed Precision (AMP)
    amp_dtype = getattr(cfg.params, "amp_dtype", "auto")  # Options: "bf16", "fp16", "none", "auto"
    
    if amp_dtype == "auto":
        # Auto-select best dtype based on hardware
        if torch.cuda.is_bf16_supported():
            amp_dtype = "bf16"
            print("Auto-selected bfloat16 for mixed precision training (A100/H100 optimized)")
        elif torch.cuda.is_available():
            amp_dtype = "fp16"
            print("Auto-selected float16 for mixed precision training")
        else:
            amp_dtype = "none"
            print("CUDA not available, disabling mixed precision training")
    elif amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported on this hardware, falling back to float16")
            amp_dtype = "fp16"
        else:
            print("Using bfloat16 for mixed precision training")
    elif amp_dtype == "fp16":
        print("Using float16 for mixed precision training")
    elif amp_dtype == "none":
        print("Mixed precision training disabled")
    else:
        raise ValueError(f"Invalid amp_dtype: {amp_dtype}. Choose from: 'bf16', 'fp16', 'none', 'auto'")

    # instantiate trainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        overwrite_dir=True,
        amp_dtype=amp_dtype,
    )

    if cfg.params.mode == "train":
        # train
        trainer.train(
            train_loader,
            val_loader,
            test_loader,
            cfg.params.epochs,
            cfg.params.load_checkpoint,
        )

    # test
    trainer.test(test_loader, overwrite=False)


if __name__ == "__main__":
    print("Run experiment")
    run_experiment()
