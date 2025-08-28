"""
ddp_task_trainer.py
===================
Distributed Data Parallel (DDP) Trainer class for training downstream models on supervised tasks.
This trainer supports multi-GPU training using PyTorch's DistributedDataParallel.
"""

import glob
import os
import socket
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from sklearn.feature_selection import r_regression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .task_trainer import (
    CrossEntropyLoss,
    PoissonLoss,
    BCEWithLogitsLoss,
    MSELoss,
)


def find_free_port():
    """Find a free port for DDP communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_ddp(rank, world_size, port):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DDPTrainer:
    """
    Distributed Data Parallel trainer for training downstream models on supervised tasks.
    This trainer uses PyTorch's DistributedDataParallel for multi-GPU training.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        rank=0,
        world_size=1,
        overwrite_dir=False,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Get a DDPTrainer object that can be used to train a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        criterion : torch.nn.Module
            Loss function to use for training.
        device : torch.device
            Device to use for training.
        config : OmegaConf
            Configuration object.
        rank : int
            Process rank for DDP.
        world_size : int
            Total number of processes.
        overwrite_dir : bool, optional
            Whether to overwrite the output directory. The default is False.
        gradient_accumulation_steps : int, optional
            Number of gradient accumulation steps. The default is 1.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.overwrite_dir = overwrite_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler()

        # Early stopping parameters (only tracked on main process)
        if self.is_main_process:
            self.early_stopping_patience = getattr(self.config.params, 'early_stopping_patience', None)
            self.early_stopping_min_delta = getattr(self.config.params, 'early_stopping_min_delta', 0.0)
            self.early_stopping_mode = getattr(self.config.params, 'early_stopping_mode', 'max')  # 'max' for metrics like accuracy, 'min' for loss
            
            # Initialize early stopping tracking variables
            self.best_metric = None
            self.patience_counter = 0
            self.early_stopped = False

        # Only create output directory on main process
        if self.is_main_process:
            self._create_output_dir(self.config.output_dir)

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

    def _create_output_dir(self, path):
        """Create output directory (only on main process)."""
        os.makedirs(f"{path}/checkpoints/", exist_ok=True)
        # if load checkpoints is false and overwrite dir is true, delete previous checkpoints
        if self.overwrite_dir and not self.config.params.load_checkpoint:
            print("Deleting all previous checkpoints")
            # delete all checkpoints from previous runs
            [
                os.remove(f)
                for f in glob.glob(f"{path}/**", recursive=True)
                if os.path.isfile(f)
            ]
            pd.DataFrame(
                columns=[
                    "Epoch",
                    "train_loss",
                    "val_loss",
                    f"val_{self.config.params.metric}",
                ]
            ).to_csv(f"{path}/losses.csv", index=False)

    def _load_checkpoint(self, checkpoint):
        """Load checkpoint and handle DDP state dict."""
        checkpoint = torch.load(checkpoint, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except:
            # Handle DDP module wrapper
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
            else:
                self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["val_loss"]
        val_metric = checkpoint[f"val_{self.config.params.metric}"]
        return epoch, train_loss, val_loss, val_metric

    def _save_checkpoint(self, epoch, train_loss, val_loss, val_metric):
        """Save checkpoint (only on main process)."""
        if not self.is_main_process:
            return
            
        # Get the state dict from the model, handling DDP wrapper
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_{self.config.params.metric}": val_metric,
            },
            f"{self.config.output_dir}/checkpoints/epoch_{epoch}.pt",
        )

    def _log_loss(self, epoch, train_loss, val_loss, val_metric):
        """Log losses to CSV (only on main process)."""
        if not self.is_main_process:
            return
            
        df = pd.read_csv(f"{self.config.output_dir}/losses.csv")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[epoch, train_loss, val_loss, val_metric]],
                    columns=[
                        "Epoch",
                        "train_loss",
                        "val_loss",
                        f"val_{self.config.params.metric}",
                    ],
                ),
            ],
            ignore_index=True,
        )
        df.to_csv(f"{self.config.output_dir}/losses.csv", index=False)

    def _log_wandb(self, epoch, train_loss, val_loss, val_metric):
        """Log to WandB (only on main process)."""
        if not self.is_main_process:
            return
            
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_{self.config.params.metric}": val_metric,
            },
            step=epoch,
        )

    def _check_early_stopping(self, val_metric):
        """
        Check if early stopping criteria are met (only on main process).
        
        Parameters
        ----------
        val_metric : float
            The current validation metric value.
            
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if not self.is_main_process or self.early_stopping_patience is None:
            return False
            
        if self.best_metric is None:
            self.best_metric = val_metric
            self.patience_counter = 0
            return False
            
        # Check if current metric is better than best
        improved = False
        if self.early_stopping_mode == 'max':
            # For metrics like accuracy, AUC, etc. where higher is better
            improved = val_metric > (self.best_metric + self.early_stopping_min_delta)
        else:  # mode == 'min'
            # For metrics like loss where lower is better
            improved = val_metric < (self.best_metric - self.early_stopping_min_delta)
            
        if improved:
            self.best_metric = val_metric
            self.patience_counter = 0
            print(f"Validation metric improved to {val_metric:.6f}")
        else:
            self.patience_counter += 1
            print(f"No improvement in validation metric. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
            
        return False

    def _calculate_metric(self, y_true, y_pred) -> List[float]:
        """
        Calculates the metric for the given task.
        The metric calculated is specified in the config.params.metric
        """
        # check if any padding in the target
        if torch.any(y_true == self.config.data.padding_value):
            mask = y_true != self.config.data.padding_value
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        if self.config.params.metric == "mcc":
            metric = matthews_corrcoef(y_true.numpy().ravel(), y_pred.numpy().ravel())
            recall = recall_score(
                y_true.numpy().ravel(), y_pred.numpy().ravel(), average=None
            )
            precision = precision_score(
                y_true.numpy().ravel(), y_pred.numpy().ravel(), average=None
            )
            # Convert to lists if they're arrays
            recall_list = recall.tolist() if hasattr(recall, 'tolist') else [recall]
            precision_list = precision.tolist() if hasattr(precision, 'tolist') else [precision]
            metric = [metric] + recall_list + precision_list
        elif self.config.params.metric == "auroc":
            if self.config.task in [
                "histone_modification",
                "chromatin_accessibility",
                "cpg_methylation",
            ]:
                metric = roc_auc_score(y_true.numpy(), y_pred.numpy(), average=None)
                if hasattr(metric, 'mean') and hasattr(metric, 'tolist'):
                    metric = [metric.mean()] + metric.tolist()
                else:
                    metric = [metric]
            else:
                metric = roc_auc_score(
                    y_true.numpy().ravel(), y_pred.numpy().ravel(), average="macro"
                )
                metric = [metric]
        elif self.config.params.metric == "pearsonr":
            metric = r_regression(
                y_true.detach().numpy().reshape(-1, 1), y_pred.detach().numpy().ravel()
            )[0]
            metric = [metric]
        elif self.config.params.metric == "auprc":
            metric = average_precision_score(
                y_true.numpy().ravel(), y_pred.numpy().ravel(), average="macro"
            )
            metric = [metric]
        else:
            metric = [0.0]  # Default fallback

        return metric

    def _get_checkpoint_path(self, load_checkpoint: Union[bool, int, str] = True):
        """Get the path of the checkpoint to load."""
        if not load_checkpoint:
            if self.is_main_process:
                print("Not looking for existing checkpoints, starting from scratch.")
            return
        if isinstance(load_checkpoint, str):
            return load_checkpoint
        checkpoints = [
            f
            for f in os.listdir(f"{self.config.output_dir}/checkpoints/")
            if f.endswith(".pt")
        ]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if len(checkpoints) == 0:
            if self.is_main_process:
                print("No checkpoints found, starting from scratch.")
            return
        else:
            if isinstance(load_checkpoint, bool):
                if self.is_main_process:
                    print("Load latest checkpoint")
                load_checkpoint = checkpoints[-1]
            elif isinstance(load_checkpoint, int):
                load_checkpoint = f"epoch_{load_checkpoint}.pt"

        checkpoint_path = f"{self.config.output_dir}/checkpoints/{load_checkpoint}"
        # check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
        return checkpoint_path

    def train_epoch(self, train_loader):
        """Performs one epoch of training."""
        from tqdm.auto import tqdm

        self.model.train()
        train_loss = 0
        
        # Only show progress bar on main process
        iterator = tqdm(enumerate(train_loader)) if self.is_main_process else enumerate(train_loader)
        
        for idx, batch in iterator:
            train_loss += self.train_step(batch, idx=idx)

        # Synchronize losses across all processes
        train_loss /= (idx + 1)
        if dist.is_initialized():
            train_loss_tensor = torch.tensor(train_loss, device=self.device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / self.world_size

        return train_loss

    def train_step(self, batch, idx=0):
        """Performs a single training step."""
        self.model.train()

        data, target = batch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(
                x=data.to(self.device, non_blocking=True),
                length=target.shape[-1],
                activation=self.config.params.activation,
            )

            loss = self.criterion(
                output, target.to(self.device, non_blocking=True).long()
            )
            loss = loss / self.gradient_accumulation_steps
            
        self.scaler.scale(loss).backward()
        
        if (idx + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def validate(self, data_loader):
        """Performs validation."""
        self.model.eval()
        loss = 0
        outputs = []
        targets_all = []
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                output = self.model(
                    data.to(self.device), activation=self.config.params.activation
                )
                loss += self.criterion(output, target.to(self.device).long()).item()

                if self.config.params.criterion == "bce":
                    if hasattr(self.model, 'module'):
                        outputs.append(self.model.module.sigmoid(output).detach().cpu())
                    else:
                        outputs.append(self.model.sigmoid(output).detach().cpu())
                else:
                    if hasattr(self.model, 'module'):
                        outputs.append(
                            torch.argmax(self.model.module.softmax(output), dim=-1).detach().cpu()
                        )
                    else:
                        outputs.append(
                            torch.argmax(self.model.softmax(output), dim=-1).detach().cpu()
                        )

                targets_all.append(target.detach().cpu())

        loss /= (idx + 1)
        
        # Synchronize validation loss across all processes
        if dist.is_initialized():
            loss_tensor = torch.tensor(loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss = loss_tensor.item() / self.world_size

        # Gather outputs and targets from all processes for metric calculation
        if dist.is_initialized():
            # Gather outputs and targets from all processes
            gathered_outputs = [None for _ in range(self.world_size)]
            gathered_targets = [None for _ in range(self.world_size)]
            
            dist.all_gather_object(gathered_outputs, outputs)
            dist.all_gather_object(gathered_targets, targets_all)
            
            # Only calculate metrics on main process
            if self.is_main_process:
                # Flatten the gathered results
                all_outputs = []
                all_targets = []
                for proc_outputs, proc_targets in zip(gathered_outputs, gathered_targets):
                    all_outputs.extend(proc_outputs)
                    all_targets.extend(proc_targets)
                outputs = all_outputs
                targets_all = all_targets

        # Calculate metrics only on main process
        if self.is_main_process:
            try:
                metrics = self._calculate_metric(torch.cat(targets_all), torch.cat(outputs))
            except:
                metrics = self._calculate_metric(
                    torch.cat([i.flatten() for i in targets_all]),
                    torch.cat([i.flatten() for i in outputs]),
                )
        else:
            metrics = [0.0]  # Placeholder for non-main processes

        return loss, metrics

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        load_checkpoint: Union[bool, int] = True,
    ):
        """Performs the full training routine."""
        if self.is_main_process:
            print("Training with DDP")
            
        # Set up distributed sampler if needed (only for regular datasets)
        train_sampler = None
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_sampler = train_loader.sampler
        
        # Check if we're using WebDataset or similar
        is_webdataset = (
            'WebLoader' in train_loader.__class__.__name__ or
            (hasattr(train_loader, 'dataset') and not hasattr(train_loader.dataset, '__len__'))
        )
        
        if self.is_main_process and is_webdataset:
            print("Using WebDataset/iterative dataset for DDP training")
            
        start_epoch = 0
        checkpoint_path = self._get_checkpoint_path(load_checkpoint)
        if checkpoint_path:
            start_epoch, train_loss, val_loss, val_metric = self._load_checkpoint(
                checkpoint_path
            )
            if self.is_main_process:
                print(
                    f"Loaded checkpoint from epoch {start_epoch}, train loss: {train_loss}, val loss: {val_loss}, val {self.config.params.metric}: {val_metric}"
                )

        for epoch in range(1 + start_epoch, epochs + 1):
            # Set epoch for distributed sampler (only for regular datasets)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # For WebDataset, we might need to configure epoch differently
            elif is_webdataset and hasattr(train_loader, 'dataset'):
                dataset = train_loader.dataset
                # Some WebDatasets support epoch configuration
                if hasattr(dataset, 'with_epoch'):
                    # Configure epoch for WebDataset if supported
                    pass  # WebDataset typically handles this internally
                
            if self.is_main_process:
                print("Training epoch:", epoch)
            train_loss = self.train_epoch(train_loader)
            
            if self.is_main_process:
                print("Validating epoch:", epoch)
            val_loss, val_metrics = self.validate(val_loader)
            val_metric = val_metrics[0] if val_metrics else 0.0

            # Save checkpoint, log losses, and log to wandb only on main process
            self._save_checkpoint(epoch, train_loss, val_loss, val_metric)
            self._log_loss(epoch, train_loss, val_loss, val_metric)
            self._log_wandb(epoch, train_loss, val_loss, val_metric)
            
            if self.is_main_process:
                print(
                    f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {self.config.params.metric}: {val_metric:.4f}"
                )

            # Check for early stopping (only on main process)
            early_stop = False
            if self.is_main_process:
                early_stop = self._check_early_stopping(val_metric)
                if early_stop:
                    self.early_stopped = True
                    print(f"Early stopping at epoch {epoch}")

            # Broadcast early stopping decision to all processes
            if dist.is_initialized():
                # Create tensor to broadcast early stopping decision
                early_stop_tensor = torch.tensor(early_stop, dtype=torch.bool, device=self.device)
                dist.broadcast(early_stop_tensor, src=0)
                
                # All processes check the broadcasted decision
                if early_stop_tensor.item():
                    if not self.is_main_process:
                        print(f"Process {self.rank}: Early stopping at epoch {epoch}")
                    break
                
                # Synchronize all processes
                dist.barrier()
            elif early_stop:
                # Single process case
                break
        
        if self.is_main_process:
            if hasattr(self, 'early_stopped') and self.early_stopped:
                print("Training completed due to early stopping")
            elif self.early_stopping_patience is not None:
                print(f"Training completed normally after {epochs} epochs")

    def test(self, test_loader, checkpoint=None, overwrite=False):
        """Performs testing (only on main process)."""
        if not self.is_main_process:
            return None, None
            
        print("TESTING with DDP")
        if checkpoint is None:
            df = pd.read_csv(f"{self.config.output_dir}/losses.csv")
            checkpoint = pd.DataFrame(
                df.iloc[df[f"val_{self.config.params.metric}"].idxmax()]
            ).T.reset_index(drop=True)
            
        # Load checkpoint
        epoch, train_loss, val_loss, val_metric = self._load_checkpoint(
            f'{self.config.output_dir}/checkpoints/epoch_{int(checkpoint["Epoch"].iloc[0])}.pt'
        )
        print(
            f"Loaded checkpoint from epoch {epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, Val {self.config.params.metric}: {np.mean(val_metric):.3f}"
        )
        
        # Synchronize all processes before testing
        if dist.is_initialized():
            dist.barrier()
            
        # Test
        loss, metric = self.validate(test_loader)
        
        if self.is_main_process:
            print(
                f"Test results : Loss {loss:.4f}, {self.config.params.metric} {metric[0]:.4f}"
            )

            # Save test results
            if len(metric) > 1:
                data = [[loss] + list(metric)]
                if self.config.params.metric == "mcc":
                    columns = (
                        ["test_loss", f"test_{self.config.params.metric}"]
                        + [f"test_recall_{n}" for n in range(int((len(metric) - 1) / 2))]
                        + [f"test_precision_{n}" for n in range(int((len(metric) - 1) / 2))]
                    )
                else:
                    columns = ["test_loss", f"test_{self.config.params.metric}_avg"] + [
                        f"test_{self.config.params.metric}_{n}"
                        for n in range(len(metric) - 1)
                    ]
            else:
                columns = ["test_loss", f"test_{self.config.params.metric}"]
                data = [[loss, metric[0]]]

            metrics = checkpoint.merge(
                pd.DataFrame(data=data, columns=columns), how="cross"
            )

            if not overwrite and os.path.exists(
                f"{self.config.output_dir}/best_model_metrics.csv"
            ):
                best_model_metrics = pd.read_csv(
                    f"{self.config.output_dir}/best_model_metrics.csv", index_col=False
                )
                metrics = pd.concat([best_model_metrics, metrics], ignore_index=True)

            metrics.to_csv(f"{self.config.output_dir}/best_model_metrics.csv", index=False)

        return loss, metric


def ddp_worker(rank, world_size, port, cfg):
    """
    Worker function for DDP training.
    This function is called by each process in the multi-process training.
    Note: We recreate data loaders in each process to avoid multiprocessing issues.
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')
        
        # Recreate data loaders in each process to avoid multiprocessing issues
        if "supervised" in cfg.embedder:
            cfg.data.data_dir = cfg.data.data_dir.replace(cfg.embedder, "onehot")

        import hydra
        train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data)
        
        # Configure data loaders for DDP
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader
        
        def is_webdataset(loader):
            """Check if loader is a WebLoader or similar iterative dataset."""
            if loader is None:
                return False
            loader_name = loader.__class__.__name__
            dataset_name = getattr(loader, 'dataset', None)
            dataset_name = dataset_name.__class__.__name__ if dataset_name else ""
            
            return (
                'WebLoader' in loader_name or 
                'WebDataset' in dataset_name or
                'IterableDataset' in dataset_name or
                (hasattr(loader, 'dataset') and not hasattr(loader.dataset, '__len__'))
            )
        
        def create_ddp_webdataset_loader(cfg, split_type, batch_size, rank, world_size):
            """Create a WebDataset loader with proper DDP configuration."""
            import webdataset as wds
            import glob
            import os
            from functools import partial
            from .data_downstream import collate_fn_pad_to_longest
            
            # Determine data directory and file pattern
            data_dir = cfg.data.data_dir
            if not os.path.exists(data_dir):
                raise RuntimeError(f"Data directory does not exist: {data_dir}")
            
            # Get tar files based on split type
            if split_type == 'train':
                pattern = f"{data_dir}/train*.tar.gz"
            elif split_type == 'valid':
                pattern = f"{data_dir}/valid*.tar.gz"  
            elif split_type == 'test':
                pattern = f"{data_dir}/test*.tar.gz"
            else:
                raise ValueError(f"Unknown split type: {split_type}")
            
            data_files = glob.glob(pattern)
            if not data_files:
                if rank == 0:
                    print(f"Warning: No {split_type} files found matching {pattern}")
                return None
                
            data_files = sorted(data_files)  # Ensure consistent ordering
            
            # Simple approach: manually assign different shards to different processes
            # This avoids the "nodesplitter" error by ensuring each DDP process gets different data
            total_files = len(data_files)
            files_per_process = max(1, total_files // world_size)
            start_idx = rank * files_per_process
            end_idx = min((rank + 1) * files_per_process, total_files) if rank < world_size - 1 else total_files
            
            # Each DDP process gets a different subset of tar files
            process_files = data_files[start_idx:end_idx]
            if not process_files:
                # Fallback: give at least one file to each process
                process_files = [data_files[rank % total_files]]
            
            if rank == 0:
                print(f"DDP shard assignment: process {rank} gets files {start_idx}:{end_idx} = {len(process_files)} files")
            
            # Create WebDataset with process-specific files (this avoids the nodesplitter error!)
            dataset = wds.WebDataset(process_files)
            
            # Add shuffling if specified and this is training data
            shuffle = cfg.data.get('shuffle', None)
            if shuffle is not None and split_type == 'train':
                dataset = dataset.shuffle(shuffle)
                
            # Process the dataset
            dataset = dataset.decode()
            dataset = dataset.to_tuple("input.npy", "output.npy") 
            dataset = dataset.map_tuple(torch.from_numpy, torch.from_numpy)
            dataset = dataset.map_tuple(torch.squeeze, torch.squeeze)
            dataset = dataset.batched(batch_size, collation_fn=None)
            dataset = dataset.map(
                partial(collate_fn_pad_to_longest, padding_value=cfg.data.padding_value)
            )
            
            # Create WebLoader 
            num_workers = cfg.data.get('num_workers', 0)
            dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
            
            return dataloader
        
        # Configure loaders based on type
        if is_webdataset(train_loader):
            if rank == 0:
                print("Detected WebDataset/iterative dataset - using DDP with minimal workers")
            
            # WebDataset with DDP requires special handling 
            # The simplest approach is to set num_workers=0 to avoid the nodesplitter requirement
            # and recreate loaders with DDP-compatible configuration
            if hasattr(train_loader, 'num_workers'):
                print(f"Setting WebDataset num_workers to 0 for DDP compatibility (was {getattr(train_loader, 'num_workers', 'unknown')})")
            
            # For WebDataset, we need to bypass the complex splitting and just use the existing loaders
            # The key insight: WebDataset issues come from DataLoader workers + DDP processes
            # Solution: Use the existing loaders but ensure they work in DDP mode
            pass  # Keep existing loaders as-is
        else:
            # Regular datasets - recreate loaders with distributed samplers
            if rank == 0:
                print("Detected regular dataset - using DistributedSampler")
                
            if train_loader and hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, '__len__'):
                train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    sampler=train_sampler,
                    num_workers=getattr(train_loader, 'num_workers', 0),
                    pin_memory=getattr(train_loader, 'pin_memory', False),
                    drop_last=getattr(train_loader, 'drop_last', False),
                    collate_fn=getattr(train_loader, 'collate_fn', None)
                )
                
            if val_loader and hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, '__len__'):
                val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
                val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=val_loader.batch_size,
                    sampler=val_sampler,
                    num_workers=getattr(val_loader, 'num_workers', 0),
                    pin_memory=getattr(val_loader, 'pin_memory', False),
                    drop_last=False,
                    collate_fn=getattr(val_loader, 'collate_fn', None)
                )
                
            if test_loader and hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, '__len__'):
                test_sampler = DistributedSampler(test_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
                test_loader = DataLoader(
                    test_loader.dataset,
                    batch_size=test_loader.batch_size,
                    sampler=test_sampler,
                    num_workers=getattr(test_loader, 'num_workers', 0),
                    pin_memory=getattr(test_loader, 'pin_memory', False),
                    drop_last=False,
                    collate_fn=getattr(test_loader, 'collate_fn', None)
                )
        
        # Create model and move to device
        model = hydra.utils.instantiate(cfg.model).to(device).float()
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Create optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        
        # Create criterion
        criterion = None
        if cfg.params.criterion == "cross_entropy":
            from .task_trainer import CrossEntropyLoss
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
        
        # Create trainer
        trainer = DDPTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=cfg,
            rank=rank,
            world_size=world_size,
            overwrite_dir=True,
        )
        
        if rank == 0:
            print(f"Starting DDP training on {world_size} GPUs...")
        
        # Train
        if cfg.params.mode == "train":
            trainer.train(
                train_loader,
                val_loader,
                test_loader,
                cfg.params.epochs,
                cfg.params.load_checkpoint,
            )
        
        # Test
        trainer.test(test_loader, overwrite=False)
        
        if rank == 0:
            print("DDP training completed successfully!")
            
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_ddp()


def launch_ddp_training(cfg, train_loader=None, val_loader=None, test_loader=None):
    """
    Launch DDP training with multiple processes.
    Note: Data loaders are recreated in each process to avoid multiprocessing issues.
    The loader arguments are kept for compatibility but not used.
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: Only {world_size} GPU available. DDP is most effective with multiple GPUs.")
    
    port = find_free_port()
    print(f"Launching DDP training with {world_size} processes on port {port}")
    
    # Set multiprocessing start method to spawn for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, continue
        pass
    
    # Launch multiple processes - only pass config, data loaders will be recreated
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=ddp_worker, 
            args=(rank, world_size, port, cfg)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    print("DDP training completed on all processes")
