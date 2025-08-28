# Early Stopping Guide

This guide explains how to use the early stopping functionality that has been added to both the `BaseTrainer` and `DDPTrainer` classes in BEND.

## Overview

Early stopping is a regularization technique that prevents overfitting by stopping training when a validation metric stops improving. This implementation monitors the validation metric specified in your configuration and stops training if it doesn't improve for a specified number of epochs (patience).

## Configuration Parameters

Add the following parameters to your configuration file under the `params` section:

```yaml
params:
  # ... other parameters ...
  
  # Early Stopping Configuration
  early_stopping_patience: 10        # Number of epochs to wait for improvement (default: None - disabled)
  early_stopping_min_delta: 0.001    # Minimum change to qualify as improvement (default: 0.0)
  early_stopping_mode: "max"         # "max" for metrics where higher is better, "min" for lower is better (default: "max")
```

### Parameter Details

- **`early_stopping_patience`**: The number of epochs with no improvement after which training will be stopped. If `None` or not specified, early stopping is disabled.
- **`early_stopping_min_delta`**: Minimum change in the monitored metric to qualify as an improvement. For example, a min_delta of 0.001 means the metric must improve by at least 0.001 to reset the patience counter.
- **`early_stopping_mode`**: 
  - `"max"` for metrics where higher values are better (e.g., accuracy, AUC, precision, recall)
  - `"min"` for metrics where lower values are better (e.g., loss)

## Supported Metrics

The early stopping works with any metric specified in your `params.metric` configuration:

- **`auroc`**: Area Under ROC Curve (use mode: "max")
- **`auprc`**: Area Under Precision-Recall Curve (use mode: "max")
- **`mcc`**: Matthews Correlation Coefficient (use mode: "max")
- **`pearsonr`**: Pearson Correlation (use mode: "max")

For custom metrics, ensure you set the appropriate `early_stopping_mode`.

## Example Configurations

### Example 1: Chromatin Accessibility with Early Stopping

```yaml
defaults:
  - datadims : [label_dims,embedding_dims, downstream_downsample]
  - hydra : multirun 
  - supervised_encoder : [resnet-supervised, basset-supervised]
  - _self_

embedder : onehot
task : chromatin_accessibility 
output_dir: ./downstream_tasks/${task}/${embedder}/

model:
  _target_: bend.models.downstream.CNN
  encoder : null
  input_size: ${datadims.${embedder}}
  output_size: ${datadims.${task}}
  hidden_size: 64
  kernel_size: 3
  output_downsample_window: ${datadims.output_downsample_window.${task}}

optimizer : 
  _target_ : torch.optim.AdamW 
  lr : 0.003
  weight_decay: 0.01

data:
  _target_: bend.utils.data_downstream.get_data
  data_dir : ./data/${task}/${embedder}/
  cross_validation : false
  batch_size : 256
  num_workers : 8
  padding_value : -100
  shuffle : 1000

params:
  epochs: 100
  load_checkpoint: false
  mode: train
  gradient_accumulation_steps: 1
  criterion: bce
  class_weights: null
  metric : auroc
  activation : none
  
  # Early stopping configuration
  early_stopping_patience: 15      # Stop if no improvement for 15 epochs
  early_stopping_min_delta: 0.001  # Require at least 0.1% improvement
  early_stopping_mode: "max"       # AUROC: higher is better

wandb:
  mode : disabled 
```

### Example 2: Gene Finding with Strict Early Stopping

```yaml
# ... other configurations ...

params:
  epochs: 50
  metric: mcc
  
  # Strict early stopping - stop if no improvement for 5 epochs
  early_stopping_patience: 5
  early_stopping_min_delta: 0.01   # Require significant improvement
  early_stopping_mode: "max"       # MCC: higher is better
```

### Example 3: Regression Task (if using loss as metric)

```yaml
# ... other configurations ...

params:
  epochs: 100
  metric: pearsonr
  
  # Early stopping based on correlation improvement
  early_stopping_patience: 20      # Be more patient for regression
  early_stopping_min_delta: 0.005  # Small improvements count
  early_stopping_mode: "max"       # Pearson correlation: higher is better
```

## How It Works

1. **Initialization**: When early stopping is enabled, the trainer initializes tracking variables:
   - `best_metric`: Tracks the best validation metric seen so far
   - `patience_counter`: Counts epochs without improvement
   - `early_stopped`: Flag indicating if training stopped early

2. **During Training**: After each epoch's validation:
   - The current metric is compared with the best metric
   - If improved (by at least `min_delta`), the counter resets
   - If not improved, the counter increments
   - If the counter reaches `patience`, training stops

3. **DDP Support**: In distributed training:
   - Only the main process (rank 0) makes early stopping decisions
   - The decision is broadcast to all other processes
   - All processes stop training simultaneously

## Output and Logging

When early stopping is active, you'll see additional console output:

```
Validation metric improved to 0.847291
No improvement in validation metric. Patience: 1/10
No improvement in validation metric. Patience: 2/10
...
No improvement in validation metric. Patience: 10/10
Early stopping triggered after 10 epochs without improvement
Early stopping at epoch 45
```

The final message will indicate whether training completed normally or stopped early:
- `"Training completed due to early stopping"`
- `"Training completed normally after {epochs} epochs"`

## Best Practices

1. **Set appropriate patience**: 
   - 10-20 epochs for most tasks
   - Higher patience (20+) for complex tasks or when using small learning rates
   - Lower patience (5-10) for quick experimentation

2. **Choose min_delta carefully**:
   - 0.0 for any improvement to count
   - 0.001-0.01 for requiring meaningful improvements
   - Higher values (0.01+) for noisy metrics

3. **Monitor your metrics**:
   - Use mode="max" for accuracy, AUC, precision, recall, MCC, correlation
   - Use mode="min" if you ever monitor validation loss directly

4. **Combine with model checkpointing**: The best model checkpoint is still saved regardless of early stopping, so you can always revert to the best performing model.

## Debugging

If early stopping seems too aggressive or too lenient:

1. Check your `early_stopping_mode` matches your metric (max vs min)
2. Adjust `early_stopping_patience` based on your learning curves
3. Modify `early_stopping_min_delta` if metric improvements are very small/large
4. Monitor the training logs to see when improvements are being detected
