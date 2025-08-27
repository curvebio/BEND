#!/bin/bash
# filepath: /home/jiajiexiao/BEND/run_embeddings_simple.sh

# Configuration
MODELS=("resnetlm" "nt_transformer_human_ref" "dnabert2")
TASKS=("gene_finding" "enhancer_annotation" "variant_effects" "histone_modification" "chromatin_accessibility" "cpg_methylation")
GPUS=(0 1 2 3)
MAX_JOBS=4
LOG_DIR="./logs/embeddings"

# Create log directory
mkdir -p "$LOG_DIR"

# Array to track running jobs
running_jobs=()

# Function to wait for any job to complete
wait_for_slot() {
    while [ ${#running_jobs[@]} -ge $MAX_JOBS ]; do
        new_jobs=()
        for job in "${running_jobs[@]}"; do
            if kill -0 $job 2>/dev/null; then
                new_jobs+=($job)
            else
                echo "Job $job completed"
            fi
        done
        running_jobs=("${new_jobs[@]}")
        
        if [ ${#running_jobs[@]} -ge $MAX_JOBS ]; then
            sleep 5
        fi
    done
}

# Function to start a job
start_job() {
    local model=$1
    local task=$2
    local gpu=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${model}_${task}_gpu${gpu}_${timestamp}.log"
    
    echo "Starting: $model + $task on GPU $gpu"
    echo "Log: $log_file"
    
    # Start job in background
    CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_embeddings.py \
        model=$model \
        task=$task \
        hydra.mode=RUN \
        > "$log_file" 2>&1 &
    
    local pid=$!
    running_jobs+=($pid)
    echo "Started PID: $pid"
    echo ""
}

# Main execution
echo "Starting embedding computation..."
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Max concurrent jobs: $MAX_JOBS"
echo "=========================================="

job_count=0
gpu_index=0

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        # Wait for available slot
        wait_for_slot
        
        # Get GPU (cycle through available GPUs)
        gpu=${GPUS[$gpu_index]}
        gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
        
        # Start job
        start_job "$model" "$task" "$gpu"
        
        ((job_count++))
        echo "Progress: $job_count/$(( ${#MODELS[@]} * ${#TASKS[@]} )) jobs submitted"
        
        # Small delay to avoid overwhelming
        sleep 2
    done
done

# Wait for all remaining jobs
echo "All jobs submitted. Waiting for completion..."
while [ ${#running_jobs[@]} -gt 0 ]; do
    new_jobs=()
    for job in "${running_jobs[@]}"; do
        if kill -0 $job 2>/dev/null; then
            new_jobs+=($job)
        else
            echo "Job $job completed"
        fi
    done
    running_jobs=("${new_jobs[@]}")
    
    if [ ${#running_jobs[@]} -gt 0 ]; then
        echo "Still running: ${#running_jobs[@]} jobs"
        sleep 10
    fi
done

echo "All jobs completed!"
echo "Check logs in: $LOG_DIR"

# Optional: Show summary
echo ""
echo "Job Summary:"
echo "============"
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        # Find the most recent log for this combination
        log_file=$(ls -t "$LOG_DIR"/${model}_${task}_*.log 2>/dev/null | head -1)
        if [ -n "$log_file" ]; then
            if grep -q -i "error\|failed\|exception" "$log_file"; then
                echo "❌ $model + $task: FAILED"
            else
                echo "✅ $model + $task: SUCCESS"
            fi
        else
            echo "❓ $model + $task: NO LOG FOUND"
        fi
    done
done