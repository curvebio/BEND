#!/bin/bash

# Advanced BEND Embedding Runner with Resource Monitoring
# Features: GPU memory monitoring, job recovery, progress tracking

set -e

# Script configuration
SCRIPT_NAME="$(basename "$0")"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORK_DIR}/logs/embedding_$(date +%Y%m%d_%H%M%S)"
STATE_FILE="${LOG_DIR}/job_state.txt"
PROGRESS_FILE="${LOG_DIR}/progress.txt"

# Create directories
mkdir -p "$LOG_DIR"

# System detection
NUM_CPUS=$(nproc)
NUM_GPUS=$(nvidia-smi -L | wc -l)
AVAILABLE_GPUS=($(seq 0 $((NUM_GPUS - 1))))

# Resource management
JOBS_PER_GPU=5  # Adjust based on model size
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))
GPU_MEMORY_THRESHOLD=70  # Percentage

# Configuration from your embed.yaml
MODELS=("dnabert2-bs-seq" "hyenadna-bs-seq")
TASKS=("gene_finding" "enhancer_annotation" "histone_modification" "chromatin_accessibility" "cpg_methylation")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_DIR}/runner.log"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_DIR}/runner.log"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_DIR}/runner.log"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_DIR}/runner.log"; }

# Function to check GPU memory usage
check_gpu_memory() {
    local gpu_id=$1
    # Get used and total memory in MB
    local mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$mem_info" ]; then
        local used=$(echo "$mem_info" | cut -d',' -f1 | tr -d ' ')
        local total=$(echo "$mem_info" | cut -d',' -f2 | tr -d ' ')
        # Calculate percentage
        if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$total" =~ ^[0-9]+$ ]] && [ "$total" -gt 0 ]; then
            local percent=$(( (used * 100) / total ))
            echo $percent
        else
            echo 0
        fi
    else
        echo 0
    fi
}

# Function to find best GPU (lowest memory usage)
find_best_gpu() {
    local best_gpu=0
    local min_usage=100
    
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        local usage=$(check_gpu_memory $gpu)
        if [[ "$usage" =~ ^[0-9]+$ ]] && [ "$usage" -lt "$min_usage" ]; then
            min_usage=$usage
            best_gpu=$gpu
        fi
    done
    
    echo $best_gpu
}

# Function to wait for GPU availability
wait_for_gpu() {
    while true; do
        for gpu in "${AVAILABLE_GPUS[@]}"; do
            local usage=$(check_gpu_memory $gpu)
            if [[ "$usage" =~ ^[0-9]+$ ]] && [ "$usage" -lt "$GPU_MEMORY_THRESHOLD" ]; then
                echo $gpu
                return
            fi
        done
        log_info "All GPUs busy (>${GPU_MEMORY_THRESHOLD}% memory). Waiting..."
        sleep 10
    done
}

# Function to save job state
save_job_state() {
    local model=$1
    local task=$2
    local status=$3
    local gpu=$4
    local start_time=$5
    local end_time=${6:-$(date)}
    
    echo "$model,$task,$status,$gpu,$start_time,$end_time" >> "$STATE_FILE"
}

# Function to update progress
update_progress() {
    local completed=$1
    local failed=$2
    local total=$3
    local percent=$(( (completed + failed) * 100 / total ))
    
    echo "Progress: $((completed + failed))/$total ($percent%) | Success: $completed | Failed: $failed" > "$PROGRESS_FILE"
    cat "$PROGRESS_FILE"
}

# Function to run embedding job with monitoring
run_embedding_job() {
    local model=$1
    local task=$2
    local job_id="${model}_${task}"
    local start_time=$(date)
    local gpu_id
    
    # Find best GPU or wait for availability
    if [ "$3" = "auto" ]; then
        gpu_id=$(wait_for_gpu)
        log_info "Auto-selected GPU $gpu_id for $job_id"
    else
        gpu_id=$3
    fi
    
    local log_file="${LOG_DIR}/${job_id}_gpu${gpu_id}.log"
    
    # Save job start state
    save_job_state "$model" "$task" "RUNNING" "$gpu_id" "$start_time"
    
    log_info "Starting $job_id on GPU $gpu_id (Memory: $(check_gpu_memory $gpu_id)%)"
    
    # Run the job
    local exit_code=0
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/precompute_embeddings.py \
        model="$model" \
        task="$task" \
        device_id=$gpu_id \
        hydra.mode=RUN \
        > "$log_file" 2>&1 || exit_code=$?
    
    local end_time=$(date)
    
    if [ $exit_code -eq 0 ]; then
        save_job_state "$model" "$task" "COMPLETED" "$gpu_id" "$start_time" "$end_time"
        log_success "Completed $job_id on GPU $gpu_id"
    else
        save_job_state "$model" "$task" "FAILED" "$gpu_id" "$start_time" "$end_time"
        log_error "Failed $job_id on GPU $gpu_id (exit code: $exit_code)"
        log_error "Check log: $log_file"
    fi
    
    return $exit_code
}

# Function to monitor system resources
monitor_resources() {
    local monitoring=true
    
    while $monitoring; do
        # Check if main process is still running
        if ! pgrep -f "$SCRIPT_NAME" > /dev/null 2>&1; then
            break
        fi
        
        # Log system status every 30 seconds
        {
            echo "=== System Status at $(date) ==="
            echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
            echo "Active Jobs: $(jobs -r | wc -l)"
            echo ""
        } >> "${LOG_DIR}/system_monitor.log"
        
        sleep 30
    done
}

# Function to handle cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any remaining background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Final report
    generate_report
    
    log_info "Cleanup completed"
}

# Function to generate final report
generate_report() {
    local report_file="${LOG_DIR}/final_report.txt"
    
    {
        echo "BEND Embedding Precomputation Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo ""
        
        if [ -f "$STATE_FILE" ]; then
            echo "Job Summary:"
            echo "Model,Task,Status,GPU,Start Time,End Time" 
            cat "$STATE_FILE"
            echo ""
            
            local total=$(wc -l < "$STATE_FILE")
            local completed=$(grep -c "COMPLETED" "$STATE_FILE" || echo "0")
            local failed=$(grep -c "FAILED" "$STATE_FILE" || echo "0")
            local running=$(grep -c "RUNNING" "$STATE_FILE" || echo "0")
            
            echo "Statistics:"
            echo "  Total Jobs: $total"
            echo "  Completed: $completed"
            echo "  Failed: $failed"  
            echo "  Still Running: $running"
            echo "  Success Rate: $(( completed * 100 / (completed + failed) ))%" 2>/dev/null || echo "  Success Rate: N/A"
        else
            echo "No job state file found"
        fi
        
        echo ""
        echo "Log Directory: $LOG_DIR"
        echo "System Info:"
        echo "  CPUs: $NUM_CPUS"
        echo "  GPUs: $NUM_GPUS"
        echo "  Max Parallel Jobs: $MAX_PARALLEL"
        
    } | tee "$report_file"
    
    log_info "Report saved to: $report_file"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Advanced BEND embedding precomputation runner with resource monitoring.

OPTIONS:
    -h, --help              Show this help
    -j, --max-jobs N        Maximum parallel jobs (default: auto)
    -g, --gpu-jobs N        Jobs per GPU (default: 2)  
    -t, --threshold N       GPU memory threshold % (default: 70)
    --models "model1,model2" Comma-separated model list
    --tasks "task1,task2"   Comma-separated task list
    --dry-run              Show execution plan without running
    --monitor              Enable continuous resource monitoring
    --resume               Resume from previous run (if state file exists)

EXAMPLES:
    $0                      # Run with defaults
    $0 -j 4 -t 80          # Max 4 jobs, 80% GPU threshold
    $0 --models "dnabert2"  # Run only dnabert2 model
    $0 --dry-run           # Preview jobs
    $0 --monitor           # Enable resource monitoring

EOF
}

# Parse command line arguments
ENABLE_MONITORING=false
DRY_RUN=false
RESUME=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -j|--max-jobs)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -g|--gpu-jobs)
            JOBS_PER_GPU="$2"
            MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))
            shift 2
            ;;
        -t|--threshold)
            GPU_MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --tasks)
            IFS=',' read -ra TASKS <<< "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --monitor)
            ENABLE_MONITORING=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution function
main() {
    # Setup signal handlers
    trap cleanup EXIT INT TERM
    
    # Print configuration
    log_info "BEND Advanced Embedding Runner"
    log_info "Log Directory: $LOG_DIR"
    log_info "System: $NUM_CPUS CPUs, $NUM_GPUS GPUs"
    log_info "Config: Max $MAX_PARALLEL jobs, $JOBS_PER_GPU per GPU, ${GPU_MEMORY_THRESHOLD}% threshold"
    log_info "Models: ${MODELS[*]}"
    log_info "Tasks: ${TASKS[*]}"
    
    local total_jobs=$((${#MODELS[@]} * ${#TASKS[@]}))
    log_info "Total jobs: $total_jobs"
    
    # Start resource monitoring if enabled
    if $ENABLE_MONITORING; then
        log_info "Starting resource monitoring"
        monitor_resources &
        MONITOR_PID=$!
    fi
    
    # Dry run mode
    if $DRY_RUN; then
        log_info "DRY RUN MODE - Jobs that would be executed:"
        local job_num=1
        for model in "${MODELS[@]}"; do
            for task in "${TASKS[@]}"; do
                echo "  Job $job_num: $model + $task"
                job_num=$((job_num + 1))
            done
        done
        return 0
    fi
    
    # Initialize counters
    local completed=0
    local failed=0
    local job_pids=()
    
    # Launch jobs
    log_info "Starting job execution..."
    
    for model in "${MODELS[@]}"; do
        for task in "${TASKS[@]}"; do
            # Wait for available slot
            while [ ${#job_pids[@]} -ge $MAX_PARALLEL ]; do
                # Check for completed jobs
                local new_pids=()
                for pid in "${job_pids[@]}"; do
                    if kill -0 $pid 2>/dev/null; then
                        new_pids+=($pid)
                    else
                        wait $pid
                        if [ $? -eq 0 ]; then
                            completed=$((completed + 1))
                        else
                            failed=$((failed + 1))
                        fi
                        update_progress $completed $failed $total_jobs
                    fi
                done
                job_pids=("${new_pids[@]}")
                
                if [ ${#job_pids[@]} -ge $MAX_PARALLEL ]; then
                    sleep 2
                fi
            done
            
            # Launch new job
            run_embedding_job "$model" "$task" "auto" &
            local job_pid=$!
            job_pids+=($job_pid)
            
            sleep 1  # Brief delay to prevent system overload
        done
    done
    
    # Wait for remaining jobs
    log_info "Waiting for remaining jobs to complete..."
    for pid in "${job_pids[@]}"; do
        wait $pid
        if [ $? -eq 0 ]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
        update_progress $completed $failed $total_jobs
    done
    
    # Stop monitoring
    if $ENABLE_MONITORING && [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    # Final results
    log_info "All jobs completed!"
    log_info "Results: $completed successful, $failed failed"
    
    if [ $failed -eq 0 ]; then
        log_success "All embedding jobs completed successfully! 🎉"
        return 0
    else
        log_error "Some jobs failed. Check logs for details."
        return 1
    fi
}

# Validate environment
if [ ! -f "scripts/precompute_embeddings.py" ]; then
    log_error "Python script not found. Run from BEND project root."
    exit 1
fi

# Check for virtual environment
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    log_warn "No virtual environment detected"
    log_warn "Make sure you have activated the correct Python environment"
fi

# Run main function
main "$@"
