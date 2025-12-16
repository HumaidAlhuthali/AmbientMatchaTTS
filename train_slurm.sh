#!/bin/bash
# SLURM Batch Script for Training Matcha-TTS
# Usage: ./train_matcha.sh <dataset_name> <partition> <num_gpus> <time> [t_max]
# Example: ./train_matcha.sh ljspeech mit_preemptable 1 48:00:00 0.5

# Check if all required parameters are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Error: Required parameters are missing."
    echo "Usage: $0 <dataset_name> <partition> <num_gpus> <time> [t_max]"
    echo ""
    echo "Parameters:"
    echo "  dataset_name: ljspeech, multispeaker (vctk), hi-fi_en-US_female, combined (ljspeech + vctk)"
    echo "  partition:    mit_preemptable or mit_normal_gpu"
    echo "  num_gpus:     Number of GPUs to request (e.g., 1, 2, 4)"
    echo "  time:         Job time limit (e.g., 48:00:00, 24:00:00)"
    echo "  t_max:        (optional) t_max for non-LJSpeech speakers (default: 0.5)"
    echo ""
    echo "Example: $0 ljspeech mit_preemptable 1 48:00:00"
    echo "Example: $0 combined mit_preemptable 2 48:00:00 0.7"
    exit 1
fi

DATASET=$1
PARTITION=$2
NUM_GPUS=$3
TIME=$4
T_MAX=${5:-0.5}  # Default to 0.5 if not provided
JOB_NAME="matcha_${DATASET}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${HOME}/Matcha-TTS/slurm_logs"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Create the SLURM batch script
BATCH_SCRIPT=$(mktemp /tmp/matcha_train_XXXXXX.sh)

cat > ${BATCH_SCRIPT} << 'EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --output=LOG_DIR_PLACEHOLDER/matcha_%j.out
#SBATCH --error=LOG_DIR_PLACEHOLDER/matcha_%j.err
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:NUM_GPUS_PLACEHOLDER
#SBATCH --mem=32G
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --requeue                          # Requeue if preempted

# Load required modules
module load miniforge/23.11.0-0
module load cuda/12.4.0
module load gcc/12.2.0

# Activate conda environment
conda activate matcha-tts

# Navigate to Matcha-TTS directory
cd ${HOME}/Matcha-TTS

# Print job information
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Dataset: DATASET_PLACEHOLDER"
echo "Partition: PARTITION_PLACEHOLDER"
echo "GPUs: NUM_GPUS_PLACEHOLDER"
echo "Time Limit: TIME_PLACEHOLDER"
echo "t_max (non-LJSpeech): T_MAX_PLACEHOLDER"
echo "Start Time: $(date)"
echo "=========================================="

# Run training
python matcha/train.py experiment=DATASET_PLACEHOLDER run_name="DATASET_PLACEHOLDER_tmax_T_MAX_PLACEHOLDER" data.batch_size=512 model.out_size=256 data.num_workers=2 trainer.max_epochs=500 model.non_target_speaker_t_max=T_MAX_PLACEHOLDER trainer=ddp

# Print completion time
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
EOF

# Replace placeholders with actual values
sed -i "s|JOB_NAME_PLACEHOLDER|${JOB_NAME}|g" ${BATCH_SCRIPT}
sed -i "s|LOG_DIR_PLACEHOLDER|${LOG_DIR}|g" ${BATCH_SCRIPT}
sed -i "s|DATASET_PLACEHOLDER|${DATASET}|g" ${BATCH_SCRIPT}
sed -i "s|PARTITION_PLACEHOLDER|${PARTITION}|g" ${BATCH_SCRIPT}
sed -i "s|NUM_GPUS_PLACEHOLDER|${NUM_GPUS}|g" ${BATCH_SCRIPT}
sed -i "s|TIME_PLACEHOLDER|${TIME}|g" ${BATCH_SCRIPT}
sed -i "s|T_MAX_PLACEHOLDER|${T_MAX}|g" ${BATCH_SCRIPT}

# Submit the job
echo "Submitting Matcha-TTS training job:"
echo "  Dataset:   ${DATASET}"
echo "  Partition: ${PARTITION}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Time:      ${TIME}"
echo "  t_max:     ${T_MAX}"
echo "Log files will be saved to: ${LOG_DIR}"
sbatch ${BATCH_SCRIPT}

# Clean up
rm ${BATCH_SCRIPT}

echo "Job submitted successfully!"
