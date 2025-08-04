#!/bin/bash
#SBATCH --output=logs/bridge/slurm-%j.out  # Save stdout to logs folder with job ID
#SBATCH --nodes=1             # Keep it to 1 node unless scaling out
#SBATCH --ntasks=1            # Single training process
#SBATCH --mem-per-cpu=16G      # Request memory per CPU
#SBATCH --gpus=rtx_4090:1     # Use all 4 RTX 4090 GPUs
#SBATCH --cpus-per-task=4    # Allocate more CPUs for data loading
#SBATCH --time=2:00:00       # Set a reasonable training time limit


source /cluster/project/cvg/students/hailuo/miniconda3/etc/profile.d/conda.sh
conda activate grounded_sam_2

module load stack/2024-06 cuda/12.1.1 eth_proxy

cd /cluster/home/hailuo/project/shape-of-motion/preproc/Grounded-SAM-2
# python grounded_sam2_tracking_demo.py
python grounded_sam2_tracking_demo_with_continuous_id.py