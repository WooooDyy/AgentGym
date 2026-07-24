#!/bin/bash
#SBATCH --job-name=lazy-pd
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/lazy_pd/slurm-%j.out

set -euo pipefail
cd /home/da839/self_distill_lazy/AgentGym
source /home/da839/anaconda3/etc/profile.d/conda.sh
conda activate AgentGym
python -m experiments.lazy_pd.lazy_pd \
  --methods frozen eager_global delayed_global immediate_local \
            lazy_local_no_base lazy_local
