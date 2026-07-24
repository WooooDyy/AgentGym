#!/bin/bash
#SBATCH --job-name=lazy-baselines
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/lazy_pd/paper_baselines/slurm-%j.out

set -euo pipefail
cd /home/da839/self_distill_lazy/AgentGym
source /home/da839/anaconda3/etc/profile.d/conda.sh
conda activate AgentGym
mkdir -p outputs/lazy_pd/paper_baselines
for experiment_seed in 17 29 43 61 89; do
  python -m experiments.lazy_pd.looped_agentic \
    --seed "$experiment_seed" \
    --output-dir outputs/lazy_pd/paper_baselines \
    --methods frozen eager eager_confident mean_teacher eager_replay lazy \
    --patterns 24 \
    --waves 6 \
    --cycles 12 \
    --horizon 8 \
    --attempts-per-step 3 \
    --tasks-per-pattern 24 \
    --eval-tasks-per-pattern 32 \
    --hidden-dim 192 \
    --batch-size 256 \
    --warmup-epochs 30 \
    --update-epochs 2 \
    --seed-patterns 24 \
    --seed-examples-per-pattern 24 \
    --verifier-weight 0.40 \
    --verifier-noise 0.08 \
    --confidence-threshold 0.70 \
    --ema-decay 0.99 \
    --replay-ratio 1.0
done
