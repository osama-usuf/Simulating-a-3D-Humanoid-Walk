#!/bin/bash
set -eux

python3.6 train_policy_dagger.py Humanoid-v2 --num_epochs 200 --num_rollouts 25 --firsttime 0 --step_size 1e-4

