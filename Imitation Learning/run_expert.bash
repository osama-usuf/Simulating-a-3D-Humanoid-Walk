#!/bin/bash
set -eux

python3.6 run_policy_expert.py expert/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts=25

