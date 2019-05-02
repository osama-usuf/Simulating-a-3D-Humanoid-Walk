#!/bin/bash
set -eux

python3.6 run_policy_regular.py '' Humanoid-v2 --render --num_rollouts 25

