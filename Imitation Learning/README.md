# Final Project - Reinforcement Learning V/S Imitation Learning
## Computational Intelligence - Spring '19 - Habib University 

###### Instructions:
- To generate required data files, run the run_expert.bash script via terminal.
- Once the files are generated, the corresponding scripts for training and running corresponding policies can be executed.

###### Dependencies/Versions:
- Python 3.6.8
- OpenAI GYM 0.12.1
- Mujoco200 Linux
- Mujoco_py 2.0.2.2
- Tensorflow 1.13.1

**Tested in Linux Mint 18.3 (Sylvia)**

###### Acknowledgements:
Credits to open-source coursework for **CS294-112: Deep Reinforcement Learning** - , by University of California Berkeley for the following files:

##### File Descriptions:

* expert/Humanoid-v2.pkl - Expert policy file for the Humanoid environment.
* load_policy.py - The basic structure of loading a policy into the environment.
* run_expert.py - File for loading the provided expert policy into the environment.
* tf_util.py - TensorFlow utilities for added functionalities.

The following files have been built up on top of this structure.

* policy.py - A simple 3-layer feed-forward neural network, acts as the policy.
* train_policy_regular.py - For training the policy via the non-aggregated expert data.
* train_policy_dagger.py - For training the policy via the aggregated expert data.
* run_policy_regular.py - For evaluating the trained non-aggregated policy.
* run_policy_dagger.py - For evaluating the trained aggregated policy.

All the above files can be executed using the provided scripts.

###### Author(s):
- Osama Yousuf - BS(CS) - oy02945
- Reeba Aslam - BS(CS) - ra02528


