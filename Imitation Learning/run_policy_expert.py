"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3.6 run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    #Parse terminal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    #Load expert policy
    print('Loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('Loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('Rollout',i+1,'/',args.num_rollouts)
            obs = env.reset()
            done = False
            totalr = 0
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, reward, done, _ = env.step(action) #env.step() returns observation, reward, done, info.
                totalr += reward
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('Returns', returns)
        print('Mean return', np.mean(returns))
        print('Std. of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        #Export expert observation data
        with open(os.path.join('rollouts', args.envname + '-' + str(args.num_rollouts) + '-expert.pkl'), 'wb+') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()