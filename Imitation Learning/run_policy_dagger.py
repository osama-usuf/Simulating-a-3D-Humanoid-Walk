"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3.6 run_policy_dagger.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 25
Modified from run_policy_expert.py from Jonathan Ho (hoj@openai.com)
"""

import pickle, tensorflow as tf, tf_util, numpy as np
import gym
import load_policy, policy as pol

def main():
    import argparse
    #Parse Terminal Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('Loading and Building expert policy')

    data=None
    try:
        with open('rollouts/'+args.envname+'-'+str(args.num_rollouts)+'-dagger.pkl', 'rb') as f:
            data = pickle.load(f)
            print('dagger data')
    except:
        with open('rollouts/'+args.envname+'-'+str(args.num_rollouts)+'-expert.pkl', 'rb') as f:
            data = pickle.load(f)
            print('expert data')

    n_in, n_out = data['observations'].shape[1], data['actions'].shape[2]    
    policy_expert = load_policy.load_policy(args.expert_policy_file)
    x, _ = pol.placeholder_inputs(None, n_in, n_out, pol.batch_size)
    policy_fn = pol.inference(x, n_in, n_out, pol.n_h1, pol.n_h2, pol.n_h3)
    obs_org=data['observations']
    act_org=data['actions']
    saver = tf.train.Saver()
    print('Loaded and Built')

    with tf.Session():
        tf_util.initialize()
        saver.restore(tf_util.get_session(), "trained-DAgger/"+args.envname)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        actions_expert = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            totalr = 0
            steps = 0
            while True:
                action_expert = policy_expert(obs[None,:])
                action = np.array(tf_util.get_session().run([policy_fn],feed_dict={x:obs[None,:]}))
                observations.append(obs)
                actions.append(action)
                actions_expert.append(action_expert)
                obs, r, _, _ = env.step(action)
                totalr += r
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

        dagger_data = {'observations': np.concatenate((obs_org,np.array(observations))),
                       'actions': np.concatenate((act_org,np.array(actions_expert)))}
        # save dagger policy observations
        with open('rollouts/'+args.envname+'-'+str(args.num_rollouts)+'-dagger.pkl', 'wb') as f:
            pickle.dump(dagger_data, f)

if __name__ == '__main__':
    main()
