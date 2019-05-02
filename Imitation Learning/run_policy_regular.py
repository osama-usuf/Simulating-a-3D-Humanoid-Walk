"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3.6 run_policy_regular.py '' Humanoid-v2 --render --num_rollouts 25
"""

import pickle, tensorflow as tf, tf_util, numpy as np
import gym
import load_policy, policy as pol

def main():
    import argparse
    #Parse Terminal Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('run_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('Loading and building regular policy')
    
    with open('rollouts/'+args.envname+'-'+str(args.num_rollouts)+'-expert.pkl', 'rb') as f:
        data = pickle.load(f)
        n_in, n_out = data['observations'].shape[1], data['actions'].shape[2]
    
    x, _ = pol.placeholder_inputs(None, n_in, n_out, pol.batch_size)
    policy_fn = pol.inference(x, n_in, n_out, pol.n_h1, pol.n_h2, pol.n_h3)
    saver = tf.train.Saver()
    print('Loaded and Built')

    with tf.Session():
        tf_util.initialize()
        saver.restore(tf_util.get_session(), "trained/"+args.envname)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            totalr = 0
            steps = 0
            while True:
                action = np.array(tf_util.get_session().run([policy_fn],feed_dict={x:obs[None,:]}))
                observations.append(obs)
                actions.append(action)
                obs, r, _, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                    if steps % 100 == 0: 
                        print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break


            returns.append(totalr)

        print('Returns', returns)
        print('Mean return', np.mean(returns))
        print('Std. of return', np.std(returns))

        reg_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        # save regular policy observations
        with open('rollouts/'+args.envname+'-regular.pkl', 'wb+') as f:
            pickle.dump(reg_data, f)

if __name__ == '__main__':
    main()