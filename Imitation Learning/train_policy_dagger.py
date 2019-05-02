import pickle, tensorflow as tf, tf_util, numpy as np
import policy as pol

'''
Code for training the policy over the aggregated expert policy data.
Assumes that the expert rollout data has been generated.

Example usage: 
python3.6 train_policy_dagger.py Humanoid-v2 --num_epochs 200 --num_rollouts 25 --firsttime 0 --step_size 1e-4
OR
Run script train_dagger.bash via terminal.
'''

def train(args):
    data=None
    try:
        with open('rollouts/'+args.name+'-'+str(args.num_rollouts)+'-dagger.pkl', 'rb') as f:
            data = pickle.load(f)
            print('dagger data')
    except:
        with open('rollouts/'+args.name+'-'+str(args.num_rollouts)+'-expert.pkl', 'rb') as f:
            data = pickle.load(f)
            print('expert data')

    #Loading net parameters from saved expert observation data
    obs, actions = data['observations'], data['actions']
    length, n_in, n_out = obs.shape[0], obs.shape[1], actions.shape[2]
    total_batch = length//pol.batch_size
    print('Neural net dimensions:',n_in,'x',n_out)
    
    #Initializing network with specified layer dimensions
    x, y = pol.placeholder_inputs(None,n_in,n_out,pol.batch_size)    
    logits = pol.inference(x, n_in, n_out, pol.n_h1, pol.n_h2, pol.n_h3)
    loss = pol.loss(logits, y)
    train_op = pol.training(loss,args.step_size)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer() 
    sess = tf.Session()
    sess.run(init)
    
    try:
        if args.firsttime!=1:
            saver.restore(sess, "trained-DAgger/" + args.name)
    except:
        print('Checkpoint invalid/not found. Training from the start!')

    writer = tf.summary.FileWriter("./graphs",sess.graph)

    obs_org=data['observations']
    act_org=data['actions']
    for epoch in range(args.num_epochs+1):
        avg_cost = 0
        arr = np.arange(len(data['actions']))
        np.random.shuffle(arr)
        data_shf = {'observations':obs_org[arr],'actions':act_org[arr]}
        for i in range(total_batch):
            feed_dict = pol.fill_feed_dict(x,y,data_shf,i,n_in,n_out,pol.batch_size)
            _, c =sess.run([train_op, loss], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print("Epoch:","%04d"%(epoch), "Cost=","{:.9f}".format(avg_cost))
        if(epoch % 25 == 0):
            summary_str = sess.run(summary, feed_dict=feed_dict)
            writer.add_summary(summary_str, epoch)
            writer.flush()
            saver.save(sess, "trained-DAgger/"+args.name)
        saver.save(sess, "trained-DAgger/"+args.name)
    sess.close()
    print('Training Completed!')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--num_epochs', type = int)
    parser.add_argument('--num_rollouts', type = int)
    parser.add_argument('--firsttime', type = int)
    parser.add_argument('--step_size', type = float)
    args = parser.parse_args()
    train(args)
