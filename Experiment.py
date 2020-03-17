import numpy as np
import sys, os
import gym
import Environments, Params
import OracleQ, Decoding, QLearning, LinQ
import argparse
import torch
import random
import pandas as pd
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import Counter

torch.set_default_tensor_type(torch.DoubleTensor)


def parse_environment_params(args):
    ep_dict = {'horizon': args.horizon,
               'dimension': args.dimension,
               'tabular': args.tabular}
    if args.env_param_1 is not None:
        ep_dict['switch'] = float(args.env_param_1)
    if args.env_param_2 is not None:
        if args.env_param_2 == 'None':
            ep_dict['noise'] = None
        else:
            ep_dict['noise'] = float(args.env_param_2)
    return (ep_dict)

def get_env(name, args):
    env = gym.make(name)
    ep_dict = parse_environment_params(args)
    env.seed(args.seed+args.iteration*31)
    env.init(env_config=ep_dict)
    return(env)

def get_alg(name, args, env):
    if name == "oracleq":
        alg_dict = {'horizon': args.horizon,
                    'alpha': args.lr,
                    'conf': args.conf }
        alg = OracleQ.OracleQ(env.action_space.n, params=alg_dict)
    elif name == 'decoding':
        alg_dict = {'horizon': env.horizon,
                    'model_type': args.model_type,
                    'n': args.n,
                    'num_cluster': args.num_cluster}
        alg = Decoding.Decoding(env.observation_space.n, env.action_space.n,params=alg_dict)
    elif name=='qlearning':
        #assert args.tabular, "[EXPERIMENT] Must run QLearning in tabular mode"
        alg_dict = {
            'alpha': float(args.lr),
            'epsfrac': float(args.epsfrac),
            'num_episodes': int(args.episodes)}
        alg = QLearning.QLearning(env.action_space.n, params=alg_dict)
    elif name == 'linq':
        alg_dict = {
            'horizon': env.horizon,
            'conf': args.conf
            }
        alg = LinQ.LinQ(env.observation_space.n, env.action_space.n,params=alg_dict)
    return (alg)

def parse_args():
    parser = argparse.ArgumentParser(description='StateDecoding Experiments')
    parser.add_argument('--seed', type=int, default=367, metavar='N',
                        help='random seed (default: 367)')
    parser.add_argument('--iteration', type=int, default=1,
                        help="Which replicate number")
    parser.add_argument('--env', type=str, default="Lock-v0",
                        help='Environment', choices=["Lock-v0", "Lock-v1"])
    parser.add_argument('--horizon', type=int, default=4,
                        help='Horizon')
    parser.add_argument('--dimension', type=int, default=5,
                        help='Dimension')
    parser.add_argument('--tabular', type=bool, default=False,
                        help='Make environment tabular')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Training Episodes')
    parser.add_argument('--env_param_1', type=str,
                        help='Additional Environment Parameters (Switching prob)', default=None)
    parser.add_argument('--env_param_2', type=str,
                        help='Additional Environment Parameters (Feature noise)', default=None)
    parser.add_argument('--alg', type=str, default='oracleq',
                        help='Learning Algorithm', choices=["oracleq", "decoding", "qlearning", "linq"])
    parser.add_argument('--model_type', type=str, default='linear',
                        help='What model class for function approximation', choices=['linear', 'nn'])
    parser.add_argument('--lr', type=float,
                        help='Learning Rate for optimization-based algorithms', default=3e-2)
    parser.add_argument('--epsfrac', type=float,
                        help='Exploration fraction for Baseline DQN.', default=0.1)
    parser.add_argument('--eps_dbscan', type=float,
                        help='parameter for dbscan.', default=0.1)
    parser.add_argument('--conf', type=float,
                        help='Exploration Bonus Parameter for Oracle Q.', default=3e-2)
    parser.add_argument('--degrade', type=float,
                        help='degradation for learning rate', default=1.)
    parser.add_argument('--n', type=int, default = 200,
                        help="Data collection parameter for decoding algoithm.")
    parser.add_argument('--num_cluster', type=int, default = 3,
                        help="Num of hidden state parameter for decoding algoithm.")
    parser.add_argument('--num_component', type=int, default = 3,
                        help="Num of principle components for PCA")
    parser.add_argument('--min_samples', type=int, default=8,
                        help="parameter for cluster.DBSCAN")
    parser.add_argument('--ulo', type=str, default='none', 
                        help='What unsupervised learning oracle to use', choices=['kmeans', 'dbscan', 'gmm', 'none'])
    parser.add_argument('--max_train_iter', type=int, default=3, 
                        help='Maximum iteration number for unsupervised learning training')
    parser.add_argument('--svm_max_iter', type=int, default=1000, 
                        help='Maximum iteration number for support vector machine')
    parser.add_argument('--em_max_iter', type=int, default=100, 
                        help='Maximum iteration number for EM in GMM')
    parser.add_argument('--batch', type=int, default=100, 
                        help='Batch size for decoding function training')
    parser.add_argument('--b_ratio', type=float,
                        help='the ratio of set size for adding new element in label standard set')
    parser.add_argument('--print', type = int, 
                        help='number of episodes for reward saving')
    args = parser.parse_args(args=[])
    return(args)

def train(env, alg, args):
    T = args.episodes
    running_reward = 0
    reward_vec = []
    for t in range(1,T+1):
        state = env.reset()
        done = False
        while not done:
            action = alg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            alg.save_transition(state, action, reward, next_state)
            state = next_state
            running_reward += reward
        alg.finish_episode()
        if t % args.print == 0:
            reward_vec.append(running_reward/t)
            #print("[EXPERIMENT] Episode %d Completed. Average reward: %0.2f" % (t, running_reward/t), flush=True)
            print(t, running_reward/t)
        if t % 5000 == 0:
            args.conf *= args.degrade;
            args.lr *= args.degrade;
    return (reward_vec)

def train_ulo(env, alg, args):
    T = args.episodes
    J = args.max_train_iter
    H = args.horizon
    B = args.batch
    running_reward = 0
    reward_vec = []
    
    # initialization
    training_set = []
    testing_set = []
    label_standard_set = {}
    decoder = cluster.KMeans(n_clusters=args.num_cluster, random_state=0).fit(gym.spaces.np_random.binomial(1,0.5,(args.num_cluster,H+3)))
    pca = PCA(n_components=args.num_component)
    pca.fit_transform(gym.spaces.np_random.binomial(1,0.5,(args.num_component,H+3)))  
    scaler = StandardScaler()
    
    for t in range(1, T+1):
        
        # check if still need to train decoder
        if t < J:              
            
            # adaptively decrease the batch size 
            if args.ulo == 'gmm' and t > 5 :
                B = 5
                
            # sample B trajectories as training set
            for b in range(B):
                state = env.reset()
                done = False
                h = 0
                training_set.append(state)
                state = np.reshape(state, (-1, H+3))
                while not done:                       
                    if t==1:
                        action = np.random.randint(4)
                    else:
                        if args.ulo == 'gmm':
                            action = alg.select_action(str(decoder.predict(state))+str(h))   
                        else:
                            state_feature = pca.transform(scaler.transform(state))
                            action = alg.select_action(str(decoder.predict(state_feature))+str(h)) 
                    next_state, reward, done, _ = env.step(action)              
                    state = next_state  
                    h = h+1
                    training_set.append(state)
                    state = np.reshape(state, (-1, H+3))
                    
            # pca + kmeans        
            if args.ulo == 'kmeans':          
                principalComponents = pca.fit_transform(scaler.fit_transform(training_set))            
                PCA_components = pd.DataFrame(principalComponents)
                decoder = cluster.KMeans(n_clusters=args.num_cluster, random_state=0).fit(PCA_components.iloc[:,:args.num_component])
    
            # pca + dbscan
            if args.ulo == 'dbscan':
                principalComponents = pca.fit_transform(scaler.fit_transform(training_set))              
                PCA_components = pd.DataFrame(principalComponents)
                y_tr = cluster.DBSCAN(min_samples=args.min_samples).fit_predict(PCA_components.iloc[:,:args.num_component])
                decoder = svm.SVC(max_iter = args.svm_max_iter).fit(PCA_components.iloc[:,:args.num_component], y_tr)           
            
            # gmm
            if args.ulo == 'gmm':
                decoder = mixture.GaussianMixture(n_components=args.num_cluster, random_state=0).fit(training_set)
             
            # Generate testing set and FixLabel
            if (t+1) % 50 == 0:              
                
                # sample B trajectories as testing set
                testing_set.clear()
                for b in range(B):
                    state = env.reset()
                    done = False
                    h = 0
                    testing_set.append(state)
                    state = np.reshape(state, (-1, H+3))
                    while not done:                      
                        if t==1:
                            action = np.random.randint(4)
                        else:
                            if args.ulo == 'gmm':
                                action = alg.select_action(str(decoder.predict(state))+str(h))   
                            else:
                                state_feature = pca.transform(scaler.transform(state))
                                action = alg.select_action(str(decoder.predict(state_feature))+str(h))                        
                        next_state, reward, done, _ = env.step(action)              
                        state = next_state  
                        h = h+1
                        testing_set.append(state)
                        state = np.reshape(state, (-1, H+3))    
                
                # FixLabel (if unsupervised learning converges fast, this step can be neglected in practice)
                for k in label_standard_set.keys():                   
                    s = int(k)
                    if args.ulo == 'gmm':
                        decoded_set = decoder.predict(label_standard_set[k])
                    else:
                        s_set = pca.transform(scaler.transform(label_standard_set[k]))
                        decoded_set = decoder.predict(s_set[:,:args.num_component])
                    count = Counter(decoded_set) 
                    # swap labels (here we only implement for kmeans which has "cluster_centers_" property)
                    #s_ = count.most_common(1)[0][0] 
                    #temp = decoder.cluster_centers_[s_].copy()
                    #decoder.cluster_centers_[s_] = decoder.cluster_centers_[s]
                    #decoder.cluster_centers_[s] = temp

                # add new element to label_standard_set    
                if args.ulo == 'gmm':
                    predicted = decoder.predict(testing_set)
                else:
                    predicted = decoder.predict(pca.transform(scaler.transform(testing_set)))
                count = Counter(predicted)

                for s in count.keys():
                    if count[s]>args.b_ratio*B and str(s) not in label_standard_set.keys():
                        indices = [i for i, s_ in enumerate(predicted) if  s_==s]
                        label_standard_set[str(s)]= [testing_set[i] for i in indices]
                        #print("new element added in label_standard_set")
                        break
        
        # sample one trajectory and feed to algorithm
        state = env.reset()
        state = np.reshape(state, (-1, 3+H))
        done = False
        h = 0
        while not done:
            if args.ulo == 'gmm':
                decoded_state = decoder.predict(state)
            else:
                decoded_state = decoder.predict(pca.transform(scaler.transform(state)))
                
            action = alg.select_action(str(decoded_state)+str(h))           
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (-1, 3+H))
            
            if args.ulo == 'gmm':
                decoded_next_state = decoder.predict(next_state)
            else:
                decoded_next_state = decoder.predict(pca.transform(scaler.transform(next_state)))
                
            alg.save_transition(str(decoded_state)+str(h), action, reward, str(decoded_next_state)+str(h+1))
            state = next_state
            running_reward += reward
            h += 1
        alg.finish_episode()
        
        if t % args.print == 0:
            # only times one B since testing_set is rarely generated
            reward_vec.append(running_reward/(t+J*B))
            print("[EXPERIMENT] Episode %d Completed. Average reward: %0.2f" % (t+J*B, running_reward/(t+J*B)), flush=True)
    return (reward_vec)
    
def main(args):
        
    random.seed(args.seed+args.iteration*29)
    np.random.seed(args.seed+args.iteration*29)

    import torch
    torch.manual_seed(args.seed+args.iteration*37)

    env = get_env(args.env, args)
    alg = get_alg(args.alg, args, env)


    P = Params.Params(vars(args))
    fname = P.get_output_file_name()
    if os.path.isfile(fname):
        print("[EXPERIMENT] Already completed")
        return None

    # training with unsupervised learning or not      
    if args.ulo == 'none':
        reward_vec = train(env, alg, args)
    else:
        reward_vec = train_ulo(env, alg, args)
        
    print("[EXPERIMENT] Learning completed")
    f = open(fname,'w')
    f.write("\n".join([str(z) for z in reward_vec]))
    f.write("\n")
    f.close()
    print("[EXPERIMENT] Done")
    
    return reward_vec

if __name__=='__main__':
    Args = parse_args()
    print(Args)
    main(Args)
