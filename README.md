The code for the numerical experiments in [Provably Efficient Exploration for Reinforcement Learning with Unsupervised Learning.](https://arxiv.org/pdf/2003.06898.pdf) by Fei Feng, Ruosong Wang, Wotao Yin, Simon S. Du, and Lin F. Yang

The code is a gentle revision of [microsoft/StateDecoding](https://github.com/microsoft/StateDecoding). Only one file is augmented:
- Experiment.py : a train_ulo function and corresponding hyperparameters are added to enable training with an unsupervised learning oracle.
Please see the original version for detailed explanations of other files.

# Run the code
To run the code, an example is:

    import Experiment as exp
    Args = exp.parse_args()
    
    # select the testing environment
    Args.env = 'Lock-v1' 
    Args.env_param_1=0.5
    Args.env_param_2=0.1
    Args.horizon = 5
    Args.dimension = Args.horizon
    Args.episodes = 20000

    # select algorithm
    Args.alg='oracleq'
    Args.tabular = True
    Args.ulo = 'kmeans'
    
    # run
    reward_vecs=[]
    for i in range(50):
        Args.iteration = i
        reward_vecs.append(exp.main(Args))

Explanation of parameters can be found in parse_args() in Experiment.py
    
# About ULO
There are 3 built-in unsupervised learning oracles to select: kmeans, dbscan, GMM. The user can specify the oracle in Args.ulo as in the above example. To disable ulo, set Args.ulo = None. The user can also implement his/her own ulo by revising the train_ulo function in Experiment.py.

