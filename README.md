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
    Args.tabular = False
    Args.ulo = 'kmeans'
    
    # run
    reward_vecs=[]
    for i in range(50):
        Args.iteration = i
        reward_vecs.append(exp.main(Args))

Explanation of parameters can be found in parse_args() in Experiment.py

# Some important parameters
- Args.tabular: This parameter determines what the agent observes. If it is True, then the agent directly sees the latent states; otherwise, the agent only sees the high-dimensional observations.
- Args.alg: This parameter determines which RL algorithm to use. 
- Args.ulo: This parameter determines which unsupervised learning oracle to apply. There are 3 built-in options: kmeans, dbscan, and GMM. The user can specify the oracle in Args.ulo. To disable ulo, set Args.ulo = None. The user can also implement his/her own ulo by revising the train_ulo function in Experiment.py.

For example, to apply oracleq with directly seeing the latent states, the user can set:
    
    Args.alg='oracleq'
    Args.tabular = True
    Args.ulo = None

To apply oracleq with only seeing the observations and disable ulo, the user can set:
    
    Args.alg='oracleq'
    Args.tabular = False
    Args.ulo = None

To apply oracleq with only seeing the observations and use e.g., kmeans for unsupervised learning, the user can set:
    
    Args.alg='oracleq'
    Args.tabular = False
    Args.ulo = 'kmeans'
    
