# [Federated Ensemble-Directed Offline Reinforcement Learning](https://arxiv.org/abs/2305.03097)

Accepted at the Workshop of Federated Learning and Analytics in Practice, colocated with  the 40th International Conference on Machine Learning (FL-ICML 2023)

[Video of real world demonstration on TurtleBot](https://youtu.be/LplasPUm3jg)

This codebase is based on the following publicly available git repositories:  
- TD3-BC: [sfujim/TD3_BC](https://github.com/sfujim/TD3_BC)
- Flower: [adap/flower](https://github.com/adap/flower)
- Structure of simulation: [CharlieDinh/pFedMe](https://github.com/CharlieDinh/pFedMe)

The Python packages required to train FEDORA are listed in `requirements.txt` 

Our experiments are performed on Python 3.8 in a Ubuntu Linux environment.


## Federated learning with Flower

Directory: `fed_flwr/`

Specify federation parameters in `config/s_config.yml` amd `config/c_config.yml`
Specify client learning parameters in `config/c_config.yml`

a. Launch server and clients individually

    1. Launch server  
        python rl_server.py
        
    2. Launch client (repeat for each client)
        python rl_client.py --gpu-index --eval-env --start-index --stop-index

    where  
        gpu-index: index of CUDA device for PyTorch training  
        eval-env: name of the D4RL data-set source for this client  
        start-index: index of D4RL data-set to begin gathering data at  
        stop-index: index of D4RL data-set to stop gathering data at

b. To simplify this process, we share an example shell script with defalt parameters.  

    1. Specify client arguments in run_FEDORA.sh
    2. Launch the shell script
        bash run_FEDORA.sh
        
The tensorboard logs will be saved in a folder called 'Results'


## Federated learning simulation

Run single-threaded simulation of FEDORA. Helpful in training with limited computing resoures.

Directory: `fed_sim/`

Launch main program (for default parameters, simply execute `python rl_main.py`)
        
    
    python rl_main.py --env-1 --env-2 --gpu-index-1 --gpu-index-2 --n-clients --ncpr \
        --n-rounds --dataset-size --seed --batch-size --alpha-0 --alpha-1 --alpha-2 \
        --local-epochs --temp-a --temp-a --decay-rate

    where  
        env-1: name of the D4RL data-set source for first half clients  
        env-2: name of the D4RL data-set source for second half clients    
        gpu-index-1: index of CUDA device for training first half clients  
        gpu-index-2: index of CUDA device for training first second clients  
        n-clients: total number of clients  
        ncpr: number of clients participating in a round of federation  
        n-rounds: total rounds of federation  
        dataset-size: size of a client's data-set
        seed: random number seed  
    and the others are FEDORA hyperparameters.

The tensorboard logs will be saved in a folder called 'Results'

