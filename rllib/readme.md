# Reinforcement Learning for Game Theory: Financial Graphs

## Usage:
Experiments are defined in configs.json.  After thus defined, they can be easily ran.  For example, running experiment 39 is as follows
```
python rllib_train.py --experiment-number 39
```

The files are described briefly below:
* custom_model.py - contains the definitions of the models used by the agents in action selection
* env.py - defines the network 
* rllib_train.py - contains the configuration for ray, rl algorithm, and environment
* utils.py - contains the graph generator and other miscellaneous
* evaluate_snapshot.py - loads a trained model and evaluates the agents behaviors
* configs.json - configuration file defining experiment parameters




## References 
* <a id="1">[1] </a> TODO
