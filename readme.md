# Reinforcement Learning for Game Theory: Financial Graphs

The purpose of this repository is to present the code used for the masters thesis, "Reinforcement Learning for Game Theory: Financial Graphs" conducted at ETH Zurich under the supervision of the DISCO Laboratory.

In this thesis, we began the investigation into the ability of learning agents (e.g. independent reinforcement learning[1]) to solve financial dilemmas.  We show how six different financial dilemmas can arise with just three nodes financial graphs, how the availability of information influences agents' decision making and resulting impact on fairness, and begin work on population based games[2].  

Serving as an interesting starting point, there remains multiple research directions to expanding this work such as, but not limited to, incorporating different learning architectures, incorporating market mechanisms, introducing spatial and ad-hoc agent considerations by introducing fixed policy agents in the fashion of agent-based models, introducing noise in agents observation sets in the form of information streams, introducing a variety of financial assets, and refining a process by which real assets impact financial instruments (establishing a link between real assets and financial instruments).  While selectively picking amongst these additions may lead to insights, it may result in more confusion than knowledge by haphazardly incorporating all these considerations.  Therefore, significant wisdom is required in determining the correct question to ask.  We hope future researchers are able to take up these ideas.  

The report can be found <a href="https://kth.diva-portal.org/smash/get/diva2:1616628/FULLTEXT01.pdf"> here </a> </br>
Slides can be found <a href="https://docs.google.com/presentation/d/1GxS8eafzcWbo8zSeEq4Rru0MJ7InjSgLbJe53kyY1o4/edit?usp=sharing"> here </a>

![alt text](https://github.com/yubryanj/MARL-Game-Theory/blob/master/Assets/game_theory.png?raw=true)
|:--:|
| <b>Coordination game in 3 node financial graph - Fig.1 - Minor modification in asset and liability allocation results in various game theoretic scenarios.  Here we demonstrate one allocation resulting in the "coordination game," a scenario where both agents have to learn to work together to maximize their rewards.</b>|

![alt text](https://github.com/yubryanj/MARL-Game-Theory/blob/master/Assets/learned_behaviors.png?raw=true)
|:--:|
| <b>Learned Behaviors - Fig.2 - We show that independent reinforcement learning agents can find equilibriums and demonstrate intricate strategies.  In this image agent 1 (bank B) learns the strategy of filling in the difference while agent 2(bank C) learns a constant allocation.  </b>|


## Usage:
Define experiment parameters in configs.json and run by referencing the experiment identifier.  For example, running experiment 39 is as follows
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
* <a id="1">[1] Tan, Ming. "Multi-agent reinforcement learning: Independent vs. cooperative agents." Proceedings of the tenth international conference on machine learning. 1993. </a>
* <a id="2">[2] Sandholm, William H. Population games and evolutionary dynamics. MIT press, 2010. </a>
