import argparse
import os

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Multi Agent Systems for Financial Graphs")
    
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=1, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")
    parser.add_argument("--maximum_position", type=int, default=1000, help="amount of fiat currency in the system")
    parser.add_argument("--haircut-multiplier", type=int, default=0.5, help="discount applied to insolvent banks")
    parser.add_argument("--n-agents", type=int, default=3, help="number of banks")


    # Training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    

    # Saving and checkpoints
    parser.add_argument("--save-dir", type=str, default="model", help="directory in which training state and model should be saved")
    parser.add_argument("--experiment-name", type=str, default="volunteers_dilemma", help="Name of the Experiment")
    parser.add_argument("--save-rate", type=int, default=200, help="save model once every time this many episodes are completed")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=100, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=1, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", dest='evaluate', action='store_true', help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=250, help="how often to evaluate model")
    args = parser.parse_args()

    args.save_dir = f'./results/{args.experiment_name}'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(f"{args.save_dir}/models")

    return args
