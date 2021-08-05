import argparse
import json
import os
from collections import OrderedDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dictionary_path",type=str,   default="./results_dictionary.json")
    parser.add_argument("--configs_path",           type=str,   default="./configs.json")
    parser.add_argument("--results_path",           type=str,   default="./data/results")
    args = parser.parse_args()


    # Open the configs file.  This file is needed to retrieve the number of iterations
    with open(args.configs_path, 'r+') as f:
        configs = json.load(f)
        f.close()

    # data stores path to all successful trials per experiment
    data = OrderedDict()

    # Retrieve experiments which are present in the results directory
    experiment_ids = [int(id) for id in os.listdir(args.results_path)]

    # For each experiment
    for experiment_id in sorted(experiment_ids):

        # Prepare some useful directories, variables, and allocate storage
        experiment_id = str(experiment_id)
        experiment_trials_dir = f'{args.results_path}/{experiment_id}/{configs.get(experiment_id).get("run")}'
        completion_checkpoint = configs.get(experiment_id).get('stop_iters')
        successful_trials = []

        # Look into the trial directory
        for trial_run in os.listdir(experiment_trials_dir):

            # Files containing "DQN_Volunteers_Dilemma" is where checkpoints are stored
            if 'DQN_Volunteers_Dilemma' in trial_run:

                trial_results_dir = f'{experiment_trials_dir}/{trial_run}'

                # For directories containing checkpoints
                for trial_result in os.listdir(trial_results_dir):
                    
                    # If the ifnal checkpoint exists, the experiment completed successfuly.
                    if f'checkpoint_{str.zfill(str(completion_checkpoint), 6)}' in trial_result:
                        
                        # Aggregate successful trials per experiment
                        successful_trials.append(trial_results_dir.strip('./data/results'))

        # Update dictionary to contain the list of successful trials per experiment
        data[experiment_id] = successful_trials

    # Save the results as a json file
    with open(args.results_dictionary_path, 'w') as file:
        json.dump(data, file, indent=4)




