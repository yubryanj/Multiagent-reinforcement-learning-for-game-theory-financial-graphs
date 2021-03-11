from runner import Runner
from common.args import get_args
from common.utils import make_env

if __name__ == "__main__":
    # Retrieve the arguments
    args = get_args()

    # Initialize the environment
    env, args = make_env(args)

    # Initialize the trainer module
    runner = Runner(args, env)

    # If the evaulation flag is enabled, evaluate the model
    if args.evaluate:
        returns = runner.evaluate(args)
    else:
        # Else train the model
        runner.run()