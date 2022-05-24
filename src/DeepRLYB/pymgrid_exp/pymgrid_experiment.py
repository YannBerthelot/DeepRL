import wandb
from deeprlyb.agents.A2C import A2C

# from .DeepRL.agents.A2C import A2C

import argparse

parser = argparse.ArgumentParser(
    description="My program!", formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("-s", type=argparse.FileType("r"), help="Filename to be passed")
args = vars(parser.parse_args())

config = args.s


from deeprlyb.pymgrid_exp.pymgrid_config import pymgrid_config
from deeprlyb.pymgrid_exp.utils import get_environments, get_train_env


def pymgrid_experiment():
    config["name"] = "pymgrid"
    config_global = {**config, **pymgrid_config}
    for n_step in [1]:
        config_global["GLOBAL"]["N_STEPS"] = n_step
        for experiment in range(1, config.getint(["GLOBAL"]["N_EXPERIMENTS"]) + 1):
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="Pymgrid test RBC",
                    entity="yann-berthelot",
                    name=f'{config["name"]} {experiment}/{config["N_EXPERIMENTS"]}',
                    reinit=True,
                    config=config_global,
                )
            else:
                run = None
            mg_env_train, mg_env_eval = get_environments(
                pv_factor=pymgrid_config.getfloat(["pv_factor"]),
                action_design=pymgrid_config["action_design"],
                export_price_factor=pymgrid_config.getfloat(["export_price_factor"]),
            )
            agent = A2C(
                mg_env_train,
                config,
                comment=f"baseline_pymgrid_{experiment}",
                run=run,
            )
            agent.run = run
            agent.train(mg_env_train, config["NB_TIMESTEPS_TRAIN"])

            # Load best agent from training
            agent.env = mg_env_eval
            agent.load(f"baseline_pymgrid_{experiment}_best")

            # Evaluate and render the policy
            agent.test(
                mg_env_eval,
                nb_episodes=config["NB_EPISODES_TEST"],
                render=False,
                # scaler_file=f"data/baseline_pymgrid_{experiment}_obs_scaler.pkl",
            )
            if config["logging"] == "wandb":
                run.finish()


if __name__ == "__main__":
    pymgrid_experiment()
