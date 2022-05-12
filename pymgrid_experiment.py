import wandb
from A2C import A2C
from config import config
from pymgrid_config import pymgrid_config
from pymgrid_utils import get_environments, get_train_env

if __name__ == "__main__":
    config["name"] = "pymgrid"
    config_global = {**config, **pymgrid_config}
    for n_step in [1]:
        config_global["N_STEPS"] = n_step
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
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
                pv_factor=pymgrid_config["pv_factor"],
                action_design=pymgrid_config["action_design"],
                export_price_factor=pymgrid_config["export_price_factor"],
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
