import gymnasium as gym

from simplyqrl.ppo import PPO
from simplyqrl.transformations import CartPoleNormalizationTransformer

# ENVIROMENT CREATION
env = gym.make("CartPole-v1")

# PPO configuration used to generate the runs reported in the paper.
# Keep these values explicit here: the generic PPO class has its own defaults,
# and those defaults are not the experimental protocol.
PAPER_PPO_CONFIG = {
    "learning_rate": 2.5e-4,
    "num_envs": 1,
    "num_steps": 512,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 16,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
}


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 1: SOLVING THE ENV USING SKOLIK ET AL. + DATA REUPLOADING ##

config_skolik = {
    "circ_type": "skolik",                                  # Uses the Skolik et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits ("n_qubits": 8 > For augemented tests)
    "n_layers_q": 5,                                        # 5 Layers of DR are applied ("n_layers_q": 1,2,5)
    "ent": True,                                            # Entanglement is left ON (Default for Skolik et al.)
    "transform_fn": CartPoleNormalizationTransformer(),     # Customized normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1, ..., 10])
run_name = "Skolkik_DR"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_skolik, run_name=run_name, seed=seed, **PAPER_PPO_CONFIG)
ppo.agent.get_agent_info(env)  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)
ppo.save("checkpoints/PPO/CP_skokik_dr")
ppo.close()


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 2: SOLVING THE ENV USING SALINAS ET AL. + DATA REUPLOADING ##

config_uqc = {
    "circ_type": "dr",                                      # Uses the UQC (Salinas et al.) architecture for the quantum layer
    "n_qubits": 1,                                          # Circuit uses 1 Qubit ("n_qubits": 2 > For augemented tests)
    "emb_indices": [1,2,3],                                 # Observation has to be reduced due to UQC embedding only having 3 rotations (Rz,Ry,Rz)
    "n_layers_q": 5,                                        # 5 Layers of DR are applied ("n_layers_q": 1,2,5)
    "ent": True,                                            # Entanglement is left ON (No effect on 1Q circuit)
    "transform_fn": CartPoleNormalizationTransformer(),     # Customized normalization for CartPole (as in Skolik et al.)
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1, ..., 10])
run_name = "Salinas_DR"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_uqc, run_name=run_name, seed=seed, **PAPER_PPO_CONFIG)
ppo.agent.get_agent_info(env)  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)
ppo.save("checkpoints/PPO/CP_salinas_dr")
ppo.close()


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##

# NOTE: Experiments 3 & 4 are performed ase above, but with "n_qubits": 8  and "n_qubits": 2, respectively
