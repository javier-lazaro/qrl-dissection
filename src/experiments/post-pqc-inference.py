import gymnasium as gym

from simplyqrl.ppo import PPO
from simplyqrl.transformations import ArctanTransformer

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


## EXPERIMENT 1: SOLVING THE ENV USING HSIAO ET AL. + OUTPUT REUSE ##

# HYBRID AGENT CONFIGURATION - HSIAO + OR
config_hybrid = {
    "circ_type": "hsiao",                                   # Uses the Hsiao et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits
    "emb_indices": [1,2,3],                                 # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "reuse_repetitions": 4,                                 # R=4: repeat the full 4-qubit output vector 4 times -> 16 head inputs
    "n_layers_q": 1,                                        # No Layers of DR are applied
    "ent": False,                                           # Entanglement is left OFF (Default for Hsiao et al.)
    "transform_fn": ArctanTransformer(),                    # ArctTan normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1, ..., 10])
run_name = "Quantum_OR"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_hybrid, run_name=run_name, seed=seed, **PAPER_PPO_CONFIG)
ppo.agent.get_agent_info(env)  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)
ppo.save("checkpoints/PPO/CP_quantum_or")
ppo.close()


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 2: SOLVING THE ENV USING CLASSICAL + OUTPUT REUSE ##

# CLASSICAL AGENT CONFIGURATION - CLASSICAL + OR
config_classical = {
    "reuse_indices": [1,2,3],   # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "n_repeats": 4,             # R=4: repeat the 3 selected features 4 times -> 12 head inputs
    "net_arch": [],             # No additional hidden layers are added, just:  input > reuse > output
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1, ..., 10])
run_name = "Classical_OR"   # Name used for logs
ppo = PPO(env=env, agent_type="mlp", agent_config=config_classical, run_name=run_name, seed=seed, **PAPER_PPO_CONFIG)
ppo.agent.get_agent_info()  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)
ppo.save("checkpoints/PPO/CP_classical_or")
ppo.close()


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##

# NOTE: All of the above experiments are repeated using varing values of OR
