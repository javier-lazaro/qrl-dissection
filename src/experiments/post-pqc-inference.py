import gymnasium as gym

from simplyqrl.ppo import PPO
from simplyqrl.transformations import ArctanTransformer

# ENVIROMENT CREATION
env = gym.make("CartPole-v1")


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 1: SOLVING THE ENV USING HSIAO ET AL. + OUTPUT REUSE ## 

# HYBRID AGENT CONFIGURATION - HSIAO + OR
config_hybrid = {
    "circ_type": "hsiao",                                   # Uses the Hsiao et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits
    "emb_indices": [1,2,3],                                 # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "reuse_repetitions": 4,                                 # Output is reused 4 times ("reuse_repetitions": 4,8,16,32)
    "n_layers_q": 1,                                        # No Layers of DR are applied
    "ent": False,                                           # Entanglement is left OFF (Default for Hsiao et al.) 
    "transform_fn": ArctanTransformer(),                    # ArctTan normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1,..,9])                 
run_name = "Quantum_OR"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_hybrid, run_name=run_name, seed=seed)
ppo.agent.get_agent_info(env)  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)   
ppo.save("checkpoints/PPO/CP_quantum_or")               
ppo.close()  


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 2: SOLVING THE ENV USING CLASSICAL + OUTPUT REUSE ## 

# HYBRID AGENT CONFIGURATION - CLASSICAL + OR
config_classical = {
    "reuse_indices": [1,2,3],   # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "n_repeats": 4,             # Output is reused 4 times ("reuse_repetitions": 4,8,16,32)
    "net_arch": [],             # No additional hidden layers are added, just:  input > reuse > output 
}

# Algorithm setup
seed = 1                    # Seed for reproducibility  (Seeds used: [1,..,9])                 
run_name = "Classical_OR"   # Name used for logs
ppo = PPO(env=env, agent_type="classic", agent_config=config_classical, run_name=run_name, seed=seed)
ppo.agent.get_agent_info(env)  # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)   
ppo.save("checkpoints/PPO/CP_classical_or")               
ppo.close() 


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##

# NOTE: All of the above experiments are repeated using varing values of OR