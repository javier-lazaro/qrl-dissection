import gymnasium as gym

from simplyqrl.ppo import PPO
from simplyqrl.transformations import CartPoleNormalizationTransformer, ArctanTransformer

# ENVIROMENT CREATION
env = gym.make("CartPole-v1")


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 1: SOLVING THE ENV USING HSIAO ET AL. + ENTANGLEMENT + OUTPUT REUSE ## 

config_tempA_or = {
    "circ_type": "hsiao",                                   # Uses the Hsiao et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits
    "emb_indices": [1,2,3],                                 # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "reuse_repetitions": 4,                                 # Output is reused 4 times ("reuse_repetitions": 4, 8)
    "n_layers_q": 1,                                        # No Layers of DR are applied
    "ent": True,                                            # Entanglement is set to ON ("ent": True, False")
    "transform_fn": ArctanTransformer(),                    # ArctTan normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                            # Seed for reproducibility  (Seeds used: [1,..,9])                 
run_name = "Hsiao_OR_Entangled"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_tempA_or, run_name=run_name, seed=seed)
ppo.agent.get_agent_info(env)       # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)   
ppo.save("checkpoints/PPO/CP_hsiao_or_entangled")               
ppo.close() 


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 2: SOLVING THE ENV USING SKOLIK ET AL. + UNENTANGLEMENT + DATA REUPLOADING ## 

config_tempB_dr = {
    "circ_type": "skolik",                                  # Uses the Skolik et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits 
    "n_layers_q": 5,                                        # 5 Layers of DR are applied ("n_layers_q": 1,2,5)
    "ent": False,                                           # Entanglement is set to OFF ("ent": True, False")
    "transform_fn": CartPoleNormalizationTransformer(),     # Customized normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                                # Seed for reproducibility  (Seeds used: [1,..,9])                 
run_name = "Skolkik_DR_Unentangled"     # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_tempB_dr, run_name=run_name, seed=seed)
ppo.agent.get_agent_info(env)           # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)   
ppo.save("checkpoints/PPO/CP_skokik_dr_unentangled")               
ppo.close() 


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##


## EXPERIMENT 3: SOLVING THE ENV USING HSIAO ET AL. + ENTANGLED + DATA REUPLOADING ## 

config_tempA_dr = {
    "circ_type": "hsiao",                                   # Uses the Hsiao et al. PQC architecture for the quantum layer
    "n_qubits": 4,                                          # Circuit uses 4 Qubits
    "emb_indices": [1,2,3],                                 # Observation array indices to use during embedding - Discard cartPos (as in the original paper)
    "reuse_repetitions": 1,                                 # Output is NOT reused 
    "n_layers_q": 5,                                        # 5 Layers of DR are applied ("n_layers_q": 1,2,5)
    "ent": True,                                            # Entanglement is set to ON ("ent": True, False")
    "transform_fn": ArctanTransformer(),                    # ArctTan normalization for CartPole (as in the original paper)
}

# Algorithm setup
seed = 1                                # Seed for reproducibility  (Seeds used: [1,..,9])                 
run_name = "Hsiao_DR_Entangled"         # Name used for logs
ppo = PPO(env=env, agent_type="hybrid", agent_config=config_tempA_dr, run_name=run_name, seed=seed)
ppo.agent.get_agent_info(env)           # Shows the full information of the agent

# Training and saving the agent
ppo.train(total_timesteps=100_000, progress_bar=True)   
ppo.save("checkpoints/PPO/CP_hsiao_dr_entangled")               
ppo.close() 


## ---------------------------------------------------------------------------------------------------------------------------------------------- ##

# NOTE: All of the above experiments are repeated using the inverse entanglement setup, as well as values of OR and DR depth 