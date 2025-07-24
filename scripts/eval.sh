# HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
# +checkpoint=logs/IsaacSimStairsDebug/20250421_144831-G1_12dof_stairs_loco_with_map-locomotion-g1_12dof/model_2000.pt

HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent_trl.py \
+checkpoint=logs/IsaacSimStairsDebug/20250423_174559-G1_12dof_stairs_loco_with_map-locomotion-g1_12dof/last.pt

