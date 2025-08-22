python state_publisher.py --config=config/h1.yaml

python rl_inference/locomotion.py --config=config/h1.yaml --model_path="/Users/tairanhe/Downloads/h1_1.5/h1_1.5_0/exported/h1_1.5_0_ckpt4000.pt" --use_jit
python rl_inference/locomotion.py --config=config/h1.yaml --model_path="/Users/tairanhe/Downloads/humanoid_rough_ckpt2400.pt" --use_jit

python rl_inference/locomotion.py --config=config/g1_29dof.yaml --model_path="/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/g1_29dof_loco/model_500.onnx"

python command_sender.py --config=config/h1.yaml


python rl_inference/locomotion.py --config=config/h1-2.yaml --model_path="/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/genesis_h1-2/h12_1_1.5_vel_ckpt1700.pt" --use_jit


## MuJoCo Sim2Sim

```bash
python -m mujoco.viewer
```

## Sim2Sim

```bash
python sim_env/base_sim.py --config=config/g1_29dof.yaml
```
```bash
python state_publisher.py --config=config/g1_29dof.yaml
```

```bash
python rl_inference/locomotion.py --config=config/g1_29dof.yaml --model_path="/home/jiawei/Research/humanoid/RoboVerse/logs/Sim2Sim/20241126_141208-trail_0_resume0.95_linkmassRand-locomotion-g1_29dof/exported/model_500.onnx"
```

```bash
python command_sender.py --config=config/g1_29dof.yaml
```


## Jiawei

```bash
python sim_env/base_sim.py --config=config/h1-2.yaml
```
```bash
python state_publisher.py --config=config/h1-2.yaml
```

```bash
python rl_inference/locomotion.py --config=config/h1-2.yaml --model_path="/home/jiawei/Research/humanoid/RoboVerse/logs/H1-2locomotion/20241203_171030-H1-227DofLoco-locomotion-h1_2_27dof/exported/model_4600.onnx"
```

```bash
python command_sender.py --config=config/h1-2.yaml
```


