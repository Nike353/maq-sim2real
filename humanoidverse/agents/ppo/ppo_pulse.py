import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.vae_modules import VAEActorPulseTask, VAEActorPulseTaskDeterministic

class PPOPulse(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super(PPOPulse, self).__init__(env, config, log_dir, device)
    
    def _setup_models_and_optimizer(self):
        self.actor = VAEActorPulseTaskDeterministic(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict,
            module_dim_dict=self.config.module_dim,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            freeze_decoder=True,
        ).to(self.device)
        
        self.load_pulse_decoder()

        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        
        
    def load_pulse_decoder(self, ):
        ckpt_path = self.config.network_load_dict.pulse_decoder.path
        logger.info(f"Loading teacher actor from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path)
        params = loaded_dict["actor_state_dict"]
        key_list = list(params.keys())
        for key in key_list:
            if "decoder" not in key:
                del params[key]
        key_list = list(params.keys())
        for key in key_list:
            assert key.startswith("decoder.")
            new_key = key[len("decoder."):]
            params[new_key] = params[key]
            del params[key]
        self.actor.decoder.load_state_dict(params)
        
    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["motion_encoder_obs"], obs_dict["policy_head_obs"])
    
    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['motion_encoder_obs'], actor_state["obs"]['policy_head_obs'])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)
        latent_batch = self.actor.latent
        actions_log_prob_batch = self.actor.get_actions_log_prob(latent_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # print("skip backward")
        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        return loss_dict
    
    
    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        actions = self._actor_act_step(obs_dict)
        policy_state_dict["actions"] = actions
        
        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach()
        latent_batch = self.actor.latent
        actions_log_prob = self.actor.get_actions_log_prob(latent_batch).detach().unsqueeze(1)
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob
        policy_state_dict["latent_z"] = latent_batch.detach()
        
        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict
    
    
    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(1,), dtype=torch.float)
        self.storage.register_key('returns', shape=(1,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.actor.latent_z_dim,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.actor.latent_z_dim,), dtype=torch.float)
        self.storage.register_key('latent_z', shape=(self.actor.latent_z_dim,), dtype=torch.float)