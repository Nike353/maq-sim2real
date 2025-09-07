import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.ppo.ppo import PPO
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
import wandb
console = Console()

class PPOIM(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)
        
        
    def _init_config(self):
        super()._init_config()
        self.eval_frequency = self.config.eval_frequency
    
    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        # do not use track, because it will confict with motion loading bar
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):
            
            self.start_time = time.time()

            # Jiawei: Need to return obs_dict to update the obs_dict for the next iteration
            # Otherwise, we will keep using the initial obs_dict for the whole training process
            obs_dict =self._rollout_step(obs_dict)

            loss_dict = self._training_step()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': num_learning_iterations
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                
            if it % self.save_latest_interval == 0:
                self.save(os.path.join(self.log_dir, 'model.pt'.format(it)))
                
            self.ep_infos.clear()
            
            if (it + 1) % self.eval_frequency == 0:
                eval_res = self.evaluate_policy()
                eval_res['it'] = it
                if self.config.get("auto_pmcp", True):
                    self.env.update_soft_sampling_weight(eval_res['failed_keys'])
                self.eval_logging(eval_res)
                self._train_mode()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def _post_evaluate_policy(self):
        self.env.set_is_training()
        
        
        
    def eval_logging(self, eval_res):
        metrics_success = eval_res['metrics_success']
        metrics_all = eval_res['metrics_all']
        
        log_dict = {}
        for k, v in metrics_success.items():
            log_dict[f'eval/success/{k}'] = v
        for k, v in metrics_all.items():
            log_dict[f'eval/all/{k}'] = v
        wandb.log(log_dict, step=eval_res['it'])
    