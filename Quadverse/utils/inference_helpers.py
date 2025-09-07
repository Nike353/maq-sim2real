import torch
from torch import nn
import os
import copy

from humanoidverse.agents.modules.encoder_modules import CenetModuleVelocity

def export_policy_as_jit(actor_critic, path, exported_policy_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')

        class PPOWrapper(nn.Module):
            def __init__(self, actor):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.actor = actor

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                return self.actor.act_inference(actor_obs)

        wrapper = PPOWrapper(actor)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )


def export_policy_z_as_onnx(inference_model, env, path, exported_policy_name, example_obs_dict):
    ###### Harded coded for now.
    
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model['actor']).to('cpu')
    z_actor = copy.deepcopy(env.z_actor).to('cpu')
    prop_obs_filter = torch.zeros(env.dim_obs, dtype=torch.bool)
    actor_obs_keys = sorted(env.config.obs['obs_dict']['actor_obs'])
    counter = 0
    
    for key in actor_obs_keys:
        obs_dim = env.config.obs.obs_dims[key] 
        if key in env.z_config.obs.obs_dict.proprioception_obs:
            prop_obs_filter[counter:counter+obs_dim] = True
        counter += obs_dim
    prop_obs_filter = prop_obs_filter.nonzero().squeeze()
    
    class PPOWrapper(nn.Module):
        def __init__(self, actor, z_actor, prop_obs_filter):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.z_actor = z_actor
            self.prop_obs_filter = prop_obs_filter

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_z = self.actor(actor_obs)
            prop_obs = actor_obs[:, self.prop_obs_filter]
            prop_obs_mean = self.z_actor.running_mean_std_proprioception(prop_obs)
            x = self.z_actor.prior(prop_obs_mean)
            prior_mu = self.z_actor.prior_mu_layer(x)
            new_z = prior_mu + actor_z
            actions_mean = self.z_actor.motion_decoder(torch.cat([prop_obs_mean, new_z], dim=-1))
            return actions_mean

    wrapper = PPOWrapper(actor, z_actor, prop_obs_filter)
    example_input_list = example_obs_dict["actor_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],       # Name the output
        opset_version=13           # Specify the opset version, if needed
    )


def export_vae_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        prior = copy.deepcopy(inference_model.prior).to('cpu')
        motion_decoder = copy.deepcopy(inference_model.motion_decoder).to('cpu')
        prior_mu_layer = copy.deepcopy(inference_model.prior_mu_layer).to('cpu')
        prior_sigma_layer = copy.deepcopy(inference_model.prior_sigma_layer).to('cpu')
        running_mean_std_proprioception = copy.deepcopy(inference_model.running_mean_std_proprioception).to('cpu')
        running_mean_std = copy.deepcopy(inference_model.running_mean_std).to('cpu')
        

        class PPOWrapper(nn.Module):
            def __init__(self, prior, prior_mu_layer, prior_sigma_layer, motion_decoder, running_mean_std_proprioception, running_mean_std):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.prior = prior
                self.prior_mu_layer = prior_mu_layer
                self.prior_sigma_layer = prior_sigma_layer
                self.motion_decoder = motion_decoder
                self.running_mean_std_proprioception = running_mean_std_proprioception
                self.running_mean_std = running_mean_std

            def forward(self, proprioception_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                proprioception_obs = self.running_mean_std_proprioception(proprioception_obs)
                z = self.prior(proprioception_obs)
                z_mu = self.prior_mu_layer(z)
                z_sigma = self.prior_sigma_layer(z)
                std = torch.exp(0.5 * z_sigma)
                eps = torch.randn_like(std)
                z = z_mu + eps * z_sigma
                decoder_input = torch.cat([proprioception_obs, z], dim=-1)
                return self.motion_decoder(decoder_input)

        wrapper = PPOWrapper(prior, prior_mu_layer, prior_sigma_layer, motion_decoder, running_mean_std_proprioception, running_mean_std)
        example_input_list = example_obs_dict["proprioception_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["proprioception_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )


def export_policy_and_estimator_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_hand_force_estimator = copy.deepcopy(inference_model['left_hand_force_estimator']).to('cpu')
        right_hand_force_estimator = copy.deepcopy(inference_model['right_hand_force_estimator']).to('cpu')

        class PPOForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_hand_force_estimator, right_hand_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_hand_force_estimator = left_hand_force_estimator
                self.right_hand_force_estimator = right_hand_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, history_for_estimator = inputs
                left_hand_force_estimator_output = self.left_hand_force_estimator(history_for_estimator)
                right_hand_force_estimator_output = self.right_hand_force_estimator(history_for_estimator)
                input_for_actor = torch.cat([actor_obs, left_hand_force_estimator_output, right_hand_force_estimator_output], dim=-1)
                return self.actor.act_inference(input_for_actor), left_hand_force_estimator_output, right_hand_force_estimator_output

        wrapper = PPOForceEstimatorWrapper(actor, left_hand_force_estimator, right_hand_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["long_history_for_estimator"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "long_history_for_estimator"],  # Specify the input names
            output_names=["action", "left_hand_force_estimator_output", "right_hand_force_estimator_output"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_policy_and_cenet_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model['actor']).to('cpu')
    cenet = copy.deepcopy(inference_model['cenet']).to('cpu')
    
    class PPO_Cenet_Wrapper(nn.Module):
        def __init__(self, actor, cenet):
            super(PPO_Cenet_Wrapper, self).__init__()
            self.actor = actor
            self.cenet = cenet

        def forward(self, inputs):
            actor_obs, cenet_obs = inputs
            # check if cenet is an instance of CenetModuleVelocity
            if isinstance(self.cenet, CenetModuleVelocity):
                code_latent, code_vel, _, _, _ = self.cenet.cenet_forward(cenet_obs)
                
                actor_input = torch.cat([actor_obs, code_latent, code_vel], dim=-1)
                # actor_input = torch.cat([actor_obs, code_latent], dim=-1)

            else:
                code_latent, _, _, _ = self.cenet.cenet_forward(cenet_obs)
                actor_input = torch.cat([actor_obs, code_latent], dim=-1)
            actor_output = self.actor.act_inference(actor_input)
            return actor_output, code_latent

    wrapper = PPO_Cenet_Wrapper(actor, cenet)
    example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["cenet_obs"]]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs", "cenet_obs"],  # Specify the input names
        output_names=["action", "code_latent"],       # Name the output
        opset_version=13           # Specify the opset version, if needed
    )