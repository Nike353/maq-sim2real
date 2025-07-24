import torch
import torch.nn as nn
import inspect

def get_norm(norm_type, dim):
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim)
    elif norm_type is None:
        return None
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_type="layer_norm", activation="SiLU"):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        norm = get_norm(norm_type, dim)
        if norm:
            layers.append(norm)
        layers.append(getattr(nn, activation)())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, norm="layer_norm", activation="SiLU"):
        super().__init__()

        # Input projection
        input_layers = [nn.Linear(input_dim, hidden_dim)]
        norm_layer = get_norm(norm, hidden_dim)
        if norm_layer:
            input_layers.append(norm_layer)
        input_layers.append(getattr(nn, activation)())
        self.input_layer = nn.Sequential(*input_layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, norm_type=norm, activation=activation) for _ in range(depth)]
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict=None, module_config_dict=None, module_dim_dict={}, env_config=None, algo_config=None, process_output_dim=False,build_final_layer = True):
        super(BaseModule, self).__init__()

        self.env_config = env_config
        self.algo_config = algo_config
        if obs_dim_dict is None:
            self.obs_dim_dict = env_config.robot.algo_obs_dim_dict
        else:
            self.obs_dim_dict = obs_dim_dict
            
        self.module_config_dict = module_config_dict
        if process_output_dim:
            self.module_config_dict = self._process_module_config(self.module_config_dict, self.env_config.robot.actions_dim)

        self.module_dim_dict = module_dim_dict
        self.build_final_layer = build_final_layer
        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            elif each_input in self.module_dim_dict:
                input_dim += self.module_dim_dict[each_input]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if each_output in self.obs_dim_dict:
                output_dim += self.obs_dim_dict[each_output]
            elif isinstance(each_output, (int, float)):
                output_dim += each_output
            elif each_output in self.module_dim_dict:
                output_dim += self.module_dim_dict[each_output]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        elif layer_config['type'] == 'CNN':
            self._build_cnn_layer(layer_config)
        elif layer_config['type'] == 'GRU':
            self._build_gru_layer(layer_config)
        elif layer_config['type'] == 'ResidualMLP':
            self._build_residual_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")
        
    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                if self.build_final_layer:
                    layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def _build_cnn_layer(self, layer_config):
        layers = []
        channel_dims = layer_config['channel_dims']
        activation = getattr(nn, layer_config['activation'])()
        
        # Get input dimensions from vision_obs
        H1 = int((self.obs_dim_dict['vision_obs'] / 3) ** 0.5)
        H2 = int((self.obs_dim_dict['vision_obs'] / 1) ** 0.5)
        if H1 * H1 * 3 == self.obs_dim_dict['vision_obs']:
            vision_obs_dim = [H1, H1, 3]
        elif H2 * H2 == self.obs_dim_dict['vision_obs']:
            vision_obs_dim = [H2, H2, 1]    
        else:
            raise ValueError(f"vision_obs dimension should be (channels, height, width), got {vision_obs_dim}")
        print("vision_obs_dim", vision_obs_dim)
        assert vision_obs_dim[0] * vision_obs_dim[1] * vision_obs_dim[2] == self.obs_dim_dict['vision_obs']
        if len(vision_obs_dim) != 3:
            raise ValueError(f"vision_obs dimension should be (channels, height, width), got {vision_obs_dim}")
        input_width, input_height, input_channels = vision_obs_dim
        
        # Get layer configurations
        layer_configs = layer_config.get('layers', [])
        use_batch_norm = layer_config.get('norm_config', {}).get('use_batch_norm', False)
        
        # Track spatial dimensions and channels
        current_height, current_width = input_height, input_width
        current_channels = input_channels
        conv_idx = 0  # Track which conv layer we're on for channel dimensions
        
        for layer_cfg in layer_configs:
            layer_type = layer_cfg['type']
            
            if layer_type == 'conv':
                # Get conv parameters
                kernel_size = layer_cfg.get('kernel_size', 3)
                stride = layer_cfg.get('stride', 1)
                padding = layer_cfg.get('padding', 1)
                
                # Determine output channels
                if conv_idx < len(channel_dims):
                    out_channels = channel_dims[conv_idx]
                else:
                    out_channels = self.output_dim
                
                # Add conv layer
                layers.append(nn.Conv2d(current_channels, out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(activation)
                
                # Update dimensions
                current_channels = out_channels
                current_height = (current_height - kernel_size + 2 * padding) // stride + 1
                current_width = (current_width - kernel_size + 2 * padding) // stride + 1
                conv_idx += 1
                
            elif layer_type == 'pool':
                # Get pool parameters
                kernel_size = layer_cfg.get('kernel_size', 2)
                stride = layer_cfg.get('stride', 2)
                
                # Add pooling layer if dimensions allow
                if current_height >= kernel_size and current_width >= kernel_size:
                    layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
                    current_height = current_height // stride
                    current_width = current_width // stride
        
        # Add global average pooling if spatial dimensions are too small
        # if current_height * current_width > 1:
        #     # import ipdb; ipdb.set_trace()
        #     layers.append(nn.AdaptiveAvgPool2d(1))

        layers.append(nn.Flatten())

        layers.append(nn.Linear(current_channels * current_height * current_width, self.output_dim))
        
        self.module = nn.Sequential(*layers)

    def forward_without_hidden_state(self, input):
        return self.module(input)
    
    def forward_with_hidden_state(self, input, hidden_state):
        # import ipdb; ipdb.set_trace()
        output, hidden_state = self.module(input, hidden_state)
        return output, hidden_state
    
    def forward(self, input, hidden_state=None):
        if hidden_state is None:
            return self.forward_without_hidden_state(input)
        else:
            return self.forward_with_hidden_state(input, hidden_state)
    
    def _build_gru_layer(self, layer_config):
        self.module = nn.GRU(input_size=self.input_dim, hidden_size=layer_config['hidden_dim'], num_layers=layer_config['num_layers'], batch_first=True)
    def forward(self, input, **kwargs):
        return self.module(input)
        