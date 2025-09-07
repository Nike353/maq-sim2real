import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize
from typing import Dict, Optional, Tuple, Union


class ImageFeatureExtractor(nn.Module):
    """
    A class for extracting image features using pretrained models.
    """
    
    AVAILABLE_MODELS = {
        "resnet18": (models.resnet18, 512),
        "resnet34": (models.resnet34, 512),
        "resnet50": (models.resnet50, 2048),
        "resnet101": (models.resnet101, 2048),
        "efficientnet_b0": (models.efficientnet_b0, 1280),
        "efficientnet_b1": (models.efficientnet_b1, 1280),
        "efficientnet_b2": (models.efficientnet_b2, 1408),
        "vit_b_16": (models.vit_b_16, 768),
        "vit_b_32": (models.vit_b_32, 768),
    }
    
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        device: Union[str, torch.device] = "cuda",
        input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize the feature extractor with a pretrained model.
        
        Args:
            model_name: Name of the pretrained model to use (resnet18, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            output_dim: If provided, adds a final linear layer to project features to this dimension
            freeze_backbone: Whether to freeze the backbone network
            device: Device to run the model on
            input_size: Target size for input images (height, width)
        """
        super().__init__()
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        model_fn, feature_dim = self.AVAILABLE_MODELS[model_name]
        
        # Initialize model with pretrained weights if specified
        model = model_fn(pretrained=pretrained)
        
        # For ResNet models, we want to extract features before the average pooling
        if "resnet" in model_name:
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # For EfficientNet models
        elif "efficientnet" in model_name:
            self.backbone = model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # For ViT models
        elif "vit" in model_name:
            model = model_fn(pretrained=pretrained)
            model.heads = nn.Identity() 
            self.backbone = model
            self.pool = None  
        else:
            raise ValueError(f"Model {model_name} extraction logic not implemented")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create projection head if output_dim is specified
        self.projection = None
        if output_dim is not None:
            self.projection = nn.AdaptiveAvgPool1d(output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = feature_dim
        
        # Move model to device
        self.device = device
        self.to(device)
        
        # ImageNet normalization values
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        
        # Store model name for reference
        self.model_name = model_name
        
        # Store input size as a tuple of integers
        # This handles OmegaConf objects that might be passed from config files
        self.input_size = input_size
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image tensor for feature extraction.
        
        Args:
            image: Image tensor of shape [batch_size, height, width, 3] with values in [0, 1]
            
        Returns:
            Preprocessed image tensor of shape [batch_size, 3, height, width]
        """
        # Convert from [B, H, W, 3] to [B, 3, H, W]
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        # Get current dimensions
        batch_size, channels, height, width = image.shape
        
        # Ensure input_size is a tuple of integers
        # This is needed because OmegaConf may pass ListConfig objects instead of a plain tuple
        input_size = tuple(map(int, self.input_size))
        
        # Resize the image to the expected input size of the model (typically 224x224)
        if height != input_size[0] or width != input_size[1]:
            # Use interpolate for resizing
            image = torch.nn.functional.interpolate(
                image, 
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize using ImageNet stats
        image = self.normalize(image)
        
        return image
    
    def forward(self, image: torch.Tensor, return_spatial_features: bool = False) -> torch.Tensor:
        """
        Extract features from the input image.
        
        Args:
            image: Image tensor of shape [batch_size, height, width, 3] or [batch_size, 3, height, width]
            return_spatial_features: If True, return spatial features before pooling
            
        Returns:
            If return_spatial_features=False (default):
                Features tensor of shape [batch_size, feature_dim]
                This is a global feature vector where spatial dimensions have been pooled.
                
            If return_spatial_features=True:
                Features tensor of shape [batch_size, feature_dim, height, width]
                where height and width are the spatial dimensions of the feature maps.
                For ResNet18 with 224x224 input, this is typically [batch_size, 512, 7, 7].
                The exact dimensions depend on the input size and network architecture.
        """
        # Preprocess the image
        x = self.preprocess_image(image)
        # Extract features
        features = self.backbone(x)

        if return_spatial_features:
            # Return spatial feature maps
            # Shape is [batch_size, channels, height, width]
            return features
        
        # Apply pooling if needed
        if self.pool is not None:
            features = self.pool(features)
            features = torch.flatten(features, 1)
        
        # Apply projection if defined
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    @classmethod
    def load_from_checkpoint(cls, 
                          checkpoint_path: str, 
                          map_location: Optional[Union[str, torch.device]] = None,
                          **kwargs) -> "ImageFeatureExtractor":
        """
        Load a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: Location to map the model to
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Loaded model
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Get model args from checkpoint or use defaults
        model_args = checkpoint.get("model_args", {})
        
        # Override with any provided kwargs
        model_args.update(kwargs)
        
        # Create model
        model = cls(**model_args)
        
        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])
        
        return model


def visualize_feature_maps(feature_maps, save_path=None, max_images=16, figure_size=(15, 10)):
    """
    Visualize feature maps from a convolutional layer.
    
    Args:
        feature_maps (torch.Tensor): Feature maps of shape [batch_size, channels, height, width]
        save_path (str, optional): Path to save the visualization. If None, the plot is displayed.
        max_images (int): Maximum number of feature maps to visualize.
        figure_size (tuple): Size of the matplotlib figure.
        
    Returns:
        matplotlib.figure.Figure: The figure with visualized feature maps
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is required for feature map visualization")
        return None
    
    # Take the first sample in the batch
    if feature_maps.dim() == 4:
        feature_maps = feature_maps[0]  # [channels, height, width]
    
    # Move to CPU and convert to numpy
    feature_maps = feature_maps.detach().cpu().numpy()
    
    # Limit the number of feature maps to display
    num_features = min(feature_maps.shape[0], max_images)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figure_size)
    
    # Flatten axes for easy indexing
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each feature map
    for i in range(grid_size * grid_size):
        ax = axes[i]
        
        if i < num_features:
            # Get feature map and normalize for visualization
            feature_map = feature_maps[i]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)
            
            # Plot feature map
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature {i+1}')
        else:
            # Hide unused subplots
            ax.axis('off')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")
    
    return fig


def extract_activation_maps(model, image, layer_name=None, register_hooks=True):
    """
    Extract activation maps from a specific layer of the model.
    
    Args:
        model (nn.Module): The model to extract activation maps from
        image (torch.Tensor): Input image tensor of shape [batch_size, height, width, 3] or [batch_size, 3, height, width]
        layer_name (str, optional): Name of the layer to extract activation maps from. 
                                   If None, extracts from the last layer of the backbone.
        register_hooks (bool): Whether to register hooks for activation extraction. 
                              Set to False if hooks are already registered.
    
    Returns:
        dict: Dictionary of activation maps
    """
    if isinstance(model, ImageFeatureExtractor):
        # If we're working with our ImageFeatureExtractor class
        backbone = model.backbone
        # Preprocess the image
        preprocessed_image = model.preprocess_image(image)
    else:
        # For any other model
        backbone = model
        preprocessed_image = image
    
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks if requested
    if register_hooks:
        # If layer_name is specified, try to find it
        if layer_name is not None:
            for name, module in backbone.named_modules():
                if layer_name in name:
                    hooks.append(module.register_forward_hook(get_activation(name)))
                    break
        else:
            # Otherwise, register hooks for all layers
            for name, module in backbone.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                    hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass to capture activations
    with torch.no_grad():
        _ = backbone(preprocessed_image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations 