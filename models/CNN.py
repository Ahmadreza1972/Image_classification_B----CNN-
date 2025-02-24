import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers_config, fc_layers_config, device, reg_lambda,dropout):
        super(CNNModel, self).__init__()

        self.device = device  # Store device info
        self.num_classes = num_classes
        self.features = nn.ModuleDict()
        self.reg_lambda = reg_lambda  # Regularization strength

        # Add convolutional layers dynamically
        in_channels = input_channels
        for i, layer_config in enumerate(conv_layers_config):
            self.features[f"conv{i+1}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"],
                padding=layer_config["padding"],
            )#.to(self.device)  # Ensure conv layer is on the correct device
            self.features[f"relu{i+1}"] = nn.ReLU()#.to(self.device)
            if layer_config["maxpool"]>0:
                self.features[f"pool{i+1}"] = nn.MaxPool2d(kernel_size=layer_config["maxpool"], stride=2)#.to(self.device)
            in_channels = layer_config["out_channels"]

        # Fully connected layer configuration
        self.flatten_size = None  # Will be dynamically determined
        self.fc_layers_config = fc_layers_config
        self.fc = nn.ModuleDict()  # Fully connected layers

        # Dropout
        self.dropout = nn.Dropout(dropout)#.to(self.device)  # Ensure dropout is on the correct device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        reg_loss = 0  # Initialize regularization loss

        # Forward pass through convolutional layers
        for layer in self.features.values():
            x = layer(x)  # No need to move x here since layers are already on the right device
            
            # Accumulate L2 regularization loss for convolutional layers
            if isinstance(layer, nn.Conv2d):  # Regularization only for weight layers
                reg_loss += self.reg_lambda * torch.sum(layer.weight ** 2)

        # Flatten the feature map
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            in_features = self.flatten_size
            for i, out_features in enumerate(self.fc_layers_config):
                self.fc[f"fc{i+1}"] = nn.Linear(in_features, out_features)#.to(self.device)
                in_features = out_features
            self.fc["output"] = nn.Linear(in_features, self.num_classes)#.to(self.device)
            self.fc.to(self.device)

        x = x.view(x.size(0), -1)  # Flatten

        # Forward pass through fully connected layers
        for name, layer in self.fc.items():
            x = F.relu(layer(x)) if name != "output" else layer(x)
            # Accumulate L2 regularization loss for fully connected layers
            if hasattr(layer, "weight"):
                reg_loss += self.reg_lambda * torch.sum(layer.weight ** 2)
            if "fc" in name:  # Apply dropout only on hidden layers
                x = self.dropout(x)

        return x

  