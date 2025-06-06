import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class SimpleNN(nn.Module):
    """
    Feed-forward neural network for factor return prediction.
    
    Architecture: Input(83) → Linear → ReLU → Dropout → ... → Linear(83)
    """
    
    def __init__(self, input_size: int = 83, hidden_sizes: List[int] = [512, 256], 
                 dropout_rate: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input factors (83)
            hidden_sizes: List of hidden layer dimensions
            dropout_rate: Dropout probability (0.0 to 0.5)
        """
        super(SimpleNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (no activation - we want raw predictions)
        layers.append(nn.Linear(prev_size, input_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created SimpleNN: {input_size} → {' → '.join(map(str, hidden_sizes))} → {input_size}")
        logger.info(f"Dropout rate: {dropout_rate}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, input_size) - predicted returns
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': self.count_parameters(),
            'architecture': str(self)
        }

def top5_return_loss(predictions: torch.Tensor, actual_returns: torch.Tensor) -> torch.Tensor:
    """
    Custom loss function that maximizes the average return of the top 5 predicted factors.
    
    Uses a differentiable soft selection mechanism to approximate top-5 selection.
    
    Args:
        predictions: Model predictions of shape (batch_size, n_factors)
        actual_returns: Actual returns of shape (batch_size, n_factors)
    
    Returns:
        loss: Negative weighted return (for minimization to maximize returns)
    """
    # Use temperature-scaled softmax to create soft top-k selection
    # Lower temperature = more focused on top predictions
    temperature = 0.1
    
    # Create selection weights that favor top predictions
    selection_weights = F.softmax(predictions / temperature, dim=1)
    
    # Weight actual returns by selection probabilities
    # Higher predicted factors get more weight in the final return calculation
    weighted_returns = (selection_weights * actual_returns).sum(dim=1)
    
    # Return negative mean (minimize negative = maximize positive)
    loss = -weighted_returns.mean()
    
    return loss

def top5_return_loss_eval(predictions: torch.Tensor, actual_returns: torch.Tensor) -> torch.Tensor:
    """
    Evaluation version of top5 loss using hard top-5 selection (non-differentiable).
    
    Args:
        predictions: Model predictions of shape (batch_size, n_factors)
        actual_returns: Actual returns of shape (batch_size, n_factors)
    
    Returns:
        loss: Negative mean return of actual top 5 predicted factors
    """
    with torch.no_grad():
        # Get indices of top 5 predictions for each sample
        _, top5_indices = torch.topk(predictions, k=5, dim=1)
        
        # Gather actual returns for the predicted top 5
        selected_returns = torch.gather(actual_returns, 1, top5_indices)
        
        # Calculate mean return
        mean_returns = selected_returns.mean(dim=1)
        loss = -mean_returns.mean()
    
    return loss

def calculate_top5_metrics(predictions: torch.Tensor, actual_returns: torch.Tensor) -> dict:
    """
    Calculate metrics for the top 5 predicted factors.
    
    Args:
        predictions: Model predictions
        actual_returns: Actual returns
    
    Returns:
        Dictionary with metrics
    """
    with torch.no_grad():
        batch_size, n_factors = predictions.shape
        
        # Get top 5 predicted and actual factors
        _, pred_top5_idx = torch.topk(predictions, k=5, dim=1)
        _, actual_top5_idx = torch.topk(actual_returns, k=5, dim=1)
        
        # Calculate average return of predicted top 5
        selected_returns = torch.gather(actual_returns, 1, pred_top5_idx)
        avg_top5_return = selected_returns.mean().item()
        
        # Calculate hit rate (overlap between predicted and actual top 5)
        hit_rates = []
        for i in range(batch_size):
            pred_set = set(pred_top5_idx[i].cpu().numpy())
            actual_set = set(actual_top5_idx[i].cpu().numpy())
            overlap = len(pred_set.intersection(actual_set))
            hit_rates.append(overlap / 5.0)  # Fraction of overlap
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        return {
            'avg_top5_return': avg_top5_return,
            'hit_rate': avg_hit_rate
        }

def create_model(config: dict, device: torch.device) -> SimpleNN:
    """
    Create and initialize a model with given configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        device: Device to place the model on
    
    Returns:
        Initialized model on the specified device
    """
    model = SimpleNN(
        input_size=config.get('input_size', 83),
        hidden_sizes=config.get('hidden_sizes', [512, 256]),
        dropout_rate=config.get('dropout_rate', 0.2)
    )
    
    model = model.to(device)
    logger.info(f"Model created and moved to device: {device}")
    
    return model

def get_optimizer(model: SimpleNN, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer for the model.
    
    Args:
        model: The neural network model
        config: Configuration dictionary
    
    Returns:
        Configured optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    logger.info(f"Created Adam optimizer: lr={config.get('learning_rate', 1e-3)}, "
                f"weight_decay={config.get('weight_decay', 1e-5)}")
    
    return optimizer