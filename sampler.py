import math
import torch
import torch.nn as nn

def get_from_idx(element: torch.Tensor, idx: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, idx)
    return ele.reshape(-1, 1, 1, 1)

class ResShiftDiffusion(nn.Module):
    def __init__(self, timesteps=15, p=0.3, kappa=2.0, etas_end=0.99, min_noise_level=0.04):
        super().__init__()
        self.timesteps = timesteps
        self.kappa = kappa
        sqrt_eta_1  = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        b0 = math.exp(1/float(timesteps-1)*math.log(etas_end/sqrt_eta_1))
        base = torch.ones(timesteps)*b0
        beta = ((torch.linspace(0,1,timesteps))**p)*(timesteps-1)
        sqrt_eta = torch.pow(base, beta) * sqrt_eta_1

        self.register_buffer("sqrt_eta", sqrt_eta)
        self.register_buffer("eta", sqrt_eta**2)

        prev_eta = torch.roll(self.eta, 1)
        prev_eta[0] = 0

        alpha = self.eta - prev_eta
        self.register_buffer("alpha", alpha)
        
        self.register_buffer("backward_mean_c1", prev_eta / self.eta)
        self.register_buffer("backward_mean_c2", self.alpha / self.eta)
        self.register_buffer("backward_std", kappa*torch.sqrt(prev_eta*self.alpha/self.eta))

    
    def add_noise(self, x, y, epsilon, t):
        eta = get_from_idx(self.eta, t)
        sqrt_eta = get_from_idx(self.sqrt_eta, t)
        
        mean = x + eta*(y-x)
        std = self.kappa*sqrt_eta

        return mean + std*epsilon
    
    def backward_step(self, x_t, predicted_x0, t):
        mean_c1, mean_c2 = get_from_idx(self.backward_mean_c1, t), get_from_idx(self.backward_mean_c2, t)
        std = get_from_idx(self.backward_std, t)

        mean = mean_c1*x_t + mean_c2*predicted_x0

        return mean + std*torch.randn_like(x_t)
    
    def prior_sample(self, y, epsilon):
        t = torch.tensor([self.timesteps-1,] * y.shape[0], device=y.device).long()
        return y + self.kappa * get_from_idx(self.sqrt_eta, t) * epsilon

class SimpleDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.timesteps = timesteps
        # The betas and the alphas
        self.register_buffer("beta" ,self.get_betas(beta_schedule, beta_start, beta_end))
        # Some intermediate values we will need
        self.register_buffer("alpha", 1 - self.beta)
        # self.register_buffer("alpha", 1 - torch.cat([torch.zeros(1).to(self.beta.device), self.beta], dim=0))
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("one_by_sqrt_alpha", 1. / torch.sqrt(self.alpha))
        self.register_buffer("sqrt_one_minus_alpha_bar",torch.sqrt(1 - self.alpha_bar))

    def get_betas(self, beta_schedule, beta_start, beta_end):
        if beta_schedule == "linear":
            return torch.linspace(
                beta_start,
                beta_end,
                self.timesteps,
                dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            return torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                self.timesteps,
                dtype=torch.float32) ** 2

    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor):
        # Generate normal noise
        epsilon = torch.randn_like(x0)
        mean    = get_from_idx(self.sqrt_alpha_bar, timesteps) * x0      # Mean
        std_dev = get_from_idx(self.sqrt_one_minus_alpha_bar, timesteps) # Standard deviation
        # Sample is mean plus the scaled noise
        sample  = mean + std_dev * epsilon
        return sample, epsilon
