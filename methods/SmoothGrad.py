
"""Utilities to compute SaliencyMasks."""

import numpy as np
import torch
import model as m

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# criterion = torch.nn.CrossEntropyLoss()
# model = m.Model()

# model.load_state_dict(torch.load("model_weights/model_baseline.dth", map_location=device))
# model.eval()

def GetSmoothedMask(
  x_value, stdev_spread=.15, nsamples=25,
  magnitude=True):
    """Returns a mask that is smoothed with the SmoothGrad method.
    Args:
      x_value: Input value, not batched.
      stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min). Defaults to 15%.
      nsamples: Number of samples to average across to get the smooth gradient.
      magnitude: If true, computes the sum of squares of gradients instead of
                 just the sum. Defaults to true.
    """
    x_np = x_value.numpy()
    stdev = stdev_spread * (np.max(x_np) - np.min(x_np))
    
    total_gradients = torch.tensor(np.zeros_like(x_value))
    for i in range(nsamples):
        noise = np.random.normal(0, stdev, x_value.shape)
        x_plus_noise = x_np + noise
        x_noise_tensor = torch.tensor(x_plus_noise, dtype = torch.float32)
        
        gradient = returnGrad(x_noise_tensor)
        
        if magnitude:
            total_gradients += (gradient * gradient)
        else:
            total_gradients += gradient
    
    return total_gradients / nsamples

def returnGrad(img):
    
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]))
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx
