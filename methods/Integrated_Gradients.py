from __future__ import division
import numpy as np
import torch
import model as m
from torchvision import transforms

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# criterion = torch.nn.CrossEntropyLoss()
# model = m.Model()

# model.load_state_dict(torch.load("model_weights/model_baseline.dth", map_location=device))
# model.eval()

def integrated_gradients(
    inp, 
    target_label_index,
    baseline,
    steps=50,
    magnitude=True):
  """Computes integrated gradients for a given network and prediction label.
  Integrated gradients is a technique for attributing a deep network's
  prediction to its input features. It was introduced by:
  https://arxiv.org/abs/1703.01365
  In addition to the integrated gradients tensor, the method also
  returns some additional debugging information for sanity checking
  the computation. See sanity_check_integrated_gradients for how this
  information is used.
  
  This method only applies to classification networks, i.e., networks 
  that predict a probability distribution across two or more class labels.
  
  Access to the specific network is provided to the method via a
  'predictions_and_gradients' function provided as argument to this method.
  The function takes a batch of inputs and a label, and returns the
  predicted probabilities of the label for the provided inputs, along with
  gradients of the prediction with respect to the input. Such a function
  should be easy to create in most deep learning frameworks.
  
  Args:
    inp: The specific input for which integrated gradients must be computed.
    target_label_index: Index of the target class for which integrated gradients
      must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is the
          number of inputs and num_classes is the number of classes for the model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape. 
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.
  Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.
      
    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients
    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated gradients.
      It has shape <steps, num_classes> where 'steps' is the number of integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
  """  
  if baseline is None:
    baseline = 0*inp
  assert(baseline.shape == inp.shape)

  # Scale input and compute gradients.
  scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
  predictions, grads = returnGrad(scaled_inputs, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
  
#  return grads, predictions
  
  # Use trapezoidal rule to approximate the integral.
  # See Section 4 of the following paper for an accuracy comparison between
  # left, right, and trapezoidal IG approximations:
  # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
  # https://arxiv.org/abs/1908.06214
  grads = np.divide(grads[:-1] + grads[1:], 2)
  avg_grads = torch.tensor(np.average(grads, axis=0))
  integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>
  if magnitude:
      gradient_output = (integrated_gradients * integrated_gradients)
  else:
      gradient_output = integrated_gradients  
  return gradient_output, predictions


def random_baseline_integrated_gradients(
    inp, 
    target_label_index,
    steps=50,
    magnitude = True,
    num_random_trials=10):
  all_intgrads = []
#  dt = np.dtype('d')  # double-precision floating-point number
  base = torch.tensor(np.float32((1.0*np.random.random(np.array(inp.shape)))))
  for i in range(num_random_trials):
    intgrads, prediction_trend = integrated_gradients(
       inp, 
       target_label_index=target_label_index,
       baseline=base,
       steps=steps, 
       magnitude=magnitude)
    all_intgrads.append(np.array(intgrads))
  avg_intgrads = torch.tensor(np.average(np.array(all_intgrads), axis=0))
  return avg_intgrads

def returnGrad(img1, index):
    
    gradient, prediction = list(), list()
    
    for i in range(len(img1)):
#        img = torch.tensor(img1[i])
#        img.requires_grad_(True)
        img = img1[i].clone().detach().requires_grad_(True)
        pred = model(img)
        loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]))
#        loss = criterion(pred, torch.tensor([int(index)]))
        loss.backward()#retain_graph=True)
        #    S_c = torch.max(pred[0].data, 0)[0]
        Sc_dx = img.grad
        prediction.append(pred)
        gradient.append(Sc_dx.data.numpy())
        
    
    return prediction, gradient