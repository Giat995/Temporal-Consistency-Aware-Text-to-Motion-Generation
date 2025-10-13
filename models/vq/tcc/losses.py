from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


"""Loss function based on classifying the correct indices.

  In the paper, this is called Cycle-back Classification.

  Args:
    logits: Tensor, Pre-softmax scores used for classification loss. These are
      similarity scores after cycling back to the starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    label_smoothing: Float, label smoothing factor which can be used to
      determine how hard the alignment should be.
  Returns:
    loss: Tensor, A scalar classification loss calculated using standard softmax
      cross-entropy loss.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
def classification_loss(logits, labels, label_smoothing):
    
    num_classes = logits.size(1)
    labels = labels.float().to(logits.device).detach()
    
    loss = F.cross_entropy(logits, labels, reduction='mean', label_smoothing=label_smoothing)
    return loss

"""Loss function based on regressing to the correct indices.

  In the paper, this is called Cycle-back Regression. There are 3 variants
  of this loss:
  i) regression_mse: MSE of the predicted indices and ground truth indices.
  ii) regression_mse_var: MSE of the predicted indices that takes into account
  the variance of the similarities. This is important when the rate at which
  sequences go through different phases changes a lot. The variance scaling
  allows dynamic weighting of the MSE loss based on the similarities.
  iii) regression_huber: Huber loss between the predicted indices and ground
  truth indices.


  Args:
    logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional temporal information to the alignment loss.
    loss_type: String, This specifies the kind of regression loss function.
      Currently supported loss functions: regression_mse, regression_mse_var,
      regression_huber.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities don't arise as sequence
      indices can be large numbers.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.

  Returns:
     loss: Tensor, A scalar loss calculated using a variant of regression.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type,
                    normalize_indices, variance_lambda, huber_delta):
    labels = labels.detach()
    steps = steps.detach()

    if normalize_indices:
        float_seq_lens = seq_lens.float()
        tile_seq_lens = float_seq_lens.unsqueeze(1).expand(-1, num_steps)
        steps = steps.float() / tile_seq_lens
    else:
        steps = steps.float()

    beta = F.softmax(logits, dim=-1)
    true_time = (steps * labels).sum(dim=1)
    pred_time = (steps * beta).sum(dim=1)

    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            # Variance aware regression.
            pred_time_tiled = pred_time.unsqueeze(1).expand(-1, num_steps)

            pred_time_variance = ((steps - pred_time_tiled) ** 2 * beta).sum(dim=1)

            # Using log of variance for numerical stability.
            pred_time_log_var = torch.log(pred_time_variance + 1e-8)  # Add small value for stability
            squared_error = (true_time - pred_time) ** 2
            return torch.mean(torch.exp(-pred_time_log_var) * squared_error +
                              variance_lambda * pred_time_log_var)
        else:
            return F.mse_loss(true_time, pred_time)
    elif loss_type == 'regression_huber':
        return F.huber_loss(pred_time, true_time, delta=huber_delta)
    else:
        raise ValueError(f'Unsupported regression loss {loss_type}. Supported losses are: '
                         'regression_mse, regression_mse_var, and regression_huber.')