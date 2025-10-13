"""Stochastic alignment between sampled cycles in the sequences in a batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from models.vq.tcc.losses import classification_loss
from models.vq.tcc.losses import regression_loss

def _align_single_cycle(cycle, embs, cycle_length, num_steps, similarity_type, temperature):
    """Takes a single cycle and returns logits (similarity scores) and labels."""
    # Choose random frame.
    n_idx = torch.randint(0, num_steps, (1,)).item()  # Random integer between 0 and num_steps
    # Create labels
    onehot_labels = F.one_hot(torch.tensor(n_idx), num_steps)

    # Choose query features for the first frame.
    query_feats = embs[cycle[0], n_idx:n_idx + 1]

    num_channels = query_feats.size(-1)
    
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]

        if similarity_type == 'l2':
            # Find L2 distance.
            mean_squared_distance = torch.sum((query_feats.expand(num_steps, -1) - candidate_feats) ** 2, dim=1)
            # Convert L2 distance to similarity.
            similarity = -mean_squared_distance

        elif similarity_type == 'cosine':
            # Dot product of embeddings.
            similarity = torch.matmul(candidate_feats, query_feats.t()).squeeze()
        else:
            raise ValueError('similarity_type can either be l2 or cosine.')

        # Scale the distance by the number of channels.
        similarity /= num_channels
        # Scale the distance by a temperature.
        similarity /= temperature

        beta = F.softmax(similarity, dim=0)
        beta = beta.unsqueeze(1)

        # Find weighted nearest neighbour.
        query_feats = torch.sum(beta * candidate_feats, dim=0, keepdim=True)

    return similarity, onehot_labels.float()

def _align(cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature):
    """Align by finding cycles in embeddings."""
    logits_list = []
    labels_list = []
    
    for i in range(num_cycles):
        logits, labels = _align_single_cycle(cycles[i], embs, cycle_length, num_steps, similarity_type, temperature)
        logits_list.append(logits)
        labels_list.append(labels)

    logits = torch.stack(logits_list)
    labels = torch.stack(labels_list).to(logits.device)

    return logits, labels

def gen_cycles(num_cycles, batch_size, cycle_length=2):
    """Generates cycles for alignment.

    Generates a batch of indices to cycle over. For example setting num_cycles=2,
    batch_size=5, cycle_length=3 might return something like this:
    cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
    the loss will be calculated. The first cycle starts at sequence 0 of the
    batch, then we find a matching step in sequence 3 of that batch, then we
    find matching step in sequence 4 and finally come back to sequence 0,
    completing a cycle.

    Args:
        num_cycles: Integer, Number of cycles that will be matched in one pass.
        batch_size: Integer, Number of sequences in one batch.
        cycle_length: Integer, Length of the cycles. If we are matching between
          2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
          This means that we go from sequence 0 to sequence 1 then back to sequence
          0. A cycle length of 3 might look like [0, 1, 2, 0].

    Returns:
        cycles: Tensor, Batch indices denoting cycles that will be used for
        calculating the alignment loss.
    """
    sorted_idxes = torch.arange(batch_size).repeat(num_cycles, 1)  # Create a tensor of indices
    cycles = sorted_idxes[torch.randperm(num_cycles)]  # Shuffle the indices

    cycles = cycles[:, :cycle_length]  # Select the first cycle_length indices
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)  # Append the first index to complete the cycle

    return cycles

def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
                                      num_steps,
                                      batch_size,
                                      loss_type,
                                      similarity_type,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      label_smoothing,
                                      variance_lambda,
                                      huber_delta,
                                      normalize_indices):
    """Compute cycle-consistency loss by stochastically sampling cycles.

    Args:
        embs: Tensor, sequential embeddings of the shape [N, T, D].
        steps: Tensor, step indices of the embeddings of the shape [N, T].
        seq_lens: Tensor, lengths of the sequences.
        num_steps: Integer, number of timesteps in the embeddings.
        batch_size: Integer, batch size.
        loss_type: String, type of loss function ('classification', 'regression_mse', etc.).
        similarity_type: String, similarity metrics ('l2', 'cosine').
        num_cycles: Integer, number of cycles to match.
        cycle_length: Integer, length of the cycle.
        temperature: Float, temperature scaling for softmax.
        label_smoothing: Float, label smoothing parameter.
        variance_lambda: Float, weight of the variance of the similarity predictions.
        huber_delta: Float, Huber delta for Huber loss.
        normalize_indices: Boolean, if True, normalizes indices by sequence lengths.

    Returns:
        loss: Tensor, scalar loss tensor for cycle-consistency loss.
    """
    # Generate cycles
    cycles = gen_cycles(num_cycles, batch_size, cycle_length)

    logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                            similarity_type, temperature)

    if loss_type == 'classification':
        loss = classification_loss(logits, labels, label_smoothing)
    elif 'regression' in loss_type:
        steps = steps[cycles[:, 0]].to(logits.device)  # Gather steps based on cycles
        seq_lens = seq_lens[cycles[:, 0]].to(logits.device)  # Gather sequence lengths based on cycles
        loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                               loss_type, normalize_indices, variance_lambda,
                               huber_delta)
    else:
        raise ValueError(f'Unidentified loss type {loss_type}. Currently supported loss types are: '
                         'regression_mse, regression_huber, classification.')

    return loss