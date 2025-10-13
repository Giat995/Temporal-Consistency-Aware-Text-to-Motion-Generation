"""Variants of the cycle-consistency loss described in TCC paper.

The Temporal Cycle-Consistency (TCC) Learning paper
(https://arxiv.org/pdf/1904.07846.pdf) describes a loss that enables learning
of self-supervised representations from sequences of embeddings that are good
at temporally fine-grained tasks like phase classification, video alignment etc.

These losses impose cycle-consistency constraints between sequences of
embeddings. Another interpretation of the cycle-consistency constraints is
that of mutual nearest-nieghbors. This means if state A in sequence 1 is the
nearest neighbor of state B in sequence 2 then it must also follow that B is the
nearest neighbor of A. We found that imposing this constraint on a dataset of
related sequences (like videos of people pitching a baseball) allows us to learn
generally useful visual representations.

This code allows the user to apply the loss while giving them the freedom to
choose the right encoder for their dataset/task. One advice for choosing an
encoder is to ensure that the encoder does not solve the mutual neighbor finding
task in a trivial fashion. For example, if one uses an LSTM or Transformer with
positional encodings, the matching between sequences may be done trivially by
counting the frame index with the encoder rather than learning good features.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from models.vq.tcc.deterministic_alignment import compute_deterministic_alignment_loss, efficient_compute_deterministic_alignment_loss
from models.vq.tcc.stochastic_alignment import compute_stochastic_alignment_loss


def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           stochastic_matching=False,
                           normalize_embeddings=False,
                           loss_type='regression_mse',
                           similarity_type='l2',
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           label_smoothing=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,
                           normalize_indices=True):

    # Get the number of timesteps in the sequence embeddings.
    num_steps = embs.size(1)

    # If steps has not been provided, assume sampling has been done uniformly.
    if steps is None:
        steps = torch.tile(torch.arange(num_steps).unsqueeze(0), (batch_size, 1))

    # If seq_lens has not been provided, assume it is equal to the size of the
    # time axis in the embeddings.
    if seq_lens is None:
        seq_lens = torch.full((batch_size,), num_steps, dtype=torch.long)

    # Check if batch size embs is consistent with provided batch size.
    assert batch_size == embs.size(0), "Batch size mismatch with embeddings."
    # Check if number of timesteps in embs is consistent with provided steps.
    assert num_steps == steps.size(1) and batch_size == steps.size(0), "Shape mismatch in steps."

    # Normalize embeddings if required.
    if normalize_embeddings:
        embs = F.normalize(embs, p=2, dim=-1)

    # Perform alignment and return loss.
    if stochastic_matching:
        loss = compute_stochastic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices)
    else:
        
        '''loss = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices)'''
        
        
        loss = efficient_compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices)
        

    return loss