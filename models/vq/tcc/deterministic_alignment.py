"""Deterministic alignment between all pairs of sequences in a batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from models.vq.tcc.losses import classification_loss
from models.vq.tcc.losses import regression_loss


def pairwise_l2_distance(embs1, embs2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    norm1 = torch.sum(embs1 ** 2, dim=1).view(-1, 1)
    norm2 = torch.sum(embs2 ** 2, dim=1).view(1, -1)

    # Max to ensure matmul doesn't produce anything negative due to floating
    # point approximations.
    dist = torch.maximum(norm1 + norm2 - 2.0 * torch.matmul(embs1, embs2.T), torch.tensor(0.0))

    return dist

def fc_pairwise_l2_distance(embs1, embs2, exclude_diag=True):
    """
    Computes pairwise distances between all rows of embs1 and embs2 for all combinations of batches.

    Args:
        embs1: Tensor of shape [B, M, D]
        embs2: Tensor of shape [B, N, D]

    Returns:
        dist: Tensor of shape [B, B, M, N]
    """
    # Compute squared norms for embs1 and embs2
    norm1 = torch.sum(embs1 ** 2, dim=2, keepdim=True)  # Shape: [B, M, 1]
    norm2 = torch.sum(embs2 ** 2, dim=2, keepdim=True)  # Shape: [B, N, 1]
    
    # Reshape norm2 to align with broadcasting for all pairs of batches
    norm2 = norm2.permute(0, 2, 1)  # Shape: [B, 1, N]

    # Expand dimensions to enable pairwise computation across batch pairs
    embs1_exp = embs1.unsqueeze(1)  # Shape: [B, 1, M, D]
    embs2_exp = embs2.unsqueeze(0)  # Shape: [1, B, N, D]

    # Compute pairwise dot products for all batch combinations
    dot_product = torch.matmul(embs1_exp, embs2_exp.transpose(2, 3))  # Shape: [B, B, M, N]

    # Compute pairwise distances
    dist = torch.maximum(
        norm1.unsqueeze(1) + norm2.unsqueeze(0) - 2.0 * dot_product,
        torch.tensor(0.0, device=embs1.device)
    )

    # Exclude diagonal elements
    if exclude_diag:
        dist = exclude_diagonal(matrix=dist) # [B*(B-1), T, T]
    
    return dist

def exclude_diagonal(matrix):
    """
    Excludes diagonal elements along the [B, B] dimensions of the input tensor.

    Args:
        matrix: Tensor of shape [B, B, M, N]

    Returns:
        result: Tensor of shape [B * (B - 1), M, N]
    """
    B, _, M, N = matrix.shape
    
    # Create a mask to exclude diagonal elements
    mask = ~torch.eye(B, dtype=torch.bool, device=matrix.device)  # Shape: [B, B]
    
    # Apply the mask and reshape
    filtered_matrix = matrix[mask].view(B * (B - 1), M, N)  # Flattening along [B * (B-1), M, N]
    
    return filtered_matrix

def batched_pairwise_l2_distance(embs1, embs2):
    """
    Computes pairwise L2 distances between all rows of embs1 and embs2 for each batch.
    
    Args:
        embs1: Tensor of shape [B, M, D].
        embs2: Tensor of shape [B, N, D].
        
    Returns:
        dist: Tensor of shape [B, M, N], where each slice along dimension B contains 
              the pairwise L2 distances between rows of embs1 and embs2.
    """
    # Compute the squared norms of embs1 and embs2 for each batch
    norm1 = torch.sum(embs1 ** 2, dim=2, keepdim=True)  # Shape [B, M, 1]
    norm2 = torch.sum(embs2 ** 2, dim=2, keepdim=True)  # Shape [B, N, 1]
    
    # Compute pairwise distances using broadcasting
    dist = norm1 + norm2.transpose(1, 2) - 2.0 * torch.bmm(embs1, embs2.transpose(1, 2))
    
    # Ensure no negative values due to floating point approximations
    dist = torch.maximum(dist, torch.tensor(0.0, device=dist.device))
    
    return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    """Returns similarity between all rows of embs1 and all rows of embs2.

    The similarity is scaled by the number of channels/embedding size and
    temperature.
    
    Args:
        embs1: Tensor, Embeddings of shape [M, D] where M is the number of
            embeddings and D is the embedding size.
        embs2: Tensor, Embeddings of shape [N, D] where N is the number of
            embeddings and D is the embedding size.
        similarity_type: String, Either 'l2' or 'cosine'.
        temperature: Float, Temperature used in scaling logits before softmax.

    Returns:
        similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
        embs2.
    """
    channels = torch.tensor(embs1.size(1), dtype=torch.float32) 
    
    if similarity_type == 'cosine':
        similarity = torch.matmul(embs1, embs2.T)
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance by the number of channels.
    similarity /= channels
    # Scale the distance by the temperature.
    similarity /= temperature

    return similarity

def fc_get_scaled_similarity(embs1, embs2, similarity_type, temperature, exclude_diag=True):
    """Returns similarity between all rows of embs1 and all rows of embs2.

        The similarity is scaled by the number of channels/embedding size and
        temperature.
        
        Args:
            embs1: Tensor, Embeddings of shape [B, M, D] where B is the batch size, M is the number of
                embeddings and D is the embedding size.
            embs2: Tensor, Embeddings of shape [B, N, D] where B is the batch size, N is the number of
                embeddings and D is the embedding size.
            similarity_type: String, Either 'l2' or 'cosine'.
            temperature: Float, Temperature used in scaling logits before softmax.

        Returns:
            similarity: Tensor, [B*(B-1), M, N] tensor if exclude_diag=True, else [B, B, M, N] tensor,
              denoting similarity between embs1 and embs2.
        """    
    channels = torch.tensor(embs1.size(2), dtype=torch.float32) 
    
    if similarity_type == 'cosine':
        similarity = fc_similarity(embs1, embs2, exclude_diag=exclude_diag) 
    elif similarity_type == 'l2':
        similarity = -1.0 * fc_pairwise_l2_distance(embs1, embs2, exclude_diag=exclude_diag)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance by the number of channels.
    similarity /= channels
    # Scale the distance by the temperature.
    similarity /= temperature

    return similarity

def fc_similarity(embs1, embs2, exclude_diag=True):
    """
    Computes similarity of shape [B, B, M, N] for batched inputs.
    
    Args:
        embs1: Tensor of shape [B, M, D]
        embs2: Tensor of shape [B, N, D]
    
    Returns:
        similarity: Tensor of shape [B*(B-1), M, N] if exclude_diag=True, else shape [B, B, M, N] 
    """
    # Expand embs1 along a new batch dimension for all-to-all batch comparison
    embs1_exp = embs1.unsqueeze(1)  # Shape: [B, 1, M, D]
    embs2_exp = embs2.unsqueeze(0)  # Shape: [1, B, N, D]

    # Compute similarity using batch matrix multiplication
    similarity = torch.matmul(embs1_exp, embs2_exp.transpose(2, 3))  # Shape: [B, B, M, N]

    # Exclude diagonal elements
    if exclude_diag:
        similarity = exclude_diagonal(matrix=similarity) # [B*(B-1), T, T]

    return similarity

def batched_get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    """Returns similarity between corresponding rows of embs1 and embs2.

    The similarity is scaled by the number of channels/embedding size and
    temperature.
    
    Args:
        embs1: Tensor, Embeddings of shape [B, M, D] where M is the number of
            embeddings and D is the embedding size.
        embs2: Tensor, Embeddings of shape [B, N, D] where N is the number of
            embeddings and D is the embedding size.
        similarity_type: String, Either 'l2' or 'cosine'.
        temperature: Float, Temperature used in scaling logits before softmax.

    Returns:
        similarity: Tensor, [B, M, N] tensor denoting similarity between embs1 and
        embs2.
    """
    channels = torch.tensor(embs1.size(2), dtype=torch.float32) 
    
    if similarity_type == 'cosine':
        similarity = torch.bmm(embs1, embs2.transpose(1, 2))
    elif similarity_type == 'l2':
        similarity = -1.0 * batched_pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance by the number of channels.
    similarity /= channels
    # Scale the distance by the temperature.
    similarity /= temperature

    return similarity


def align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
    """Align a given pair embedding sequences.

    Args:
        embs1: Tensor, Embeddings of shape [M, D] where M is the number of
            embeddings and D is the embedding size.
        embs2: Tensor, Embeddings of shape [N, D] where N is the number of
            embeddings and D is the embedding size.
        similarity_type: String, Either one of 'l2' or 'cosine'.
        temperature: Float, Temperature used in scaling logits before softmax.
        
    Returns:
        logits: Tensor [M, M], Pre-softmax similarity scores after cycling back to the
            starting sequence.
        labels: Tensor [M, M], One hot labels containing the ground truth. The index where
            the cycle started is 1.
    """
    max_num_steps = embs1.size(0) # M

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature) # [M,D], [N,D] -> [M, N]
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, dim=1) # [M, N]

    # Calculate soft-nearest neighbors.
    nn_embs = torch.matmul(softmaxed_sim_12, embs2) # [M, N], [N, D] -> [M, D]

    # Find distances between nn_embs and embs1.
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature) # [M,D], [M,D] -> [M, M]

    logits = sim_21 # [M, M]
    labels = F.one_hot(torch.arange(max_num_steps), num_classes=max_num_steps).to(logits.device) # [M, M]

    return logits, labels.float()


def efficient_align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
    """Align a given pair embedding sequences.

    Args:
        embs1: Tensor, Embeddings of shape [B, M, D] where M is the number of
            embeddings and D is the embedding size.
        embs2: Tensor, Embeddings of shape [B, N, D] where N is the number of
            embeddings and D is the embedding size.
        similarity_type: String, Either one of 'l2' or 'cosine'.
        temperature: Float, Temperature used in scaling logits before softmax.
        
    Returns:
        logits: Tensor [B*(B-1), M, M], Pre-softmax similarity scores after cycling back to the
            starting sequence.
        labels: Tensor [B*(B-1), M, M], One hot labels containing the ground truth. The index where
            the cycle started is 1.
    """
    B, M, D = embs1.shape
    _, N, _ = embs2.shape

    # Find distances between embs1 and embs2.
    sim_12 = fc_get_scaled_similarity(embs1, embs2, similarity_type, temperature, exclude_diag=False) # [B,M,D], [B,N,D] -> [B, B, M, N]
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, dim=-1) # [B, B, M, N]

    # Calculate soft-nearest neighbors.
    # [B*B, M, N], [B*B, N, D] -> [B*B, M, D]
    softmaxed_sim_12_reshape = softmaxed_sim_12.view(-1, M, N)
    embs2_reshape = embs2.repeat(B, 1, 1) # B samples; B samples; ...
    nn_embs = torch.bmm(softmaxed_sim_12_reshape, embs2_reshape)  # [B*B, M, D]

    # Find distances between nn_embs and embs1. [B,M,D] [B,N,D] -> [B, B, M, N]
    # [B*B, M, D], [B*B, M, D] -> [B*B, M, M]
    embs1_reshape = embs1.unsqueeze(1).repeat(1, B, 1, 1).view(-1, M, D) # sample 1 * B; sample 2 * B; ... ; sample B * B
    sim_21 = batched_get_scaled_similarity(nn_embs, embs1_reshape, similarity_type, temperature)  # [B*B, M, M]
    sim_21_reshape = sim_21.view(B, B, M, M)
    sim_21_filtered = exclude_diagonal(matrix=sim_21_reshape)

    logits = sim_21_filtered.view(-1, M) # [B*(B-1),M, M] -> [B*(B-1)*M, M]
    labels = F.one_hot(torch.arange(M), num_classes=M).repeat(B * (B - 1), 1).float()  # [B*(B-1), M, M] -> [B*(B-1)*M, M]

    return logits, labels.float()


def compute_deterministic_alignment_loss(embs, steps, seq_lens, num_steps, batch_size,
                                         loss_type, similarity_type, temperature,
                                         label_smoothing, variance_lambda,
                                         huber_delta, normalize_indices):
    
    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            # We do not align the sequence with itself.
            if i != j:
                logits, labels = align_pair_of_sequences(embs[i], embs[j], similarity_type, temperature)
                logits_list.append(logits) # list of [T,T]
                labels_list.append(labels) # list of [T,T]
                steps_list.append(steps[i].unsqueeze(0).repeat(num_steps, 1)) # list of [num_steps, T]
                seq_lens_list.append(seq_lens[i].unsqueeze(0).repeat(num_steps)) # list of [num_steps,]

    logits = torch.cat(logits_list, dim=0) # [B*(B-1)*T, T]
    labels = torch.cat(labels_list, dim=0) # [B*(B-1)*T, T]
    steps = torch.cat(steps_list, dim=0).to(logits.device) # [B*(B-1)*num_steps, T]
    seq_lens = torch.cat(seq_lens_list, dim=0).to(logits.device) # [B*(B-1)*num_steps, ]

    if loss_type == 'classification':
        loss = classification_loss(logits, labels, label_smoothing)
    elif 'regression' in loss_type:
        loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                               loss_type, normalize_indices, variance_lambda,
                               huber_delta)
    else:
        raise ValueError(f'Unidentified loss_type {loss_type}. Currently supported loss '
                         'types are: regression_mse, regression_huber, '
                         'classification.')

    return loss

def efficient_compute_deterministic_alignment_loss(embs, steps, seq_lens, num_steps, batch_size,
                                         loss_type, similarity_type, temperature,
                                         label_smoothing, variance_lambda,
                                         huber_delta, normalize_indices):
    
    logits, labels = efficient_align_pair_of_sequences(embs1=embs, embs2=embs, similarity_type=similarity_type, temperature=temperature)

    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        steps_list.extend([steps[i].unsqueeze(0).repeat(num_steps, 1)]*(batch_size-1)) # list of [num_steps, T]
        seq_lens_list.extend([seq_lens[i].unsqueeze(0).repeat(num_steps)]*(batch_size-1)) # list of [num_steps,]

    steps = torch.cat(steps_list, dim=0).to(logits.device) # [B*(B-1)*num_steps, T]
    seq_lens = torch.cat(seq_lens_list, dim=0).to(logits.device) # [B*(B-1)*num_steps, ]

    if loss_type == 'classification':
        loss = classification_loss(logits, labels, label_smoothing)
    elif 'regression' in loss_type:
        loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                               loss_type, normalize_indices, variance_lambda,
                               huber_delta)
    else:
        raise ValueError(f'Unidentified loss_type {loss_type}. Currently supported loss '
                         'types are: regression_mse, regression_huber, '
                         'classification.')

    return loss


def test_cda_loss():
    N = 2
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    steps = torch.randint(low=1, high=10, size=(N, T))
    seq_lens = torch.tensor([10, 10])
    num_steps = 5
    loss_type = 'classification'
    similarity_type = 'l2'
    temperature = 1.0
    label_smoothing = 0.5
    variance_lambda = 0.5
    huber_delta = 1.0
    normalize_indices = True

    loss = compute_deterministic_alignment_loss(embs=embs, steps=steps, seq_lens=seq_lens,
                                                num_steps=num_steps, batch_size=N, loss_type=loss_type,
                                                similarity_type=similarity_type, temperature=temperature,
                                                label_smoothing=label_smoothing, variance_lambda=variance_lambda,
                                                huber_delta=huber_delta, normalize_indices=normalize_indices)
    print("loss", loss)


def test_aps():
    M = 6
    D = 5
    N = 4

    embs1 = torch.rand(M, D)
    embs2 = torch.rand(N, D)

    similarity_type = 'l2'
    temperature = 1.0

    logits, labels = align_pair_of_sequences(embs1=embs1, embs2=embs2, similarity_type=similarity_type, temperature=temperature)

    print("logits", logits.size()) # [M, M]
    print("labels", labels.size()) # [M, M]


def test_aps_batched():
    B = 48
    M = 6
    D = 5

    embs = torch.rand(B, M, D)

    similarity_type = 'l2'
    temperature = 1.0

    labels_list = []
    logits_list = []

    for i in range(B):
        for j in range(B):
            # We do not align the sequence with itself.
            if i != j:
                logits, labels = align_pair_of_sequences(embs[i], embs[j], similarity_type, temperature)
                logits_list.append(logits) # list of [T,T]
                labels_list.append(labels) # list of [T,T]

    logits = torch.cat(logits_list, dim=0) # [B*(B-1)*T, T]
    labels = torch.cat(labels_list, dim=0) # [B*(B-1)*T, T]

    logits_batched, labels_batched = efficient_align_pair_of_sequences(embs1=embs, embs2=embs, similarity_type=similarity_type, temperature=temperature)

    print("logits", torch.equal(logits, logits_batched)) 
    print("labels", torch.equal(labels, labels_batched)) 


def test_fc_pl2d():
    N = 256
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    dist_list = []
    for i in range(N):
        for j in range(N):
            dist = pairwise_l2_distance(embs1=embs[i], embs2=embs[j])
            dist_list.append(dist)

    dists = torch.stack(dist_list, dim=0) # [N*N, T, T]

    dists_batched = fc_pairwise_l2_distance(embs1=embs, embs2=embs, exclude_diag=False) # [N, N, T, T]
    dists_batched_reshaped = dists_batched.reshape(-1, T, T)

    print(torch.equal(dists, dists_batched_reshaped)) # [N*N, T, T]

    # exclude diagonal elements
    dist_list = []
    for i in range(N):
        for j in range(N):
            if j!=i:
                dist = pairwise_l2_distance(embs1=embs[i], embs2=embs[j])
                dist_list.append(dist)

    dists = torch.stack(dist_list, dim=0) # [B*(B-1), T, T]

    dists_batched_exclude = fc_pairwise_l2_distance(embs1=embs, embs2=embs) # [B*(B-1), T, T]

    print(torch.equal(dists, dists_batched_exclude)) # [N*N, T, T]


def test_bt_pl2d():
    N = 6
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    dist_list = []
    for i in range(N):
        dist = pairwise_l2_distance(embs1=embs[i], embs2=embs[i])
        dist_list.append(dist)

    dists = torch.stack(dist_list, dim=0) # [N, T, T]

    dists_batched = batched_pairwise_l2_distance(embs1=embs, embs2=embs) # [N, T, T]

    print(torch.equal(dists, dists_batched)) # [N, T, T]


def test_fc_sim():
    N = 6
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    sim_list = []
    for i in range(N):
        for j in range(N):
            sim = torch.matmul(embs[i], embs[j].T)
            sim_list.append(sim)

    sims = torch.stack(sim_list, dim=0) # [N*N, T, T]

    sims_batched = fc_similarity(embs1=embs, embs2=embs, exclude_diag=False) # [N, N, T, T]
    sims_batched_reshaped = sims_batched.reshape(-1, T, T)

    print(torch.equal(sims, sims_batched_reshaped)) # [N*N, T, T]

    # exclude diagonal elements
    sim_list = []
    for i in range(N):
        for j in range(N):
            if j!=i:
                sim = torch.matmul(embs[i], embs[j].T)
                sim_list.append(sim)

    sims = torch.stack(sim_list, dim=0) # [B*(B-1), T, T]

    sims_batched_exclude = fc_similarity(embs1=embs, embs2=embs) # [B*(B-1), T, T]

    print(torch.equal(sims, sims_batched_exclude)) # [N*N, T, T]


def test_bt_sim():
    N = 12
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    sim_list = []
    for i in range(N):
        sim = torch.matmul(embs[i], embs[i].T)
        sim_list.append(sim)

    sims = torch.stack(sim_list, dim=0) # [N, T, T]

    sims_batched = torch.bmm(embs, embs.transpose(1, 2))

    print(torch.equal(sims, sims_batched)) # [N, T, T]


if __name__ == '__main__':
    ####### TESTING ########
    import time

    N = 256
    T = 4
    D = 5

    embs = torch.rand(N, T, D)
    steps = torch.randint(low=1, high=10, size=(N, T))
    seq_lens = torch.randint(low=10, high=20, size=(N,))
    num_steps = T
    # loss_type = 'regression_huber'
    # loss_type = 'regression_mse'
    loss_type = 'regression_mse_var'
    # loss_type = 'classification'
    similarity_type = 'l2'
    temperature = 1.0
    label_smoothing = 0.5
    variance_lambda = 0.5
    huber_delta = 1.0
    normalize_indices = True

    start_time = time.time()
    loss = compute_deterministic_alignment_loss(embs=embs, steps=steps, seq_lens=seq_lens,
                                                num_steps=num_steps, batch_size=N, loss_type=loss_type,
                                                similarity_type=similarity_type, temperature=temperature,
                                                label_smoothing=label_smoothing, variance_lambda=variance_lambda,
                                                huber_delta=huber_delta, normalize_indices=normalize_indices)
    end_time = time.time()
    print(f"Running time: {end_time - start_time:.6f} seconds")

    start_time = time.time()
    loss_ef = efficient_compute_deterministic_alignment_loss(embs=embs, steps=steps, seq_lens=seq_lens,
                                                num_steps=num_steps, batch_size=N, loss_type=loss_type,
                                                similarity_type=similarity_type, temperature=temperature,
                                                label_smoothing=label_smoothing, variance_lambda=variance_lambda,
                                                huber_delta=huber_delta, normalize_indices=normalize_indices)
    end_time = time.time()
    print(f"Running time: {end_time - start_time:.6f} seconds")

    print("loss", loss)
    print("loss_ef", loss_ef)
    print("loss equals loss_ef?", torch.equal(loss, loss_ef))