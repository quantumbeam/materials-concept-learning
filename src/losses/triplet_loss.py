import warnings    
import torch
import torch.nn.functional as F
import numpy as np

def large_cdist(x, y, p=2, k=8, compute_mode='donot_use_mm_for_euclid_dist'):
    n_batch_x = x.shape[0]
    n_batch_y = n_batch_x
    n_dim = x.shape[1]
    
    f = -n_batch_x % k
    if f > 0:
        z = torch.zeros((f, n_dim), dtype=x.dtype, device=x.device)
        x = torch.cat((x, z), dim=0)
        n_batch_x += f

    x_ = x.reshape(k, n_batch_x//k, n_dim)
    y_ = y.reshape(1, n_batch_y, n_dim)
    dist = [torch.cdist(x_[i:i+1], y_, p=p, compute_mode=compute_mode) for i in range(k)]
    dist = torch.cat(dist,dim=0)
    dist = dist.reshape(n_batch_x, n_batch_y)
    if f > 0:
        dist = dist[:-f, :]
    return dist

def _distance_weighted_sampling(distance, dim, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False, **kwargs):
    n = distance.shape[0]

    mask = ~torch.eye(n, dtype=torch.bool, device=distance.device)
    if 'adaptive_nonzero_loss_cutoff' in kwargs and kwargs['adaptive_nonzero_loss_cutoff']:
        far_cutoff = (distance - distance.diag()[None]) < nonzero_loss_cutoff
    else:
        # Original implementation
        far_cutoff = distance < nonzero_loss_cutoff
    
    dclamp = distance.detach().clamp(min=cutoff)
    log_weights = ((2.0 - float(dim)) * dclamp.log() - (float(dim-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(dclamp*dclamp), min=1e-8)))

    if normalize:
        # normalization includes positive pair distances.
        # so following off-diag min max may be better.
        # offdiag = log_weights[mask]
        # minv, maxv = offdiag.min(), offdiag.max()
        minv, maxv = log_weights.min(), log_weights.max()
        log_weights = (log_weights - minv) / (maxv - minv + 1e-8)

    lwmax, _ = log_weights.max(dim=0, keepdim=True)
    weights = torch.exp(log_weights - lwmax)

    mask = mask & far_cutoff & (weights>0)
    weights[~mask] = 0

    # workaround: multinomial crashes when there is zero cols,
    # so remove cols with no valid elements
    valid_col_mask = mask.sum(dim=0) > 0
    weights = weights[:, valid_col_mask]
    inds = torch.multinomial(weights.t(), 1).t()

    # recover inds as the full size tensor
    if inds.shape[1] != n:
        inds_full = torch.zeros((1, n), dtype=inds.dtype, device=inds.device)
        inds_full[:, valid_col_mask] = inds
        inds = inds_full
    
    return  inds, valid_col_mask


def _get_negative_mask(dist_mat, sampling_policy='semihard', margin=1.0, **kwargs):
    """
    Doing negative sample mining along the vertical direction (dim0) of the pairwise dist mat of x and y. 
    The mask is "1 for elements that meet the conditions of that sampling_policy, 0 otherwise".
    """
    dist_mat = dist_mat.detach()
    # The broadcast direction here determines the axis of negative sample mining.
    # Transpose to the x-axis direction (dim0) for mining (axes with dim size of 1 are automatically broadcast).
    positive_dist = dist_mat.diag()[None]   # (1, n)

    random_idx = None

    # Set the diagonal elements of the all-ones matrix to 0 (so that the positive pair is not selected as the negative sample)
    mask = 1.0 - torch.eye(dist_mat.shape[0], dtype=dist_mat.dtype, device=dist_mat.device)
    if sampling_policy == 'semihard':
        # Mask elements that do not satisfy the semihard negative condition (i.e. replace by inf)
        # Since the product of following conditions is necessary and sufficient condition for semihard negative, invert the bool and mask with inf.
        bool1 = dist_mat < positive_dist + margin
        bool2 = positive_dist <= dist_mat
        mask[~(bool1 & bool2)] = 0
        
    elif sampling_policy == 'semiharder':
        positive_dist = positive_dist.t()
        bool1 = dist_mat < positive_dist + margin
        bool2 = positive_dist <= dist_mat
        mask[~(bool1 & bool2)] = 0
        
    elif sampling_policy == 'semiharder+':
        positive_dist = positive_dist.t()
        bool1 = dist_mat < positive_dist + margin
        bool2 = positive_dist <= dist_mat
        mask[~(bool1 & bool2)] = 0

        # exclude easy (zero loss) samples
        positive_dist = positive_dist.t()
        mask[(dist_mat >= positive_dist + margin)] = 0
        
    elif sampling_policy == 'hard':
        # It is a necessary and sufficient condition for hard negative that the following is positive.
        # The bool are inverted and masked by inf (same as below).
        # これが正なのがhard negativeの必要十分条件なので、条件満たさないものをinfでマスクする（
        mask[~(positive_dist - dist_mat > 0)] = 0
        
    elif sampling_policy == 'noneasy':
        # It is a necessary and sufficient condition for non-easy negative that the following is positive.
        mask[(dist_mat >= positive_dist + margin)] = 0
        
    elif sampling_policy == 'random':
        pass
        
    elif sampling_policy == 'easy':
        # It is a necessary and sufficient condition for easy negative that the following is positive.
        mask[~(dist_mat - (positive_dist + margin) >= 0)] = 0
        
    elif sampling_policy == 'dweighted':
        random_idx, valid_col_mask = _distance_weighted_sampling(dist_mat, **kwargs)
        
    else:
        raise NameError

    if random_idx is None:
        random_mat = torch.rand_like(mask)
        random_idx = torch.argmax(mask * random_mat, dim=0, keepdim=True)
        valid_col_mask = mask.sum(dim=0) > 0

    return random_idx, valid_col_mask

def _bi_directional_loss(x_embedding, y_embedding, sampling_policy='semihard',  margin=1.0, **kwargs):
    """
    Calculate bi-directional triplet loss (Lxy) along dim0(=x) from pairwise distance of embedding and mask of negative pair
    """
    # pairwise_dist_mat = calc_pairwise_distances(x_embedding=x_embedding, y_embedding=y_embedding, squared=False)
    pairwise_dist_mat = large_cdist(x_embedding, y_embedding, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    random_idx, valid_col_mask = _get_negative_mask(pairwise_dist_mat, margin=margin, sampling_policy=sampling_policy, **kwargs)
        
    # Randomly select one negative pair in dim0 that satisfies the sampling policy (one from each column)
    sampled_negative_pair_dist = pairwise_dist_mat.gather(0, random_idx).squeeze()

    # loss calculation
    x_minus_y = pairwise_dist_mat.diag()[valid_col_mask]
    x_minus_y_dash = sampled_negative_pair_dist[valid_col_mask]

    loss = (x_minus_y - x_minus_y_dash + margin).clamp_min(0)
    return torch.mean(loss)

def _symmetric_bidirectional_loss(x_embedding, y_embedding, sampling_policy, margin=1, **kwargs):
    """
    
    """
    # Dxy = calc_pairwise_distances(x_embedding=x_embedding, y_embedding=y_embedding, squared=False)
    Dxy = large_cdist(x_embedding, y_embedding, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    # Dyy = calc_pairwise_distances(x_embedding=y_embedding, y_embedding=y_embedding, squared=False)
    Dyy = large_cdist(y_embedding, y_embedding, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    is_diag_mask = torch.diag(torch.eye(x_embedding.shape[0], device=x_embedding.device)).type(torch.bool)
    D = torch.where(is_diag_mask, Dxy, (Dxy+Dyy)/2.0)

    sampling_target = kwargs['sampling_target'] if 'sampling_target' in kwargs else 'D'
    if sampling_target == 'D':
        random_idx, valid_col_mask = _get_negative_mask(D, margin=margin, sampling_policy=sampling_policy, **kwargs)
    elif sampling_target == 'Dxy':
        random_idx, valid_col_mask = _get_negative_mask(Dxy, margin=margin, sampling_policy=sampling_policy, **kwargs)
    else:
        raise NameError

    # Randomly select one negative pair in dim0 that satisfies the sampling policy (one from each column)
    sampled_negative_pair_dist = D.gather(0, random_idx).squeeze()

    # loss calculation
    x_minus_y = D.diag()[valid_col_mask]
    second_term = sampled_negative_pair_dist[valid_col_mask]

    loss = (x_minus_y - second_term + margin).clamp_min(0)
    return torch.mean(loss)

def multimodal_triplet_loss(x_embedding, y_embedding, params):
    """
    bi-directional triplet lossを計算する。
    具体的には次のLy, Lxについて平均したもの。
    Ly(x, y, y') = max(0, ||x-y|| - ||x-y'|| + m)
    Lx(x, y, x') = max(0, ||x-y|| - ||x'-y|| + m)
    """
    
    margin = params.margin
    policy = params.sampling_policy

    kwargs = {
        # distance-weighted sampling options
        'dim': x_embedding.shape[1],
        'nonzero_loss_cutoff': margin,
        'adaptive_nonzero_loss_cutoff': True,
        'normalize': True,

        # symmetric loss options
        'sampling_target': params.sampling_target,
    }

    if params.loss_mode == 'bi_directional':
        Ly = _bi_directional_loss(x_embedding, y_embedding, policy, margin, **kwargs)
        Lx = _bi_directional_loss(y_embedding, x_embedding, policy, margin, **kwargs)
        loss = (Lx + Ly)/2

    elif params.loss_mode == 'symmetric_bi_directional':
        Lsy = _symmetric_bidirectional_loss(x_embedding, y_embedding, policy, margin, **kwargs)
        Lsx = _symmetric_bidirectional_loss(y_embedding, x_embedding, policy, margin, **kwargs)
        loss = (Lsx + Lsy)/2
    else:
        raise NameError

    if hasattr(params, 'improved_triplet_weight') and params.improved_triplet_weight > 0:
        # Implements the improved triplet loss proposed in
        # Chang+, Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function (CVPR 2016).
        # Default params: margin1 = 1, margin2 (below) = 0.01, weight = 0.002.
        loss2 = torch.norm(x_embedding - y_embedding, p=2, dim=1) - params.improved_triplet_margin
        loss2 = loss2.clamp_min(0).mean()*params.improved_triplet_weight
        loss = loss + loss2

    return loss
