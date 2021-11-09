"""
Inference routines from Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class CRW(object):

    """Propagation algorithm"""
    def __init__(self, cfg):
        self.n_context = cfg.CXT_SIZE
        self.radius = cfg.RADIUS
        self.temperature = cfg.TEMP
        self.topk = cfg.KNN

        print("Inference Opts:")
        print("Context size: {}".format(self.n_context))
        print("      Radius: {}".format(self.radius))
        print("        Temp: {}".format(self.temperature))
        print("        TopK: {}".format(self.topk))

        # always keeping the first frame
        # TODO: move to cfg
        self.long_mem = [0]
        # for bwd-compatibility
        self.norm_mask = False

        self.mask = None
        self.mask_hw = None

    def _prep_context(self, feats, lbls, hw):
        """Adjust for context
        Args:
            lbls: [N,M,H,W]
        """
        lbls = F.interpolate(lbls, hw, mode="bilinear", align_corners=True)

        ref = lbls[:1].expand(self.n_context,-1,-1,-1)
        lbls = torch.cat([ref, lbls], 0)

        fref = feats[:1].expand(self.n_context,-1,-1,-1)
        fref = torch.cat([fref, feats], 0)

        return fref, lbls

    def forward(self, feats, lbls):
        """Propagate features

        Args:
            feats: [N,K,h,w]
            lbls: [N,M,H,W]
        """
        N,K,h,w = feats.shape
        M,H,W = lbls.shape[-3:]
        feats, lbls = self._prep_context(feats, lbls, (h,w))

        # [N,K,h,w] -> [1,K,N,h,w]
        feats = feats.permute([1,0,2,3])
        # singleton for compatibility
        feats = feats[None,...]

        # [BC+N,M,h,w] -> [BC+N,h,w,M]
        lbls = lbls.permute([0,2,3,1])

        n_context = self.n_context

        torch.cuda.empty_cache()

        # Prepare source (keys) and target (query) frame features
        key_indices = context_index_bank(n_context, self.long_mem, N)
        key_indices = torch.cat(key_indices, dim=-1)
        keys = feats[:, :, key_indices]
        query = feats[:, :, n_context:]

        # Make spatial radius mask TODO use torch.sparse
        if self.mask is None or self.mask_hw != (h, w):
            restrict = MaskedAttention(self.radius, flat=False)
            D = restrict.mask(h, w)[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0
            self.mask = D.cuda()
            self.mask_hw = (h, w)

        # Flatten source frame features to make context feature set
        keys, query = keys.flatten(-2), query.flatten(-2)

        Ws, Is = mem_efficient_batched_affinity(query, keys, self.mask,  \
                self.temperature, self.topk, self.long_mem)
        
        ##################################################################
        # Propagate Labels and Save Predictions
        ###################################################################

        masks_idx = torch.LongTensor(N,H,W)
        masks_prob = torch.FloatTensor(N,H,W)

        for t in range(key_indices.shape[0]):
            # Soft labels of source nodes
            ctx_lbls = lbls[key_indices[t]].cuda()
            ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

            # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
            pred = (ctx_lbls[:, Is[t]] * Ws[t][None].cuda()).sum(1)
            pred = pred.view(-1, h, w)
            pred = pred.permute(1,2,0)

            if t > 0:
                lbls[t + n_context] = pred
            else:
                pred = lbls[0]
                lbls[t + n_context] = pred

            if self.norm_mask:
                pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                pred[:, :, :] /= pred.max(-1)[0][:, :, None]

            # Adding Predictions            
            pred_ = pred.permute([2,0,1])[None, ...]
            pred_up = F.interpolate(pred_, (H,W), mode="bilinear", align_corners=True)
            pred_up = pred_up[0].cpu()
            masks_idx[t] = pred_up.argmax(0)
            masks_prob[t] = pred_up[1]

        out = {}
        out["masks_pred_idx"] = masks_idx
        out["masks_pred_conf"] = masks_prob
        return out

def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context    
    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  torch.arange(N)[:, None])[:, :]]

    return ll + ss

def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''

    A = torch.einsum('ijklm,ijkn->iklmn', keys, query)

    # Mask
    A[0, :, len(long_mem):] += mask.to(device)

    _, N, T, h1w1, hw = A.shape
    A = A.view(N, T*h1w1, hw)
    A /= temperature

    weights, ids = torch.topk(A, topk, dim=-2)
    weights = F.softmax(weights, dim=-2)

    Ws = [w for w in weights]
    Is = [ii for ii in ids]

    return Ws, Is

def mem_efficient_batched_affinity(query, keys, mask, temperature, topk, long_mem):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    bsize, pbsize = 2, 100
    Ws, Is = [], []

    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].cuda(), query[:, :, b:b+bsize].cuda()
        w_s, i_s = [], []

        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            A[0, :, len(long_mem):] += mask[..., pb:pb+pbsize]

            _, N, T, h1w1, hw = A.shape
            A = A.view(N, T*h1w1, hw)
            A /= temperature

            weights, ids = torch.topk(A, topk, dim=-2)
            weights = F.softmax(weights, dim=-2)

            w_s.append(weights)
            i_s.append(ids)

        weights = torch.cat(w_s, dim=-1)
        ids = torch.cat(i_s, dim=-1)
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is


class MaskedAttention(nn.Module):
    '''
    A module that implements masked attention based on spatial locality 
    TODO implement in a more efficient way (torch sparse or correlation filter)
    '''
    def __init__(self, radius, flat=True):
        super(MaskedAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.index = {}

    def mask(self, H, W):
        if not ('%s-%s' %(H,W) in self.masks):
            self.make(H, W)
        return self.masks['%s-%s' %(H,W)]

    def index(self, H, W):
        if not ('%s-%s' %(H,W) in self.index):
            self.make_index(H, W)
        return self.index['%s-%s' %(H,W)]

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)
        
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = self.flatten(D)
        self.masks['%s-%s' %(H,W)] = D

        return D

    def flatten(self, D):
        return torch.flatten(torch.flatten(D, 1, 2), -2, -1)

    def make_index(self, H, W, pad=False):
        mask = self.mask(H, W).view(1, -1).byte()
        idx = torch.arange(0, mask.numel())[mask[0]][None]

        self.index['%s-%s' %(H,W)] = idx

        return idx
        
    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H,W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]

        return x * mask[0]
