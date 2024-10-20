import torch
import numpy as np
from transformer import Block


class SVTx(torch.nn.Module):
    def __init__(self, seqlen, in_dim, params):
        super().__init__()
        self.seqlen = seqlen
        self.in_dim = in_dim

        mdl_cfg = params['model']
        self.enc_emb_dim = mdl_cfg['enc_emb_dim']

        # Token embedding with LayerNorm and ReLU activation
        self.token_emb = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),  # Layer normalization
            torch.nn.ReLU()  # ReLU activation
        )

        # Learnable position embedding
        self.enc_pos_emb = torch.nn.Parameter(torch.zeros(1, self.seqlen, self.enc_emb_dim), requires_grad=True)

        # Encoder blocks
        self.enc_blocks = torch.nn.ModuleList([
            Block(self.enc_emb_dim, mdl_cfg['enc_nhead'], mlp_ratio=1, qkv_bias=True,
                  norm_layer=torch.nn.LayerNorm) for _ in range(mdl_cfg['enc_nlayer'])
        ])
        self.enc_norm = torch.nn.LayerNorm(self.enc_emb_dim)

        self.initialize_weights()

    def pos_embd_gen(self, seqlen, emb_dim, cls_token):
        pos_emb = np.zeros((seqlen, emb_dim), dtype=np.float32)
        for pos in range(seqlen):
            for c in range(emb_dim):
                pos_emb[pos, c] = pos / np.power(10000, 2 * (c // 2) / emb_dim)

        pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])  # dim 2i
        pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])  # dim 2i+1

        if cls_token:
            pos_emb = np.concatenate([np.zeros([1, emb_dim]), pos_emb], axis=0)
        return pos_emb

    def initialize_weights(self):
        # Initialize positional embedding using sin-cos embedding
        enc_pos_emb = self.pos_embd_gen(self.seqlen, self.enc_emb_dim, cls_token=False)
        self.enc_pos_emb.data.copy_(torch.from_numpy(enc_pos_emb).float().unsqueeze(0))

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def get_mask_index(self, x):
        N, L, D = x.shape  # batch, length, dim
        ids_keep = []
        for i in range(x.shape[0]):
            ids = torch.nonzero(x[i, :, x.shape[2] // 2] > torch.min(x))
            ids_keep.append(ids)
        min_length = min([len(t) for t in ids_keep])
        for i, tensor in enumerate(ids_keep):
            ids_keep[i] = torch.nn.functional.pad(tensor, (0, 0, 0, min_length - len(tensor)))
        ids_keep = torch.stack(ids_keep)
        mask = torch.ones((N, L), device=x.device)
        mask = mask.scatter_(1, ids_keep.squeeze(dim=2), 0)

        masked_ids = torch.nonzero(mask).flatten()[1::2].resize(N, L - min_length)
        ids_restore = torch.argsort(torch.cat([ids_keep.squeeze(dim=2), masked_ids], dim=1), dim=1)

        return mask, ids_restore, ids_keep

    def get_image_mask(self, x, ids_keep):
        N, L, D = x.shape  # batch, length, dim
        x_masked = torch.gather(x, dim=1, index=ids_keep.repeat(1, 1, D))

        return x_masked

    def forward_enc(self, x):
        x_temp = x.squeeze(dim=1)  # Ensure input is correctly sized

        mask, ids_restore, ids_keep = self.get_mask_index(x_temp)

        # Apply token embedding (LayerNorm + ReLU) to the input
        x_temp = self.token_emb(x_temp)

        # Add positional embedding
        _emb = x_temp + self.enc_pos_emb

        for blk in self.enc_blocks:
            _emb = blk(_emb)
        _emb = self.enc_norm(_emb)

        return _emb, mask, ids_restore

    def forward_loss(self, target, pred):
        loss = torch.nn.functional.l1_loss(pred, target)
        return loss

    def forward(self, imgs, target):
        pred, mask, ids_restore = self.forward_enc(imgs)

        pred_mask = pred.unsqueeze(1)
        mask_expand = mask.unsqueeze(2).expand(pred.size()).unsqueeze(1)
        pred_mask[mask_expand != 1] = target[mask_expand != 1]

        loss = self.forward_loss(target, pred_mask)
        return loss, pred, mask
