import numpy as np 
import h5py, logging
from torch.utils.data import Dataset

class SinogramDataset(Dataset):
    def __init__(self, ifn, params, world_size=None, rank=None):
        ds_cfg = params['dataset']

        h5fd = h5py.File(ifn, 'r')
        self.sample = h5fd[ds_cfg['input']][:]
        logging.info(f"loading sino_missing_wg data from index 1 to {self.sample.shape[0]}")
        h5fd.close()

        h5fd = h5py.File(ifn, 'r')
        self.target = h5fd[ds_cfg['target']][:]
        logging.info(f"loading sino_full data from index 1 to {self.target.shape[0]}")
        h5fd.close()

        # norm or not will influence pos-encodding
        if ds_cfg['norm']:
            if ds_cfg.get('mean4norm') is None:
                _avg = self.target.mean()
                _std = self.target.std()
                self.target = ((self.target - _avg) / _std).astype(np.float32)
                self.sample = ((self.sample - _avg) / _std).astype(np.float32)
                logging.info(f'features are normalized with computed mean: {_avg}, std: {_std}')
            else:
                self.target = ((self.target - ds_cfg['mean4norm']) / ds_cfg['std4norm']).astype(np.float32)
                self.sample = ((self.sample - ds_cfg['mean4norm']) / ds_cfg['std4norm']).astype(np.float32)
                logging.info(f"features are normalized with provided mean: {ds_cfg['mean4norm']}, std: {ds_cfg['std4norm']}")

        self.len    = self.target.shape[0]
        self.seqlen = self.target.shape[-2] # angle
        self.cdim   = self.target.shape[-1] # resolution
        self.shape  = self.target.shape

    def __getitem__(self, idx):
            return self.sample[idx][None].astype(np.float32), self.target[idx][None].astype(np.float32)

    def __len__(self):
        return self.len