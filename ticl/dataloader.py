import numpy as np
import torch
from torch.utils.data import DataLoader

import ticl.priors as priors
from ticl.priors import ClassificationAdapterPrior, BagPrior, GPPrior
from ticl.distributions import parse_distribution

# import wandb
from tqdm import tqdm

class PriorDataLoader(DataLoader):
    def __init__(
        self, 
        prior, 
        num_steps, 
        batch_size, 
        min_eval_pos, 
        n_samples, 
        device, 
        num_features,
        treatment_part,
        model = None,
        random_n_samples = False,
        n_test_samples = False,
    ):

        if (random_n_samples and not n_test_samples) or (not random_n_samples and n_test_samples):
            raise ValueError("random_n_samples and test_samples must be set together.")
        
        self.prior = prior
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos

        self.random_n_samples = random_n_samples
        if random_n_samples:
            self.n_samples = np.random.randint(min_eval_pos, random_n_samples)
            self.n_test_samples = n_test_samples
        else:
            self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.epoch_count = 0
        self.model = model
        self.treatment_part = parse_distribution('treatment_part', **treatment_part)

    def gbm(self, epoch=None, single_eval_pos = None):
        # Actually can only sample up to n_samples-1
        if self.random_n_samples:
            single_eval_pos = self.n_samples - self.n_test_samples
            
        # comment this for debug
        if single_eval_pos is None:
            single_eval_pos = np.random.randint(self.min_eval_pos, self.n_samples)
        # single_eval_pos = 31496
        
        batch = self.prior.get_batch(
            batch_size=self.batch_size, 
            n_samples=self.n_samples, 
            num_features=self.num_features, 
            device=self.device,
            epoch=epoch,
            single_eval_pos=single_eval_pos,
        )
        # we return sampled hyperparameters from get_batch for testing but we don't want to use them as style.
        x, y_0, y_1, info = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
        
        c = np.zeros((x.shape[0], x.shape[1]), dtype=np.int64)
        p = self.treatment_part()
        c[:single_eval_pos, :] = np.random.binomial(1, p, (single_eval_pos, x.shape[1]))
        c = torch.from_numpy(c).to(self.device)
                
        diff = y_1 - y_0
        y = torch.empty_like(y_0)
        mask_0 = c == 0
        mask_1 = c == 1 
        y[mask_0] = y_0[mask_0]
        y[mask_1] = y_1[mask_1]
        # y_mean_0 = torch.mean(y_0, dim=0, keepdim=True)
        # y_mean_1 = torch.mean(y_1, dim=0, keepdim=True)
        # y_std = torch.std(torch.cat((y_0, y_1), dim=0), dim=0, keepdim=True)
        # y_std[y_std < 1e-6] = 1
        # y[mask_0] = (y[mask_0] - y_mean_0.expand_as(mask_0)[mask_0]) / y_std.expand_as(mask_0)[mask_0]
        # y[mask_1] = (y[mask_1] - y_mean_1.expand_as(mask_1)[mask_1]) / y_std.expand_as(mask_1)[mask_1]
        # diff = (y_1 - y_mean_1 - (y_0 - y_mean_0)) / y_std
        
        return (x, y, c), diff, single_eval_pos

    def __len__(self):
        return self.num_steps

    def get_test_batch(self):  # does not increase epoch_count
        return self.gbm(epoch=self.epoch_count)
    
    def iter_safe_gbm(self):
        for _ in range(self.num_steps):
            yield self.gbm(epoch=self.epoch_count - 1)

    def __iter__(self):
        self.epoch_count += 1
        return iter(self.iter_safe_gbm())


def get_dataloader(prior_config, dataloader_config, device, model = None):

    prior_type = prior_config['prior_type']
    gp_flexible = ClassificationAdapterPrior(priors.GPPrior(prior_config['gp']), num_features=prior_config['num_features'], **prior_config['classification'])
    mlp_flexible = ClassificationAdapterPrior(priors.MLPPrior(prior_config['mlp']), num_features=prior_config['num_features'], **prior_config['classification'])

    if prior_type == 'prior_bag':
        # Prior bag combines priors
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                         prior_weights={'mlp': 0.961, 'gp': 0.039})
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    return PriorDataLoader(
        prior=prior, 
        n_samples=prior_config['n_samples'],
        device=device, 
        num_features=prior_config['num_features'], 
        model = model,
        **dataloader_config
    )
