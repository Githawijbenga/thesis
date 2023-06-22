import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device
import pdb

data_dict = torch.load('data/dict_total.pt')

class RumourTwitter(object):
    def __init__(self, data_dict):
        self.data = data_dict

    def __getitem__(self, index):
        sample = self.data[index]
        tweet_id = sample['tweetid']
        embedding = torch.Tensor(sample['embedding'])
        timestamps = sample['timestamp']
        timestamp = torch.Tensor([i.timestamp() for i in timestamps])
        label = torch.Tensor(sample['label']).to(dtype = torch.int64)
        label = torch.nn.functional.one_hot(label, num_classes = 4)
        mask = torch.ones_like(embedding)

        return tweet_id, timestamp, embedding, mask, label
    
    def __len__(self):
	    return len(self.data)

def get_data_min_max_tweet(records):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (tweet_id, tt, vals, mask, labels) in enumerate(records):        
		n_features = vals.shape[1]
		
		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max

def variable_time_collate_tweet(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	N = batch[0][-1].shape[1] # number of labels
	
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)
	
	for b, (tweet_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask
		combined_labels[b, indices] = labels.to(torch.float32)

	combined_tt = combined_tt.float()

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)

	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict
    
if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = RumourTwitter(data_dict)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_tweet)
	print(dataloader.__iter__().next())
