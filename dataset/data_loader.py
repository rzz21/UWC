import os
import numpy as np
import h5py
from utils import logger

import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['SPPDataset', 'load_data']

class SPPDataset(torch.utils.data.Dataset):
    r''' SPPDataset
    data小深度对应的TL label大深度对应的TL
    '''

    def __init__(self, dataset, batch_size):
        super(SPPDataset, self).__init__()

        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]
        
    def __getitem__(self, idx):
        return self.dataset[idx, :2, :, :], self.dataset[idx, 2, :, :]
    

def load_data(data_file, free_data_file, batch_size, train_val_num, test_num, num_workers, pin_memory, seed, save_path):
    assert os.path.isfile(data_file)

    # dataset munk ssp  dim1(0) data dim1(1) label
    dataset = h5py.File(data_file)['Coh_data_matrix_256512']
    dataset = np.array(dataset, dtype=np.float32).transpose(3, 2, 1, 0)

    # max_val = np.max(dataset[0, 0, :, :])
    # for i in range(dataset.shape[0]):
    #     max_val_0 = np.max(dataset[i, 0, :, :])
    #     max_val_1 = np.max(dataset[i, 1, :, :])
    #     assert max_val_0 == max_val, f"Max values do not match for index {i},0: {max_val_0} != {max_val}"
    #     assert max_val_1 == max_val, f"Max values do not match for index {i},1: {max_val_1} != {max_val}"
    # dataset = dataset / max_val # Normalize

    dataset = torch.from_numpy(dataset)

    # free_dataset free space 
    free_dataset = h5py.File(free_data_file)['Coh_data_matrix_free_256512']
    free_dataset = np.array(free_dataset, dtype=np.float32).transpose(3, 2, 1, 0)
    free_dataset = free_dataset[:, 1, :, :]
    free_dataset = np.expand_dims(free_dataset, axis=1)

    # max_val = np.max(free_dataset[0, 0, :, :])
    # for i in range(free_dataset.shape[0]):
    #     max_val_0 = np.max(free_dataset[i, 0, :, :])
    #     assert max_val_0 == max_val, f"Max values do not match for index {i},0: {max_val_0} != {max_val}"
    # free_dataset = free_dataset / max_val # Normalize

    free_dataset = torch.from_numpy(free_dataset)

    dataset = torch.concat((free_dataset, dataset), dim=1)

    # 所有数据打乱 取train val test dataset
    np.random.seed(seed)
    per = np.random.permutation(dataset.shape[0])
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'permutation.npy'), per)
    dataset = dataset[per, :, :, :]
    train_val_dataset = dataset[: train_val_num, :, :, :]
    test_dataset = dataset[train_val_num: train_val_num + test_num, :, :, :]

    train_loader = DataLoader(SPPDataset(train_val_dataset[:int(np.round(train_val_num*4/5)), :, :, :], batch_size), batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Train loader has {len(train_loader.dataset)} samples, number of batches is {len(train_loader)}', root=save_path)
    val_loader = DataLoader(SPPDataset(train_val_dataset[int(np.round(train_val_num*4/5)):, :, :, :], batch_size), batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Val loader has {len(val_loader.dataset)} samples, number of batches is {len(val_loader)}', root=save_path)
    test_loader = DataLoader(SPPDataset(test_dataset, batch_size), batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Test loader has {len(test_loader.dataset)} samples, number of batches is {len(test_loader)}', root=save_path)

    return train_loader, val_loader, test_loader
