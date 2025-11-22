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
        return self.dataset[idx, :2, :, :], self.dataset[idx, 2:, :, :]
    
def add_noise_to_data(data, snr_db):
    """
    向数据的第1个通道添加指定SNR的高斯白噪声
    
    Args:
        data: 形状为 [N, C, H, W] 的tensor 只对第0个通道加噪声
        snr_db: 信噪比(dB)
    
    Returns:
        加噪后的数据
    """
    if snr_db is None or snr_db == float('inf'):
        return data
    
    # 获取第1和第2个通道的数据
    signal_data_1 = data[:, 1, :, :].clone()  # [N, H, W]
    signal_data_2 = data[:, 2, :, :].clone()  # [N, H, W]
    
    # 将两个通道的TL转换回声压幅度
    pressure_signal_1 = torch.pow(10, -signal_data_1 / 20.0)
    pressure_signal_2 = torch.pow(10, -signal_data_2 / 20.0)
    
    # 计算两个通道的声压平方均值
    signal_power_1 = torch.mean(pressure_signal_1 ** 2)
    signal_power_2 = torch.mean(pressure_signal_2 ** 2)
    
    # 使用两个通道的平均功率作为总信号功率
    signal_power = (signal_power_1 + signal_power_2) / 2
    
    # 计算噪声方差
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    
    # 为两个通道分别生成高斯白噪声
    noise_1 = torch.randn_like(pressure_signal_1) * torch.sqrt(noise_variance)
    noise_2 = torch.randn_like(pressure_signal_2) * torch.sqrt(noise_variance)
    
    # 将噪声加到声压幅度上
    pressure_noisy_1 = pressure_signal_1 + noise_1
    pressure_noisy_2 = pressure_signal_2 + noise_2
    
    # 将加噪后的声压转换回TL域
    epsilon = 1e-37
    pressure_noisy_1 = torch.clamp(pressure_noisy_1, min=epsilon)
    pressure_noisy_2 = torch.clamp(pressure_noisy_2, min=epsilon)
    
    TL_noisy_1 = -20 * torch.log10(pressure_noisy_1)
    TL_noisy_2 = -20 * torch.log10(pressure_noisy_2)
    
    # 将加噪后的数据替换回原数据
    data_noisy = data.clone()
    data_noisy[:, 1, :, :] = TL_noisy_1
    data_noisy[:, 2, :, :] = TL_noisy_2

    return data_noisy, signal_power, noise_variance
    

def load_data(data_file, free_data_file, batch_size, train_val_num, test_num, num_workers, pin_memory, seed, snr_db, save_path):
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
    dataset = torch.cat((dataset, dataset[:, 2, :, :].unsqueeze(1)), dim=1)

    # 所有数据打乱 取train val test dataset
    np.random.seed(seed)
    per = np.random.permutation(dataset.shape[0])
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'permutation.npy'), per)
    dataset = dataset[per, :, :, :]
    train_dataset = dataset[: int(np.round(train_val_num*4/5)), :, :, :]
    val_dataset = dataset[int(np.round(train_val_num*4/5)): train_val_num, :, :, :]
    test_dataset = dataset[train_val_num: train_val_num + test_num, :, :, :]

    # 添加噪声到各个数据集
    if snr_db is not None:
        logger.info(f'Adding noise with SNR = {snr_db} dB to datasets', root=save_path)
        logger.info(f'TL data range: min={dataset[:, 0, :, :].min().item():.2f}, max={dataset[:, 0, :, :].max().item():.2f}', root=save_path)
        
        train_dataset, signal_power, noise_variance = add_noise_to_data(train_dataset, snr_db)
        logger.info(f'Training dataset: signal power: {signal_power.item():.6e}, Noise variance: {noise_variance.item():.6e}', root=save_path)
        val_dataset, signal_power, noise_variance = add_noise_to_data(val_dataset, snr_db)
        logger.info(f'Validation dataset: signal power: {signal_power.item():.6e}, Noise variance: {noise_variance.item():.6e}', root=save_path)
        test_dataset, signal_power, noise_variance = add_noise_to_data(test_dataset, snr_db)
        logger.info(f'Test dataset: signal power: {signal_power.item():.6e}, Noise variance: {noise_variance.item():.6e}', root=save_path)
        
        logger.info(f'After adding noise - Train TL range: min={train_dataset[:, 0, :, :].min().item():.2f}, max={train_dataset[:, 0, :, :].max().item():.2f}', root=save_path)
        logger.info('Noise added successfully to train, val, and test datasets', root=save_path)
    else:
        logger.info('No noise added (snr_db is None)', root=save_path)

    train_loader = DataLoader(SPPDataset(train_dataset, batch_size), batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Train loader has {len(train_loader.dataset)} samples, number of batches is {len(train_loader)}', root=save_path)
    val_loader = DataLoader(SPPDataset(val_dataset, batch_size), batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Val loader has {len(val_loader.dataset)} samples, number of batches is {len(val_loader)}', root=save_path)
    test_loader = DataLoader(SPPDataset(test_dataset, batch_size), batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f'Test loader has {len(test_loader.dataset)} samples, number of batches is {len(test_loader)}', root=save_path)

    return train_loader, val_loader, test_loader
