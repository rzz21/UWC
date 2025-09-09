import torch
import numpy
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
from utils import logger

__all__ = ['AverageMeter', 'evaluator', 'masked_MSE', 'masked_image_gradient_loss']

class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"
    

def evaluator(output, label):
    SSIM_list = []
    NMSE_list = []
    for idx in range(output.shape[0]):
        output_np = output[idx, ...].squeeze().cpu().numpy()
        label_np = label[idx, ...].squeeze().cpu().numpy()
        SSIM_list.append(ssim(output_np, label_np, data_range=100))
        # output_mean = uniform_filter(output_np, size=7).mean()
        # label_mean = uniform_filter(label_np, size=7).mean()
        # output_var = uniform_filter(output_np ** 2, size=7).mean() - output_mean ** 2
        # label_var = uniform_filter(label_np ** 2, size=7).mean() - label_mean ** 2
        # covariance = uniform_filter(output_np * label_np, size=7).mean() - output_mean * label_mean

        # logger.info(f'Uniform filter - output mean: {output_mean}, label mean: {label_mean}, output variance: {output_var}, label variance: {label_var}, covariance: {covariance}')

        # output_mean_np = numpy.mean(output_np)
        # label_mean_np = numpy.mean(label_np)
        # output_var_np = numpy.var(output_np)
        # label_var_np = numpy.var(label_np)
        # covariance_np = numpy.mean((output_np - output_mean_np) * (label_np - label_mean_np))

        # logger.info(f'Numpy - output mean: {output_mean_np}, label mean: {label_mean_np}, output variance: {output_var_np}, label variance: {label_var_np}, covariance: {covariance_np}')
        nmse = numpy.mean((output_np - label_np) ** 2) / numpy.mean(label_np ** 2)
        nmse_db = 10 * numpy.log10(nmse)
        NMSE_list.append(nmse_db)
    return numpy.mean(SSIM_list), numpy.mean(NMSE_list)


def masked_MSE(output, label, enable_mask=False, mask_data_range=None, mask_weight=None):
    if enable_mask:
        mask = torch.ones_like(label)
        mask[(label >= mask_data_range[0]) & (label <= mask_data_range[1])] = mask_weight
        mse = torch.mean(mask * (output - label) ** 2)
    else:
        mse = torch.nn.functional.mse_loss(output, label)
    return mse

def masked_image_gradient_loss(output, label, enable_mask=False, mask_data_range=None, mask_weight=None):
    
    # 计算图像的梯度
    def compute_gradient(img):
        grad_x = img[:, :, :-1] - img[:, :, 1:]
        grad_y = img[:, :-1, :] - img[:, 1:, :]
        return grad_x, grad_y

    # 计算输出和标签的梯度
    grad_output_x, grad_output_y = compute_gradient(output)
    grad_label_x, grad_label_y = compute_gradient(label)

    if enable_mask:
        mask_x = torch.ones_like(grad_label_x)  
        mask_y = torch.ones_like(grad_label_y)
        mask_x[(torch.abs(grad_label_x) >= mask_data_range[0]) & (torch.abs(grad_label_x) <= mask_data_range[1])] = mask_weight
        mask_y[(torch.abs(grad_label_y) >= mask_data_range[0]) & (torch.abs(grad_label_y) <= mask_data_range[1])] = mask_weight

        loss_x = torch.mean(mask_x * (grad_output_x - grad_label_x) ** 2)
        loss_y = torch.mean(mask_y * (grad_output_y - grad_label_y) ** 2)
    else:
        loss_x = torch.nn.functional.mse_loss(grad_output_x, grad_label_x)
        loss_y = torch.nn.functional.mse_loss(grad_output_y, grad_label_y)

    loss = loss_x + loss_y
    return loss