import time, multiprocessing
import os
import torch
from collections import namedtuple
import matplotlib.pylab as plt
from matplotlib import font_manager

from utils import logger
from utils.statics import AverageMeter, evaluator, masked_MSE, masked_image_gradient_loss

__all__ = ['Trainer', 'Tester']

field = ('ssim', 'nmse', 'epoch') # To be determined
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r'''
    train
    loss用MSE
    To be finished 训练初步定用MSE做loss 测试用什么 暂定ssim？
    '''

    def __init__(self, model, device, optimizer, criterion, scheduler, alpha, 
                 mask_MSE=False, mask_IG=False, MSE_mask_range=None, MSE_mask_weight=None, IG_mask_range=None, IG_mask_weight=None,
                 resume=None, save_path=None, print_freq=20, val_freq=10, test_freq=10):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.alpha = alpha

        self.mask_MSE = mask_MSE
        self.mask_IG = mask_IG
        self.MSE_mask_range = MSE_mask_range
        self.MSE_mask_weight = MSE_mask_weight
        self.IG_mask_range = IG_mask_range
        self.IG_mask_weight = IG_mask_weight

        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.ssims = [] # To be determined
        self.nmses = []
        self.best_ssim = Result()
        self.best_nmse = Result()  

        self.tester = Tester(model, device, criterion, self.alpha, 
                             mask_MSE, mask_IG, MSE_mask_range, MSE_mask_weight, IG_mask_range, IG_mask_weight,
                             print_freq=print_freq, save_path=save_path)
        self.test_loader = None

    def loop(self, epochs, train_loader, valid_loader, test_loader):
        r'''
        each epoch
        '''

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # train
            self.train_loss = self.train(train_loader)
            self.train_losses.append((ep, self.train_loss.item()))

            # val
            if ep % self.val_freq == 0:
                self.val_loss = self.val(valid_loader)
                self.val_losses.append((ep, self.val_loss.item()))

            # test
            if ep % self.test_freq == 0:
                self.test_loss, ssim, nmse = self.test(test_loader) # To be determined
                self.ssims.append((ep, ssim)) # To be determined
                self.nmses.append((ep, nmse))
                self.test_losses.append((ep, self.test_loss.item()))
            else:
                ssim=None
                nmse=None

            self._loop_postprocessing(ssim, nmse)

        self.plot_losses([[epoch for (epoch, loss) in self.train_losses], [epoch for (epoch, loss) in self.val_losses], [epoch for (epoch, loss) in self.test_losses]], # To be determined, 
                         [[loss for (epoch, loss) in self.train_losses], [loss for (epoch, loss) in self.val_losses], [loss for (epoch, loss) in self.test_losses]], # To be determined, 
                         'loss', 'epoch', 'loss', 'figs_loss.png', ['train_loss', 'val_loss','test_loss'])
        self.plot_losses([[epoch for (epoch, ssim) in self.ssims]],
                         [[ssim for (epoch, ssim) in self.ssims]],
                         'ssim', 'epoch', 'ssim', 'figs_ssim.png', ['ssim'])
        self.plot_losses([[epoch for (epoch, nmse) in self.nmses]],
                         [[nmse for (epoch, nmse) in self.nmses]],
                         'nmse', 'epoch', 'nmse', 'figs_nmse.png', ['nmse'])
        
    def _resume(self):
        if self.resume_file is None:
            return None
        
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}', root=self.save_path)
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_ssim = checkpoint['best_ssim'] # To be determined
        self.best_nmse = checkpoint['best_nmse']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.ssims = checkpoint['ssims'] # To be determined
        self.nmses = checkpoint['nmses']
        self.cur_epoch += 1

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} from epoch {self.cur_epoch}.\n', root=self.save_path)

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No save path specified, checkpoint not saved.')
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)
        
    def val(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(valid_loader)
        
    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=True)
        
    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_MSE_loss = AverageMeter('Iter MSE loss')
        iter_IG_loss = AverageMeter('Iter IG loss')

        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(self.device), label.to(self.device)
            data = torch.clamp(data, min=40, max=140)
            label = torch.clamp(label, min=40, max=140)
            data = (data - 40) / 100
            label = (label - 40) / 100
            output = self.model(data)
            MSE_loss = masked_MSE(output.squeeze(1), label, self.mask_MSE, self.MSE_mask_range, self.MSE_mask_weight)
            IG_loss = masked_image_gradient_loss(output.squeeze(1), label, self.mask_IG, self.IG_mask_range, self.IG_mask_weight)
            loss = MSE_loss + self.alpha*IG_loss

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            iter_loss.update(loss)
            iter_MSE_loss.update(MSE_loss)
            iter_IG_loss.update(IG_loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MSE loss: {iter_MSE_loss.avg:.3e} | '
                            f'IG loss: {iter_IG_loss.avg:.3e} | '
                            f'Total loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f} |'
                            f'num workkers: {multiprocessing.cpu_count()} \n',
                            root=self.save_path)
                
        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                    f' {mode} Loss: {iter_loss.avg:.3e} \n',
                    root=self.save_path)
        
        return iter_loss.avg
    
    def _loop_postprocessing(self, ssim, nmse):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_ssim': self.best_ssim,
            'best_nmse': self.best_nmse,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ssims': self.ssims,
            'nmses': self.nmses
        }

        if ssim is not None:
            if self.best_ssim.ssim is None or ssim > self.best_ssim.ssim:
                self.best_ssim = Result(ssim, nmse, self.cur_epoch)
                state['best_ssim'] = self.best_ssim
                self._save(state, name='best_ssim.pth')
        
        if nmse is not None:
            if self.best_nmse.nmse is None or nmse < self.best_nmse.nmse:
                self.best_nmse = Result(ssim, nmse, self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name='best_nmse.pth')

        self._save(state, name='last.pth')

        if self.best_ssim.ssim is not None:
            logger.info(f'=> Best ssim: {self.best_ssim.ssim:.3f} at epoch {self.best_ssim.epoch}\n', root=self.save_path)
        if self.best_nmse.nmse is not None:
            logger.info(f'=> Best nmse: {self.best_nmse.nmse:.3f} at epoch {self.best_nmse.epoch}\n', root=self.save_path)

    def plot_losses(self, x_values, y_values, title, xlabel, ylabel, filename, labels):
        fig, ax = plt.subplots(figsize=(20, 10))
        for x, y, label in zip(x_values, y_values, labels):
            ax.semilogy(x, y, linewidth=3, marker='o', markersize=10, label=label)
        ax.set_title(title, fontsize=40)
        ax.set_xlabel(xlabel, fontsize=30)
        ax.set_ylabel(ylabel, fontsize=30)
        ax.legend(fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

        if self.save_path is None:
            logger.warning('No save path specified, plot not saved.')
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, filename))


class Tester:
    r'''
    test
    ssim  to be determined
    '''

    def __init__(self, model, device, criterion, alpha, 
                 mask_MSE=False, mask_IG=False, MSE_mask_range=None, MSE_mask_weight=None, IG_mask_range=None, IG_mask_weight=None,
                 print_freq=20, save_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.alpha = alpha

        self.mask_MSE = mask_MSE
        self.mask_IG = mask_IG
        self.MSE_mask_range = MSE_mask_range
        self.MSE_mask_weight = MSE_mask_weight
        self.IG_mask_range = IG_mask_range
        self.IG_mask_weight = IG_mask_weight

        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, ssim, nmse = self._iteration(test_loader)
        if verbose:
            logger.info(f'=> Test ssim: {ssim:.3f}, nmse: {nmse:.3f}, loss: {loss:.3f}\n', root=self.save_path)
        return loss, ssim, nmse
    
    def _iteration(self, data_loader):
        iter_ssim = AverageMeter('Iter ssim') #`ssim` to be determined
        iter_nmse = AverageMeter('Iter nmse')
        iter_MSE_loss = AverageMeter('Iter MSE loss')
        iter_IG_loss = AverageMeter('Iter IG loss')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(self.device), label.to(self.device)
            data = torch.clamp(data, min=40, max=140)
            label = torch.clamp(label, min=40, max=140)
            data = (data - 40) / 100
            label = (label - 40) / 100
            output = self.model(data)
            MSE_loss = masked_MSE(output.squeeze(1), label, self.mask_MSE, self.MSE_mask_range, self.MSE_mask_weight)
            IG_loss = masked_image_gradient_loss(output.squeeze(1), label, self.mask_IG, self.IG_mask_range, self.IG_mask_weight)
            loss = MSE_loss + self.alpha*IG_loss

            # output = output * 100 + 40
            # label = label * 100 + 40
            # logger.info(f'shape of output:{output.shape}, shape of label:{label.shape}\n', root=self.save_path)
            ssim, nmse = evaluator(output, label) #`ssim` to be determined

            iter_ssim.update(ssim)
            iter_nmse.update(nmse)
            iter_MSE_loss.update(MSE_loss)
            iter_IG_loss.update(IG_loss)
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # if (batch_idx + 1) % self.print_freq == 0:
            if (batch_idx + 1) % 1 == 0:
                logger.info(f'Test: [{batch_idx + 1}/{len(data_loader)}] '
                            f'ssim: {iter_ssim.avg:.3f} | '
                            f'nmse: {iter_nmse.avg:.3f} | '
                            f'MSE loss: {iter_MSE_loss.avg:.4f} |'
                            f'IG loss: {iter_IG_loss.avg:.4f} |'
                            f'Total loss: {iter_loss.avg:.4f} | '
                            f'time: {iter_time.avg:.3f}\n',
                            root=self.save_path)
                
        logger.info(f'=> Test ssim: {iter_ssim.avg:.3f}, test nmse: {iter_nmse.avg:.3f}\n', root=self.save_path)

        return iter_loss.avg, iter_ssim.avg, iter_nmse.avg
    
    def plot_TL(self, num_pair, data_loader):
        for i in range(num_pair):
            data, label = next(iter(data_loader))
            logger.info(f'shape of data: {data.shape}, shape of label: {label.shape}\n', root=self.save_path)
            idx = torch.randint(0, data.size(0), (1,)).item()
            logger.info(f'num_pair:{i}, idx: {idx}\n', root=self.save_path)
            data, label = data[idx].to(self.device), label[idx].to(self.device)
            
            data = torch.clamp(data, min=40, max=140)
            label = torch.clamp(label, min=40, max=140)
            data = (data - 40) / 100
            label = (label - 40) / 100
            
            output = self.model(data.unsqueeze(0)).squeeze(0).squeeze(0)

            font_path = '/home/zhizhen/.fonts/times.ttf'
            font_bold_path = '/home/zhizhen/.fonts/timesbd.ttf'
            font_prop = font_manager.FontProperties(fname=font_path)
            font_bold_prop = font_manager.FontProperties(fname=font_bold_path)

            extent = [0, 100, 5000, 0]  # [xmin, xmax, ymax, ymin]

            save_dir = os.path.join(self.save_path, f'TL_figs/pair_{i+1}')
            os.makedirs(save_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(data[1, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('Range (km)', fontsize=35, fontproperties=font_bold_prop)
            ax.set_ylabel('Depth (m)', fontsize=35, fontproperties=font_bold_prop)
            ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
            for label_text in ax.get_xticklabels() + ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30, direction='in')
            for label_text in cbar.ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Data.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(data[1, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Data_pure.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(output.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('Range (km)', fontsize=35, fontproperties=font_bold_prop)
            ax.set_ylabel('Depth (m)', fontsize=35, fontproperties=font_bold_prop)
            ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
            for label_text in ax.get_xticklabels() + ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30, direction='in')
            for label_text in cbar.ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Output.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(output.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Output_pure.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(label.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('Range (km)', fontsize=35, fontproperties=font_bold_prop)
            ax.set_ylabel('Depth (m)', fontsize=35, fontproperties=font_bold_prop)
            ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
            for label_text in ax.get_xticklabels() + ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30, direction='in')
            for label_text in cbar.ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Label.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(data[0, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('Range (km)', fontsize=35, fontproperties=font_bold_prop)
            ax.set_ylabel('Depth (m)', fontsize=35, fontproperties=font_bold_prop)
            ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
            for label_text in ax.get_xticklabels() + ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30, direction='in')
            for label_text in cbar.ax.get_yticklabels():
                label_text.set_fontproperties(font_prop)
                label_text.set_fontsize(30)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Free_Space.svg'), format='svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20, 10))
            im = ax.imshow(data[0, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            ax.set_aspect(0.01)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'Free_Space_pure.svg'), format='svg')
            plt.close(fig)
            # fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # # 设置坐标范围
            # extent = [0, 200, 5000, 0]  # [xmin, xmax, ymax, ymin]

            # # 绘制 data[1, :, :]
            # im1 = axes[0, 0].imshow(data[1, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            # axes[0, 0].set_aspect(0.02)
            # axes[0, 0].set_title('Data')
            # axes[0, 0].set_xlabel('Range (km)')
            # axes[0, 0].set_ylabel('Depth (m)')
            # fig.colorbar(im1, ax=axes[0, 0])  # 添加colorbar

            # # 绘制 output
            # im2 = axes[0, 1].imshow(output.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            # axes[0, 1].set_aspect(0.02)
            # axes[0, 1].set_title('Output')
            # axes[0, 1].set_xlabel('Range (km)')
            # axes[0, 1].set_ylabel('Depth (m)')
            # fig.colorbar(im2, ax=axes[0, 1])  # 添加colorbar

            # # 绘制 label
            # im3 = axes[1, 0].imshow(label.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            # axes[1, 0].set_aspect(0.02)
            # axes[1, 0].set_title('Label')
            # axes[1, 0].set_xlabel('Range (km)')
            # axes[1, 0].set_ylabel('Depth (m)')
            # fig.colorbar(im3, ax=axes[1, 0])  # 添加colorbar

            # # 绘制 data[0, :, :]
            # im4 = axes[1, 1].imshow(data[0, :, :].detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=0, vmax=1)
            # axes[1, 1].set_aspect(0.02)
            # axes[1, 1].set_title('Free space')
            # axes[1, 1].set_xlabel('Range (km)')
            # axes[1, 1].set_ylabel('Depth (m)')
            # fig.colorbar(im4, ax=axes[1, 1])  # 添加colorbar

            # plt.tight_layout()
            # plt.show()
            # fig.suptitle(f'Pair {i + 1}')
            # fig.tight_layout()
            # os.makedirs(os.path.join(self.save_path, 'TL_figs'), exist_ok=True)
            # fig.savefig(os.path.join(self.save_path, 'TL_figs', f'pair_{i + 1}.png'))
            # plt.close(fig)
        logger.info('TL figs saved.\n', root=self.save_path)
        pass

    
    def plot_IG(self, num_pair, data_loader):
        for i in range(num_pair):
            data, label = next(iter(data_loader))
            idx = torch.randint(0, data.size(0), (1,)).item()
            data, label = data[idx].to(self.device), label[idx].to(self.device)
            output = self.model(data.unsqueeze(0)).squeeze(0).squeeze(0)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 设置坐标范围
            extent = [0, 200, 5000, 0]  # [xmin, xmax, ymax, ymin]

            # 绘制 label x IG
            label_x_IG = label[:, :-1] - label[:, 1:]
            im1 = axes[0, 0].imshow(label_x_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[0, 0].set_aspect(0.02)
            axes[0, 0].set_title('Label Image Gradient -- x')
            axes[0, 0].set_xlabel('Range (km)')
            axes[0, 0].set_ylabel('Depth (m)')
            fig.colorbar(im1, ax=axes[0, 0])  # 添加colorbar

            # 绘制 label y IG
            label_y_IG = label[:-1, :] - label[1:, :]
            im2 = axes[0, 1].imshow(label_y_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[0, 1].set_aspect(0.02)
            axes[0, 1].set_title('Label Image Gradient -- y')
            axes[0, 1].set_xlabel('Range (km)')
            axes[0, 1].set_ylabel('Depth (m)')
            fig.colorbar(im2, ax=axes[0, 1])  # 添加colorbar

            # 绘制 output x IG
            output_x_IG = output[:, :-1] - output[:, 1:]
            im3 = axes[1, 0].imshow(output_x_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[1, 0].set_aspect(0.02)
            axes[1, 0].set_title('Output Image Gradient -- x')
            axes[1, 0].set_xlabel('Range (km)')
            axes[1, 0].set_ylabel('Depth (m)')
            fig.colorbar(im3, ax=axes[1, 0])  # 添加colorbar

            # 绘制 output y IG
            output_y_IG = output[:-1, :] - output[1:, :]
            im4 = axes[1, 1].imshow(output_y_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[1, 1].set_aspect(0.02)
            axes[1, 1].set_title('Output Image Gradient -- y')
            axes[1, 1].set_xlabel('Range (km)')
            axes[1, 1].set_ylabel('Depth (m)')
            fig.colorbar(im4, ax=axes[1, 1])  # 添加colorbar

            plt.tight_layout()
            plt.show()

            fig.suptitle(f'Pair {i + 1}')
            fig.tight_layout()
            os.makedirs(os.path.join(self.save_path, 'IG_figs'), exist_ok=True)
            fig.savefig(os.path.join(self.save_path, 'IG_figs', f'pair_{i + 1}.png'))
            plt.close(fig)
        logger.info('IG figs saved.\n', root=self.save_path)


    def plot_losses(self, x_values, y_values, xlabel, ylabel, filename, labels):
        font_path = '/home/zhizhen/.fonts/times.ttf'
        font_blod_path = '/home/zhizhen/.fonts/timesbd.ttf'
        font_prop = font_manager.FontProperties(fname=font_path)
        font_bold_prop = font_manager.FontProperties(fname=font_blod_path)
        fig, ax = plt.subplots(figsize=(10, 7))
        for x, y, label in zip(x_values, y_values, labels):
            ax.semilogy(x, y, linewidth=2, markersize=10, label=label)
        
        ax.set_xlabel(xlabel, fontsize=15, fontweight='bold', fontproperties=font_bold_prop)
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold', fontproperties=font_bold_prop)
        plt.rcParams.update({'font.size': 40}) 
        # ax.legend(loc='upper right', borderpad=2.5, labelspacing=2.5, prop=font_prop, handlelength=4, handleheight=2, markerscale=2)
        ax.legend(loc='upper right', prop=font_prop, fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=12, direction='in', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=12, direction='in', width=1.5, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(12)
            
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e-1)
        ax.set_yticks([1e-3, 1e-2, 1e-1])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        ax.grid(True, which='major', linestyle='--', linewidth=1.5, color='lightgray')
        ax.grid(True, which='minor', linestyle='--', linewidth=1.5, color='lightgray')
        
        if self.save_path is None:
            logger.warning('No save path specified, plot not saved.')
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, filename), format='pdf')    