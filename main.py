import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import load_data
import numpy as np
import scipy.io, re

def main():
    if args.finetune or (args.pretrained and 'finetuneTrue' in args.pretrained):
        match = re.search(r'seed_(\d+)_', args.pretrained)
        if match:
            pretrained_seed = match.group(1)
        save_path = args.root_dir + f'large/checkpoints_{args.scenario}_range{args.range}km/pretrained_seed{pretrained_seed}_seed_{args.seed}_trainvalnum_{args.train_val_num}_testnum_{args.test_num}_epoch_{args.epochs}_finetune{args.finetune}_finetunescenario{args.finetune_scenario}_alpha_{args.alpha}_scheduler{args.scheduler}/'
    else:
        save_path = args.root_dir + f'large/checkpoints_{args.scenario}_range{args.range}km/seed_{args.seed}_trainvalnum_{args.train_val_num}_testnum_{args.test_num}_epoch_{args.epochs}_finetune{args.finetune}_alpha_{args.alpha}_scheduler{args.scheduler}/'
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}', root=save_path)
    if args.evaluate:
        save_path = args.pretrained[:-8]

    if args.bty != 'flat':
        save_path = args.root_dir + f'large/checkpoints_{args.scenario}_{args.bty}_range{args.range}km/seed_{args.seed}_trainvalnum_{args.train_val_num}_testnum_{args.test_num}_epoch_{args.epochs}_finetune{args.finetune}_alpha_{args.alpha}_scheduler{args.scheduler}/'

    logger.info('=> PyTorch Version: {}'.format(torch.__version__), root=save_path)

    # Environment initialization
    device, pin_memory, num_workers = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity, save_path)
    
    # Create the data loader
    if args.bty == 'flat':
        if args.finetune or (args.pretrained and 'finetuneTrue' in args.pretrained):
            data_dir = args.root_dir + f'Coh_data_matrix_256512_{args.finetune_scenario}_range{args.finetune_range}.mat'
            free_data_dir = args.root_dir + f'Coh_data_matrix_free_256512_{args.finetune_scenario}_range{args.finetune_range}.mat'
            logger.info(f'Fine tune model in {args.finetune_scenario}\n', root=save_path)
        else:
            # data_dir = args.root_dir + 'Coh_data_matrix_256512.mat'
            # free_data_dir = args.root_dir + 'Coh_data_matrix_free_256512.mat'
            data_dir = args.root_dir + f'Coh_data_matrix_256512_{args.scenario}_range{args.range}.mat'
            free_data_dir = args.root_dir + f'Coh_data_matrix_free_256512_{args.scenario}_range{args.range}.mat'
    else:
        data_dir = args.root_dir + f'Coh_data_matrix_256512_{args.scenario}_{args.bty}_range{args.range}.mat'
        free_data_dir = args.root_dir + f'Coh_data_matrix_free_256512_{args.scenario}_range{args.range}.mat'

    train_loader, val_loader, test_loader = load_data(data_dir, free_data_dir, args.batch_size, args.train_val_num, args.test_num, args.workers, pin_memory, args.seed_data, save_path)

    # Define model
    model = init_model(args.pretrained, w=512, h=256, save_path=save_path)
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        tester = Tester(model, device, criterion, args.alpha, 
                        args.mask_MSE, args.mask_IG, args.MSE_mask_range, args.MSE_mask_weight, args.IG_mask_range, args.IG_mask_weight,
                        save_path=save_path)
        loss, ssim, nmse = tester(test_loader)

        logger.info(f"\n=! Evaluation test loss: {loss:.3e}"
                    f"\n              test ssim: {ssim:.3e}\n"
                    f"\n              test nmse: {nmse:.3e}\n", root=save_path)
        logger.info(f'Test loader has {len(test_loader.dataset)} samples, number of batches is {len(test_loader)}', root=save_path)
        tester.plot_TL(10, test_loader)
        # tester.plot_IG(40, test_loader)
        # checkpoint = torch.load(args.pretrained, map_location=device)
        # train_losses = checkpoint['train_losses']
        # val_losses = checkpoint['val_losses']
        # tester.plot_losses([[epoch for (epoch, loss) in train_losses], [epoch for (epoch, loss) in val_losses]], # To be determined, 
        #                  [[loss for (epoch, loss) in train_losses], [loss for (epoch, loss) in val_losses]], # To be determined, 
        #                   'epoch', 'loss', 'figs_loss.pdf', ['train_loss', 'val_loss'])
        
        # train_epochs = [epoch for (epoch, loss) in train_losses]
        # train_loss = [loss for (epoch, loss) in train_losses]

        # val_epochs = [epoch for (epoch, loss) in val_losses]
        # val_loss = [loss for (epoch, loss) in val_losses]

        # loss_path = save_path + 'losses.mat'
        # scipy.io.savemat(loss_path, {
        #     'train_epochs': np.array(train_epochs),
        #     'train_loss': np.array(train_loss),
        #     'val_epochs': np.array(val_epochs),
        #     'val_loss': np.array(val_loss)
        # })
        return
    
    
    # Define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)
        
    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      alpha = args.alpha,
                      mask_MSE=args.mask_MSE,
                      mask_IG=args.mask_IG,
                      MSE_mask_range=args.MSE_mask_range,
                      MSE_mask_weight=args.MSE_mask_weight,
                      IG_mask_range=args.IG_mask_range,
                      IG_mask_weight=args.IG_mask_weight,
                      resume=args.resume,
                      save_path=save_path)
    
    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    # Final testing
    tester = Tester(model, device, criterion, args.alpha, 
                    args.mask_MSE, args.mask_IG, args.MSE_mask_range, args.MSE_mask_weight, args.IG_mask_range, args.IG_mask_weight,
                    save_path=save_path)
    loss, ssim, nmse = tester(test_loader)
    
    tester.plot_TL(10, test_loader)
    # tester.plot_IG(40, test_loader)

    logger.info(f"\n=! Final test loss: {loss:.3e}"
                f"\n         test ssim: {ssim:.3e}"
                f"\n         test nmse: {nmse:.3e}\n", root=save_path)

if __name__ == '__main__':
    main()