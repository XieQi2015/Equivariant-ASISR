""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""


import warnings
warnings.filterwarnings("ignore")

def contrast_enhancement(x, contrast_rito = 2, cR = 0.025):
    gausFilter = utils.create_gaussian_kernel(21, 5)
    padX = torch.cat((torch.flip(x, [2])[:,:,-10:,:], x, torch.flip(x, [2])[:,:,:10,:]), 2)
    padX = torch.cat((torch.flip(padX, [3])[:,:,:,-10:], padX, torch.flip(padX, [3])[:,:,:,:10]), 3)
    y = x - F.conv2d(padX.permute(1,0,2,3), gausFilter.cuda(), padding = 0).permute(1,0,2,3)
    y = (utils.setRange(y, y.mean()+cR, y.mean()-cR)-0.5)*contrast_rito+0.5
    print(contrast_rito)
    return y.clamp_(0, 1)

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    Im_loader = make_data_loader(config.get('Im_dataset'), tag='Imaging')
    return train_loader, val_loader, Im_loader

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False,args = None, ImNumber = 0, contrast_rito = 2):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()


    batch_num = 0
    
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        # inp = batch['inp']
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        if args is not None:
                pred = contrast_enhancement(pred, contrast_rito)
                utils.save_results( args.save_path+'/{:d}_{:d}_SR'.format(ImNumber, batch_num), pred[0,:,:,:])
                # sio.savemat(args.save_path+'/temp_Images/{:d}_{:d}_SR.mat'.format(ImNumber, batch_num), {'Y': pred[0,:,:,:].cpu().numpy(),'X': batch['gt'][0,:,:,:].cpu().numpy()})
                if ImNumber==0:
                    utils.save_results( args.save_path+'/{:d}_{:d}_HR'.format(ImNumber, batch_num), batch['gt'][0,:,:,:])
        batch_num = batch_num+1
    return 

def main(config_, save_path, args=None):
    global config, log#, writer
    config = config_
#    log, writer = utils.set_save_path(save_path)
    log = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader, Im_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

        #show the temp images for observe the initail performance

    args.save_path = save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('mkdir', save_path)
    
    eval_psnr(Im_loader, model,
    data_norm=config['data_norm'],
    eval_type=config.get('eval_type'),
    eval_bsize=config.get('eval_bsize'), 
    args = args, ImNumber = 1-1, contrast_rito = config.get('contrast_rito'))

    
        #writer.flush()


if __name__ == '__main__':
    
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = 'configs/observation/Observe-edsr-baseline-liif-eq.yaml' )
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='1')
    parser.add_argument('--device', default='0')
    parser.add_argument('--show_tempImage', default=True)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    import yaml
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    
    import datasets
    import models
    import utils
    # from test_Observe_EQ import eval_psnr
    import scipy.io as sio    


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    print(save_path)

    main(config, save_path, args)
