import warnings
warnings.filterwarnings("ignore")

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        # if model.scale_decode is not None or model.scale_encode is not None:
        #     model.get_scale(inp.shape,cell)

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


def eval_psnr(loader, model, args, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False,window_size = None):
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

    if eval_type is None:
        # metric_fn = utils.calc_psnr
        metric_fn = utils.calc_psnr_ssim
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr_ssim, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr_ssim, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_psnr = utils.Averager()
    val_ssim = utils.Averager()
    val_time = utils.Averager()
    if args.save_results:
        Im_num = 0
        if not os.path.exists('save/'+args.model+'/'+args.data):
            os.mkdir('save/'+args.model+'/'+args.data)
            print('mkdir', 'save/'+args.model+'/'+args.data)    

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        
        if window_size != None:
            # if eval_type is None:
            #     scale = float(torch.mean(batch['scale']))

            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
            # print('window_size = {:d}'.format(window_size))
        else:
            h_pad = 0
            w_pad = 0
            # print('window_size = None')
            coord = batch['coord']
            cell = batch['cell']
        
        
        if eval_bsize is None:
            with torch.no_grad():
                tic = time.time()
                pred = model(inp, coord, cell)
                toc = time.time()
        else:
            tic = time.time()
            pred = batched_predict(model, inp,
                coord, cell, eval_bsize)
            toc = time.time()
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]

            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
        


        psnr, ssim = metric_fn(pred, batch['gt'])
        val_psnr.add(psnr.item(), inp.shape[0])
        val_ssim.add(ssim.item(), inp.shape[0])
        val_time.add(toc-tic, inp.shape[0])
        
        if args.save_results:
            Im_num =Im_num+1
            utils.save_results( 'save/'+args.model+'/'+args.data+'/{:d}'.format(Im_num), pred[0,:,:,:])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_psnr.item()))

    return val_psnr.item(), val_ssim.item(), val_time.item()


if __name__ == '__main__':

    import argparse
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default = 'urban100-4')
#    parser.add_argument('--config',default = 'configs/test/test-urban100-4.yaml')
#    parser.add_argument('--model', default = 'save/_proposed_edsr-baseline-liif/epoch-best.pth')
    parser.add_argument('--model', default = '_train_edsr-baseline-liif')
    parser.add_argument('--epoch', default = 'best')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    
    import math
    from functools import partial

    import yaml
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import datasets
    import models
    import utils
    import scipy.io as sio    
    import time
    

    with open('configs/test/test-'+args.data+'.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load('save/'+args.model+'/epoch-'+args.epoch+'.pth')['model']
    model = models.make(model_spec, load_sd=True).cuda()

    psrn, ssim, tim = eval_psnr(loader, model, args,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize= config.get('eval_bsize'),
        window_size=8,
        verbose=True)
#    print('result: {:.4f}'.format(res))
    
    log = args.data + ': PSNR: {:.4f},'.format(psrn) + ' SSIM: {:.4f},'.format(ssim)+ ' Time: {:.4f}.'.format(tim)
    
    open_type = 'a' if os.path.exists('save/'+args.model+'/testResult.txt')else 'w'
    log_file = open('save/'+args.model+'/testResult.txt', open_type)
    print(log)
    log_file.write(log + '\n')
    sio.savemat('save/'+args.model+'/'+args.data+'.mat', {'psnr': psrn, 'ssim': ssim, 'times': tim})
    
