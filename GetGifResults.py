import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io as sio    
import datasets
import models
import utils
from PIL import Image 
import numpy as np

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


def eval_psnr(dataset, targetInd, center, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False,args = None, ImNumber = 0):
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


    
    # batch = dataset[targetInd]
    batch = dataset[targetInd]
    for k, v in batch.items():
        batch[k] = v.cuda()

    inp = (batch['inp'] - inp_sub) / inp_div
    print(inp.shape)
    ranges = [[-1, 1], [-1,1]]
    
    Bandnum = 100
    image_all = []

    for Scale_num in range(Bandnum):
        scale = 1+11/(Bandnum-1)*Scale_num
        ih, iw = inp.shape[-2:]
        # _, _, h_inp, w_inp = inp.size()
        for i in range(2):
            ranges[i] = [(center[i]-center[i]/scale)*2-1, ((1-center[i])/scale+center[i])*2-1]
        
        coord = utils.make_coord((ih, iw), ranges).unsqueeze(0).cuda()
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / inp.shape[-2] / scale
        cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        
        
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            pred = batched_predict(model, inp,
                coord, cell, eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
    
        if eval_type is not None: # reshape for shaving-eval
            pred = pred.view(inp.shape[0], ih, iw , 3) \
                .permute(0, 3, 1, 2).contiguous()
                
        # image_all = torch.cat((image_all,pred),dim = 0)
        pred = pred*255
        image_all.append(Image.fromarray(pred.permute([2,3,1,0]).squeeze(3).cpu().numpy().astype(np.uint8)))
        if Scale_num ==0:
            image_all2 = pred
        else:
            image_all2 = torch.cat((image_all2, pred), dim = 0)
    # image_all = image_all.permute([2,3,1,0]).cpu().numpy()
    image_all[0].save(args.save_path+'/Gif_Images/{:d}.gif'.format(targetInd), save_all=True, append_images=image_all[1:], bitrate=-1, duration=5, loop=10000)
    sio.savemat(args.save_path+'/Gif_Images/{:d}.mat'.format(targetInd), {'All_x': image_all2.permute(2,3,1,0).cpu().numpy()})


        # if args is not None:
        #     utils.save_results( args.save_path+'/Gif_Images/{:d}'.format(Scale_num), pred[0,:,:,:])
        #     sio.savemat(args.save_path+'/Gif_Images/{:d}.mat'.format(Scale_num), {'output': pred[0,:,:,:].cpu().numpy()})
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default = 'configs/test/test-urban100-4.yaml')
    parser.add_argument('--model', default = 'save_local/_train_edsr-baseline-liif_rot/epoch-best.pth')
    parser.add_argument('--device', default='0')
    parser.add_argument("-i", '--targetInd', default='10')
    parser.add_argument('--ch', default='0.5')
    parser.add_argument('--cw', default='0.5')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    save_name = args.name
    if save_name is None:
        save_name =  args.model.split('/')
        save_name = save_name[0]+'/'+ save_name[1]
    print(save_name)
    save_path = os.path.join('./', save_name)
    print(save_path)
    args.save_path = save_path
    if not os.path.exists(save_path+'/Gif_Images'):
        os.mkdir(save_path+'/Gif_Images')
        print('mkdir', save_path+'/Gif_Images')
    

    res = eval_psnr(dataset, int(args.targetInd), [float(args.ch), float(args.cw)],  model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True, args = args)
