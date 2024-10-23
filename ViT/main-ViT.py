
# torchrun --standalone --nnodes=1 --nproc_per_node=8  ./main-ViT.py -cfg=config/ViT.yaml -expName=ViT

from model_ViT import SVTx
import torch, argparse, os, time, sys, shutil, yaml, scipy.io
from util import save2img, restore_and_save2tiff
from data import SinogramDataset
import numpy as np
import logging
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='ViT_sino')
parser.add_argument('-gpus',   type=str, default="1", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="ViTtrain-simu", help='Experiment name')
parser.add_argument('-cfg',    type=str, default="config/ViT.yaml", help='path to config yaml file')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def main(args):

    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir):
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    logging.basicConfig(filename=f"{args.expName}-itrOut/SVTx.log", level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)s %(module)s: %(message)s')
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # training task init
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)

    train_ds = SinogramDataset(ifn=params['dataset']['th5'], params=params)
    train_dl = DataLoader(train_ds, batch_size=params['train']['mbsz'], \
                          pin_memory=True, drop_last=True, shuffle=True)
    logging.info("Init validation dataset ...")
    valid_ds = SinogramDataset(ifn=params['dataset']['vh5'], params=params)
    valid_dl = DataLoader(valid_ds, batch_size=params['train']['mbsz'], \
                         pin_memory=True, \
                          drop_last=False, shuffle=False)
    logging.info(f"%d samples, {valid_ds.shape},  will be used for validation" % (len(valid_ds), ))

    model = SVTx(seqlen=train_ds.seqlen, in_dim=train_ds.cdim, params=params).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['train']['lr'], betas=(0.9, 0.95))

    train_loss = []
    val_loss = []

    for ep in range(1, params['train']['maxep'] + 1):
        train_ep_tick = time.time()
        for x_train, y_train in train_dl:
            optimizer.zero_grad()
            loss, pred, mask = model.forward(x_train.cuda(), y_train.cuda())
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()

        time_e2e = time.time() - train_ep_tick
        itr_prints = '[Train] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch, %d steps with lbs=%d' % ( \
            ep, loss.cpu().detach().numpy(), time_e2e, len(train_dl), x_train.shape[0])
        logging.info(itr_prints)
        train_loss.append(loss.cpu().detach().numpy())
        scipy.io.savemat('%s/train_loss.mat' % (itr_out_dir), {'train_loss': np.array(train_loss)})

        loss_v = []
        valid_ep_tick = time.time()
        for x_val, y_val in valid_dl:
            with torch.no_grad():
                _vloss, _vpred, _vmask = model.forward(x_val.cuda(), y_val.cuda())
                loss_v.append(_vloss.cpu().numpy())
        val_loss.append(np.mean(loss_v))
        scipy.io.savemat('%s/val_loss.mat' % (itr_out_dir), {'val_loss': np.array(val_loss)})

        valid_e2e = time.time() - valid_ep_tick
        _prints = '[Valid] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch\n' % (ep, np.mean(loss_v), valid_e2e)
        logging.info(_prints)

        image_train_gt = train_ds.target[train_ds.len // 2] ## -1 for the last slice
        image_train_ns = train_ds.sample[train_ds.len // 2]  ## -1 for the last slice
        image_valid_gt = valid_ds.target[valid_ds.len // 2]  ## -1 for the last slice
        image_valid_ns = valid_ds.sample[valid_ds.len // 2]  ## -1 for the last slice

        if ep == 1:
            save2img(image_train_gt, '%s/ep%05d-train-gt.tiff' % (itr_out_dir, ep))
            save2img(image_train_ns, '%s/ep%05d-train-ns.tiff' % (itr_out_dir, ep))
            save2img(image_valid_gt, '%s/ep%05d-valid-gt.tiff' % (itr_out_dir, ep))
            save2img(image_valid_ns, '%s/ep%05d-valid-ns.tiff' % (itr_out_dir, ep))

        with torch.no_grad():
            _tloss, _tpred, _tmask = model.forward(torch.tensor(image_train_ns).unsqueeze(0).unsqueeze(0).cuda(),
                                               torch.tensor(image_train_gt).unsqueeze(0).unsqueeze(0).cuda())
            _vloss, _vpred, _vmask = model.forward(torch.tensor(image_valid_ns).unsqueeze(0).unsqueeze(0).cuda(),
                                               torch.tensor(image_valid_gt).unsqueeze(0).unsqueeze(0).cuda())

        if ep % params['train']['ckp_steps'] != 0: continue
        restore_and_save2tiff(pred=_tpred.cpu().numpy().squeeze(), ori=np.array(image_train_gt), \
                              mask=_tmask.cpu().numpy().squeeze(), fn='%s/ep%05d-train-pd.tiff' % (itr_out_dir, ep))
        restore_and_save2tiff(pred=_vpred.cpu().numpy().squeeze(), ori=np.array(image_valid_gt), \
                              mask=_vmask.cpu().numpy().squeeze(), fn='%s/ep%05d-valid-pd.tiff' % (itr_out_dir, ep))


        torch.save(model.state_dict(), "%s/mdl-ep%05d.pth" % (itr_out_dir, ep))
        # torch.jit.save(torch.jit.trace(model, (imgs_tr[:1].cuda(), 0.7)), "%s/script-ep%05d.pth" % (itr_out_dir, ep))
        with open(f'{itr_out_dir}/config.yaml', 'w') as fp:
            yaml.dump(params, fp)

if __name__ == "__main__":
    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable printing INFO, WARNING, and ERROR

    main(args)
