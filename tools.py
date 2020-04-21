import apex
from apex import amp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
import os

from cvcore.data import cutmix_data, mixup_data
from cvcore.utils import AverageMeter, save_checkpoint


def valid_model(_print, cfg, model, valid_loader,
                optimizer, epoch,
                best_metric=None, checkpoint=False):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, lb) in enumerate(tbar):
            image = image.cuda()
            lb = lb.cuda()
            w_output = torch.nn.functional.softmax(model(image), 1)

            preds.append(w_output.cpu())
            targets.append(lb.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    # record
    val_loss = torch.nn.functional.cross_entropy(preds, targets)
    score = accuracy_score(targets, torch.argmax(preds, 1))

    _print("VAL LOSS: %.5f, SCORE: %.5f"  % (val_loss, score))
    # checkpoint
    if checkpoint:
        is_best = score < best_metric
        best_metric = min(score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.EXP,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric}
        save_filename = f"{cfg.EXP}.pth"
        if is_best: # only save best checkpoint, no need resume
            save_checkpoint(save_dict, is_best,
                            root=cfg.DIRS.WEIGHTS, filename=save_filename)
        return score, best_metric


def test_model(cfg, args, model, test_loader):
    # switch to evaluate mode
    model.eval()

    preds = []
    names = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, name) in enumerate(tbar):
            image = image.cuda()
            image_lr = torch.flip(image, [-1])
            w_output = (torch.sigmoid(model(image)) + torch.sigmoid(model(image_lr))) / 2.0

            preds.append(w_output.cpu())
            for n in name:
                names.append(n)

    preds = torch.cat(preds, 0).numpy()
    names = np.array(names)

    np.save(os.path.join(cfg.DIRS.OUTPUTS, f"{args.folder}", cfg.EXP + f"_fold{args.fold}.npy"), preds)
    np.save(os.path.join(cfg.DIRS.OUTPUTS, "names.npy"), names)




def train_loop(_print, cfg, model, train_loader,
               criterion, optimizer, scheduler, epoch):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    for i, (image, target) in enumerate(tbar):
        image = image.cuda()
        target = target.cuda()

        # mixup/ cutmix
        if cfg.DATA.MIXUP:
            image = mixup_data(image, alpha=cfg.DATA.CM_ALPHA)
        elif cfg.DATA.CUTMIX:
            image = cutmix_data(image, alpha=cfg.DATA.CM_ALPHA)
        w_output = model(image)
        # compute loss
        # loss = criterion(w_output[:,0], leaf_target) + criterion(w_output[:,1], stem_target) + criterion(w_output[:,2], healthy_target)
        loss = criterion(w_output, target)
        # gradient accumulation
        loss = loss / cfg.OPT.GD_STEPS
        if cfg.SYSTEM.FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.OPT.GD_STEPS == 0:
            scheduler(optimizer, i, epoch)
            optimizer.step()
            optimizer.zero_grad()
        # record loss
        losses.update(loss.item() * cfg.OPT.GD_STEPS, target.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _, _, _) in enumerate(tbar):
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))