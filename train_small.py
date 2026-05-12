import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.simple_detector import SimpleDetector, count_parameters
from datasets.synth_dataset import SynthDetectionDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', default='data/images_generee')
    p.add_argument('--placements_csv', default=None)
    p.add_argument('--src_crops', default='data/images_crop')
    p.add_argument('--img_w', type=int, default=800)
    p.add_argument('--img_h', type=int, default=532)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args()


def simple_loss(pred, targets, num_classes):
    # pred: B x C x H x W
    # split channels
    b, c, h, w = pred.shape
    # channels organization: [obj(1), cls(num_classes), dx,dy, sin,cos]
    obj = torch.sigmoid(pred[:, 0:1])
    cls_logits = pred[:, 1:1+num_classes]
    offs = pred[:, 1+num_classes:1+num_classes+2]
    ang = pred[:, 1+num_classes+2:1+num_classes+4]

    # flatten and compute simple losses
    obj_target = targets['obj'].unsqueeze(0) if len(targets['obj'].shape)==3 else targets['obj']
    obj_target = obj_target.to(pred.device)

    bce = nn.BCELoss()
    loss_obj = bce(obj, obj_target)

    # class loss (only where obj==1)
    cls_target = targets['cls'].to(pred.device)
    # upsample targets to match batch dim
    # For simplicity compute class loss only where object present across batch
    ce = nn.CrossEntropyLoss()
    # reduce preds to shape B x num_classes x H x W
    cls_logits = cls_logits
    # pick a single location per image (approx) — simplified for quick test
    # compute cls loss using all cells but masked
    mask = obj_target > 0.5
    if mask.sum() > 0:
        cls_loss = ce(cls_logits.permute(0,2,3,1)[mask.repeat(1,1,1)], cls_target[mask])
    else:
        cls_loss = torch.tensor(0.0, device=pred.device)

    # offsets and angle L1 only where object
    offs_target = targets['offs'].to(pred.device)
    ang_target = targets['ang'].to(pred.device)
    l1 = nn.L1Loss()
    if mask.sum() > 0:
        offs_pred = offs.permute(0,2,3,1)[mask.repeat(1,1,1)]
        offs_t = offs_target[mask]
        loss_offs = l1(offs_pred, offs_t)

        ang_pred = ang.permute(0,2,3,1)[mask.repeat(1,1,1)]
        ang_t = ang_target[mask]
        loss_ang = l1(ang_pred, ang_t)
    else:
        loss_offs = torch.tensor(0.0, device=pred.device)
        loss_ang = torch.tensor(0.0, device=pred.device)

    loss = loss_obj + cls_loss + loss_offs + loss_ang
    return loss


def train():
    args = parse_args()
    # simple dataset: expects placements CSV already created
    if args.placements_csv is None:
        # auto-choose the latest placements file in images_dir
        p = Path(args.images_dir)
        csvs = sorted(p.glob('placements_*.csv'))
        if not csvs:
            raise RuntimeError('No placements CSV found in images_dir')
        args.placements_csv = str(csvs[-1])

    ds = SynthDetectionDataset(images_dir=args.images_dir, placements_csv=args.placements_csv,
                              src_crops_dir=args.src_crops, img_size=(args.img_w, args.img_h))

    num_classes = max(1, len(ds.crop_sizes))
    print('Num samples:', len(ds), 'Num classes (detected):', num_classes)
    model = SimpleDetector(num_classes=num_classes)
    print('Model params:', count_parameters(model))

    device = torch.device('cpu')
    model.to(device)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (img, target) in enumerate(dl):
            img = img.to(device)
            out = model(img)
            # wrap targets to batch (simple repeat)
            # Build a batch-like targets dict for loss fn
            b_targets = {k: v for k, v in target.items()}
            # naive: add batch dim to targets by repeating
            for k in b_targets:
                b_targets[k] = b_targets[k]
                b_targets[k] = b_targets[k].unsqueeze(0).repeat(img.shape[0], 1, 1, *([] if k in ['obj','cls'] else [])) if False else b_targets[k]
            loss = simple_loss(out, target, num_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch} iter {i+1}/{len(dl)} loss {running/(i+1):.4f}')
        print(f'Epoch {epoch} finished, avg loss {running/len(dl):.4f}, time {time.time()-t0:.1f}s')

    torch.save(model.state_dict(), 'models/detector_small.pth')
    print('Saved model to models/detector_small.pth')


if __name__ == '__main__':
    train()
