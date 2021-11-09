import torch
from torch.utils.data.dataset import T
import torchvision.transforms as transforms
from torchstat import stat

import os, argparse, sys
from datetime import datetime

from model import LGFA
from dataset import ImageGroundTruthFolder

from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default='input test data path', help='path to datasets, default = ./testData')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--res_mod', default=None, help='Path to the model to resume from', type=str)
parser.add_argument('--imgres', type=int, default=256, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=500, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate,  default = 0.0001')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size,  default = 10')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin, default = 0.5')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate, default = 0.1')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate,  default = 30')
args = parser.parse_args()

viz = Visdom()

def train(train_loader, model, optimizer, epoch):

    total_steps = len(train_loader)
    CE = torch.nn.BCEWithLogitsLoss()
    model.train()
    for step, pack in enumerate(train_loader, start=1):
        global_step = (epoch-1) * total_steps + step
        optimizer.zero_grad()
        imgs, gts, _, _, _ = pack
        imgs = imgs.to(device)
        gts = gts.to(device)

        preds = model(imgs)
        loss = CE(preds, gts)

        # save each loss
        totxt = str(epoch) + ' ' + str(step) + ' ' +  str(loss.item()) +'\n'
        f.write(totxt)
        f.flush()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # show results in training
        viz.images(imgs,win='imputs')
        viz.images(gts,win='gts')
        viz.images(preds,win='preds')

        sys.stdout.write('\r {0}/{1}'.format(step*epoch,total_steps*epoch))
        if step % 20 == 0 or step == total_steps:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, step, total_steps, loss.data))
    
    save_path = 'ckpts/{}/'.format(model.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 1 == 0:
        torch.save(model.state_dict(), '{}{}.pth.{:03d}'.format(save_path, model.name, epoch))

device = torch.device(args.device)
print('Device: {}'.format(device))

model = LGFA().to(device)

# Load model and optimizer if resumed
if args.res_mod is not None:
    model.load_state_dict(torch.load(args.res_mod, map_location=torch.device(device)))
    print("Resuming training with checkpoint : {}\n".format(args.res_mod))

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('{}\t{}'.format(model.name, params))

optimizer = torch.optim.Adam(model.parameters(), args.lr)
# optimizer = torch.optim.SGD(model.parameters(),args.lr)

# load dataset
transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

dataset = ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

save_loss_path = './results/' + model.name + '.txt'
if not save_loss_path:
    os.makedirs('./results/',exist_ok=True)
f = open(save_loss_path,'w',encoding='utf-8')


print('Dataset loaded successfully')
for epoch in range(1, args.epoch+1):
    print('Started epoch {:03d}/{}'.format(epoch, args.epoch))
    lr_lambda = lambda epoch: args.decay_rate ** (epoch // args.decay_epoch)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    train(train_loader, model, optimizer, epoch)
f.close()
