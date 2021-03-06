import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as utils
import torchvision.transforms as transforms

import time
import os, argparse

from dataset import ImageGroundTruthFolder
from model import LGFA


parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='input test data path', help='path to datasets, default = ./datasets/test')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results, default = ./results')
parser.add_argument('--pth', type=str, default='./ckpts/***.pth.008', help='model filename, default = CPD.pth')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--imgres', type=int, default=256, help='image input and output resolution, default = 352')
parser.add_argument('--time', action='store_true', default=False)
args = parser.parse_args()

device = torch.device(args.device)
print('Device: {}'.format(device))


model = LGFA().to(device)
model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))
model.eval()
print('Loaded:', model.name)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

if args.time:
    model.eval()
    with torch.no_grad():
        n = 1000
        input = torch.rand([n, 1, 3, args.imgres, args.imgres]).to(device)
        t0 = time.time()
        with torch.autograd.profiler.profile() as prof:
            for img in input:
                pred = model(img)
        avg_t = (time.time() - t0) / n
    print('Inference time', avg_t)
    print('FPS', 1/avg_t)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

else:
    dataset = ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for pack in test_loader:
        img, _, dataset, img_name, img_res = pack
        print('{} - {}'.format(dataset[0], img_name[0]))
        img = img.to(device)

        pred = model(img)

        pred = F.interpolate(pred, size=img_res[::-1], mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu()

        save_path = './results/{}/{}/accordion/'.format(model.name, dataset[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = '{}{}.png'.format(save_path, img_name[0])
        # filename_att = '{}{}.jpg'.format(save_path, img_name[0])
        utils.save_image(pred,  filename)
        # utils.save_image(att,  filename_att)
