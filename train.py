import argparse

from torchvision import transforms
import torch
import torch.nn as nn
import time
import sys
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CMTFusion
import utils
import losses
import torch.nn.functional as F
import os
from thop import profile



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dataset', type=str, default='./visible_20000/', help='path of rgb dataset')
    parser.add_argument('--ir_dataset', type=str, default='./lwir_20000/', help='path of ir dataset')
    parser.add_argument('--test_images', type=str, default='test_images', help='path of image visualization')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval of saving image')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval of saving model')
    parser.add_argument('--b1', type=float, default=0.5, help='fixed!')
    parser.add_argument('--b2', type=float, default=0.999, help='fixed!')
    args = parser.parse_args()

    # -----------------
    # device setting
    # -----------------

    ######### device setting for gpu users #########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    torch.backends.cudnn.enabled = False

    ######### DataLoaders ###########

    trans = transforms.Compose(
        [transforms.RandomCrop(256),
         transforms.ToTensor(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Normalize((0.5,), (0.5,))])

    dataset = utils.Customdataset(transform=trans, rgb_dataset=args.rgb_dataset, ir_dataset=args.ir_dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print('===> Loading datasets')

    ######### Model ###########

    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    torch.cuda.set_device('cuda:0')
    fusion_model = nn.DataParallel(CMTFusion(), device_ids=[0,1,2])  # [0], device_ids
    # fusion_model = CMTFusion()
    fusion_model.cuda()

    optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

    ######### Loss ###########
    MSE_loss = nn.MSELoss().cuda()
    loss_p = losses.perceptual_loss().cuda()
    loss_spa = losses.L_spa().cuda()
    loss_fre = losses.frequency().cuda()

    prev_time = time.time()
    out_path = './fusion_outputs/'
    if os.path.exists(out_path) is False:
        os.mkdir(out_path)

    for epoch in range(args.epochs):
        ######### Train ###########
        fusion_model.train()
        for i, (rgb_target, ir_target) in enumerate(train_dataloader):
            # zero_grad
            for param in fusion_model.parameters():
                param.grad = None

            # Configure input
            real_rgb_imgs = rgb_target.cuda()
            real_ir_imgs = ir_target.cuda()
            real_ir_imgs_2 = F.interpolate(real_ir_imgs, scale_factor=0.5, mode='bilinear')
            real_ir_imgs_3 = F.interpolate(real_ir_imgs_2, scale_factor=0.5, mode='bilinear')
            real_rgb_imgs_2 = F.interpolate(real_rgb_imgs, scale_factor=0.5, mode='bilinear')
            real_rgb_imgs_3 = F.interpolate(real_rgb_imgs_2, scale_factor=0.5, mode='bilinear')

            fake_imgs1, fake_imgs2, fake_imgs3 = fusion_model(real_rgb_imgs, real_ir_imgs)

            mse_loss = MSE_loss(input=fake_imgs3.cuda(), target=real_rgb_imgs_3.cuda()) \
                       + MSE_loss(input=fake_imgs3.cuda(), target=real_ir_imgs_3.cuda()) \
                       + MSE_loss(input=fake_imgs2.cuda(), target=real_rgb_imgs_2.cuda()) \
                       + MSE_loss(input=fake_imgs2.cuda(), target=real_ir_imgs_2.cuda()) \
                       + MSE_loss(input=fake_imgs1.cuda(), target=real_rgb_imgs.cuda()) \
                       + MSE_loss(input=fake_imgs1.cuda(), target=real_ir_imgs.cuda())

            fre_loss = loss_fre(fake_imgs1, real_rgb_imgs.cuda(), real_ir_imgs.cuda())
            spa_loss = 0.5 * torch.mean(loss_spa(fake_imgs1, real_rgb_imgs)) + 0.5 * torch.mean(
                loss_spa(fake_imgs1, real_ir_imgs))
            loss_per = 0.5 * loss_p(fake_imgs1, real_rgb_imgs.cuda()) + 0.5 * loss_p(fake_imgs1, real_ir_imgs.cuda())
            fuse_loss = (mse_loss / 6) + 0.8 * spa_loss + 0.02 * loss_per + 0.05 * fre_loss
            total_loss = fuse_loss

            total_loss.backward()
            optimizer.step()

            batches_done = epoch * len(train_dataloader) + i
            batches_left = args.epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\rTrain : [Epoch %d/%d] [Batch %d/%d] [fuse_loss: %f] [mse_loss: %f] [spa_loss: %f] [per_loss: %f]  ETA: %s"
                % (
                    epoch,
                    args.epochs,
                    i,
                    len(train_dataloader),
                    fuse_loss.item(),
                    mse_loss.item(),
                    spa_loss.item(),
                    loss_per.item(),
                    time_left,
                )
            )

        torch.save(fusion_model.state_dict(), "./saved_models/%s/model_fusion%d.pth" % ("basic", epoch))



if __name__ == "__main__":
    main()
