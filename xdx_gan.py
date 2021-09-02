#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

data_path = 'dataset_gan/data/' # 数据集存放路径
num_workers = 1 # 多进程加载数据所用的进程数
image_size = 96 # 图片尺寸
batch_size = 128
max_epoch =  20000
lr1 = 2e-4 # 生成器的学习率
lr2 = 2e-4 # 判别器的学习率
beta1=0.5 # Adam优化器的beta1参数
gpu=False # 是否使用GPU --nogpu或者--gpu=False不使用gpu
nz=100 # 噪声维度
ngf = 64 # 生成器feature map数
ndf = 64 # 判别器feature map数

save_path = 'dataset_gan/train_result/' #训练时生成图片保存路径

# vis = False # 是否使用visdom可视化
# env = 'GAN' # visdom的env
plot_every = 20 # 每间隔20 batch，visdom画图一次
save_every = 3
# debug_file='/tmp/debuggan' # 存在该文件则进入debug模式
d_every=1 # 每1个batch训练一次判别器
g_every=5 # 每5个batch训练一次生成器
decay_every=10 # 没10个epoch保存一次模型
netd_path = 'dataset_gan/models/netd_last.pth' #预训练模型
netg_path = 'dataset_gan/models/netg_last.pth'

# 只测试不训练
gen_img = 'result.png'
# 从512张生成的图片中保存最好的64张
gen_num = 64
gen_search_num = 512
gen_mean = 0 # 噪声的均值
gen_std = 1 #噪声的方差

# ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)

class Generator(nn.Module):
    """
    定义一个生成模型，通过输入噪声来产生一张图片
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 假定输入为一张1x1xnz维的数据(nz维的向量)
            nn.ConvTranspose2d(nz , ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # 输入一个４*4*ngf*8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 输入一个8*8*ngf*4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 输入一个16*16*ngf*2
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # 输入一个32*32*ngf
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # 输出一张96*96*3
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """
    构建一个判别器，相当与一个二分类问题, 生成一个值
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 输入96*96*3
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 输入32*32*ndf
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),

            # 输入16*16*ndf*2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),

            # 输入为8*8*ndf*4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),

            # 输入为4*4*ndf*8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()  # 分类问题
        )

    def forward(self, x):
        return self.main(x).view(-1)

def train():
    """training NetWork"""

    device = torch.device("cuda") if gpu else torch.device("cpu")

    # 1.预处理数据
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),  # 3*96*96
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #  1.1 加载数据
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)  # TODO 复习这个封装方法
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)  # TODO 查看drop_last操作

    # 2．初始化网络
    netg = Generator()
    netd = Discriminator()
    # 2.1判断网络是否已有权重数值
    map_location = lambda storage, loc: storage  # TODO 复习map_location操作

    if netg_path:
        netg.load_state_dict(torch.load(netg_path, map_location=map_location))
    if netd_path:
        netd.load_state_dict(torch.load(netd_path, map_location=map_location))
    # 2.2 搬移模型到指定设备
    netd.to(device)
    netg.to(device)

    # 3. 定义优化策略
    #  TODO 复习Adam算法
    optimize_g = torch.optim.Adam(netg.parameters(), lr=lr1, betas=(beta1,0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=lr2, betas=(beta1, 0.999))
    criterions = nn.BCELoss().to(device)  # TODO 重新复习BCELoss方法

    # 4. 定义标签, 并且开始注入生成器的输入noise
    true_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)
    noises = torch.randn(batch_size, nz, 1, 1).to(device)
    fix_noises = torch.randn(batch_size, nz, 1, 1).to(device)

#     errord_meter = AverageValueMeter()  # TODO 重新阅读torchnet
#     errorg_meter = AverageValueMeter()

    #  6.训练网络
    epochs = range(max_epoch)
#     write = SummaryWriter(log_dir=virs, comment='loss')

    # 6.1 设置迭代
    for epoch in iter(epochs):
        #  6.2 读取每一个batch 数据
        for ii_, (img, _) in enumerate(dataloader):#tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            #  6.3开始训练生成器和判别器
            #  注意要使得生成的训练次数小于一些
            if ii_ % d_every == 0:
                optimize_d.zero_grad()
                # 训练判别器
                # 真图
                output = netd(real_img)
                error_d_real = criterions(output, true_labels)
                error_d_real.backward()

                # 随机生成的假图
                noises = noises.detach()
                fake_image = netg(noises).detach()
                output = netd(fake_image)
                error_d_fake = criterions(output, fake_labels)
                error_d_fake.backward()
                optimize_d.step()

                # 计算loss
                error_d = error_d_fake + error_d_real
                print("[{}-{}]/{} Discriminator loss_real:{} loss_fake:{} loss_total:{}".format(ii_, epoch, max_epoch, error_d_real.item(), 
                                                                                                error_d_fake.item(), error_d.item()))
#                 errord_meter.add(error_d.item())

            # 训练判别器
            if ii_ % g_every == 0:
                optimize_g.zero_grad()
                noises.data.copy_(torch.randn(batch_size, nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterions(output, true_labels)
                error_g.backward()
                optimize_g.step()
                print("[{}-{}]/{} Generator loss:{}".format(ii_, epoch, max_epoch, error_g.item()))
#                 errorg_meter.add(error_g.item())
#             # 绘制数据
#             if ii_ % 5 == 0:
#                 write.add_scalar("Discriminator_loss", errord_meter.value()[0])
#                 write.add_scalar("Generator_loss", errorg_meter.value()[0])

        #  7.保存模型
        if (epoch + 1) % save_every == 0:
            fix_fake_image = netg(fix_noises)
            torchvision.utils.save_image(fix_fake_image.data[:64], "%s/second_%s.png" % (
                save_path, epoch), normalize=True)

            torch.save(netd.state_dict(), 'dataset_gan/models/netd_%s.pth' % epoch)
            torch.save(netg.state_dict(), 'dataset_gan/models/netg_%s.pth' % epoch)
#             errord_meter.reset()
#             errorg_meter.reset()

#     write.close()

if __name__ == "__main__":
    train()
