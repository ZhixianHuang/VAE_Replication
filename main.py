import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optimizer
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# VAE Model
class VAE_Model(nn.Module):
    # input -->  Encoder  --> latent space(noise adding) --> Decoder  --> output (reconstruction of input)
    def __init__(self, input, feature, latent, output):
        super().__init__()
        self.input_dim = input
        self.feature_dim = feature
        self.latent_dim = latent
        self.output_dim = output
        self.Linear1 = nn.Linear(self.input_dim, self.feature_dim)
        # the Encoder of VAE gives two outputs which is mean and std(?)
        # !!! However keep in mind !!! mean and standard are not calculated but trained from Encoder!
        # mean
        self.Linear2 = nn.Linear(self.feature_dim,self.latent_dim)
        # std
        self.Linear3 = nn.Linear(self.feature_dim,self.latent_dim)
        self.Linear4 = nn.Linear(self.latent_dim, self.feature_dim)
        self.Linear5 = nn.Linear(self.feature_dim, self.output_dim)
        '''
        nn.Sequential(nn.Linear(self.input_dim, self.feature_dim),
                      nn.Linear(self.feature_dim,self.latent_dim),
                      nn.Linear(self.latent_dim, self.feature_dim),
                      nn.Linear(self.feature_dim, self.output_dim))
        '''

    def encoder(self, x):
        features = nn.functional.relu(self.Linear1(x))
        return self.Linear2(features), self.Linear3(features)

    def reparameterize_trick(self, x_mean , x_std):
        # z = μ + σ * ε  with epsilon \thicksim\mathcal{N}(0, 1) is so called reparameterize trick
        std = torch.exp(x_std)
        epsilon = torch.randn_like(std)
        z = x_mean + std * epsilon
        return z

    def decoder(self, x):
        features = nn.functional.relu(self.Linear4(x))
        return nn.functional.sigmoid(self.Linear5(features))

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize_trick(mean, var)
        x_recon = self.decoder(z)
        return x_recon, mean, var


# initialization
epochs = 15
batch_size = 128
learning_rate = 1e-3

# MNIST image 28*28
input_dim = 784
output_dim = input_dim
latent_dim = 20
feature_dim = 400

# create dataset and dataloader
train_dataset = datasets.MNIST(root='../dataset/MNIST', download=True, train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device}')
model = VAE_Model(input_dim, feature_dim, latent_dim, output_dim)
model = model.to(device)
optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter('./logs')


def train():
    for epoch in range(epochs):
        loss_sum = 0
        recon_loss_sum = 0
        kl_div_sum = 0

        for i, (x,_) in enumerate(train_dataloader):
            x = x.to(device).view(-1, input_dim)
            x_recon, mean, std = model(x)

            reconst_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
            # D_{KL}(\mathcal{N}(μ,σ^2)||\mathcal{N}(0,1)) = 1/2 *(σ^2+μ^2-1-log(σ^2))
            kl_divergence = 0.5 * torch.sum(std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)))

            loss = reconst_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            recon_loss_sum += reconst_loss.item()
            kl_div_sum += kl_divergence.item()

        avg_loss = loss_sum / len(train_dataset)
        avg_recon_loss = recon_loss_sum / len(train_dataset)
        avg_kl_div = kl_div_sum / len(train_dataset)

        writer.add_scalar('avg_loss', avg_loss, epoch)
        writer.add_scalar('avg_recon_loss', avg_recon_loss, epoch)
        writer.add_scalar('avg_kl_div', avg_kl_div, epoch)

        print(f"Epoch[{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg KL Div: {avg_kl_div:.4f}")

    with torch.no_grad():
        # randomly sample from latent space
        # generate images using samples
        z = torch.randn(batch_size, latent_dim).to(device)
        out = model.decoder(z).view(-1, 1, 28, 28)

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Sampled Images")
        plt.imshow(np.transpose(torchvision.utils.make_grid(out, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()

        # reconstruction quality check
        # x as input image from last batch of dataloader and compared with it's output passed throught network
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Reconstructed Images")
        plt.imshow(np.transpose(torchvision.utils.make_grid(x_concat, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    train()
    writer.close()


'''
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # down_conv for Encoder
        self.down_conv1 = self.conv_block(1, 64)
        self.down_conv2 = self.conv_block(64, 128)
        # up_conv for Decoder
        self.up_conv2 = self.conv_block(256, 64)
        self.up_conv1 = self.conv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1 = self.down_conv1(x)
        x = nn.functional.max_pool2d(conv1, 2)
        conv2 = self.down_conv2(x)
        x =  nn.functional.max_pool2d(conv2, 2)
        # Upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # skip connection
        x = torch.cat([x, conv2], dim=1)
        
        #ToDO
        
        x = self.up_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        return x
'''