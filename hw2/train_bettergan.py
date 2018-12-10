"""
Implement a generative model that could beat both of the baseline in all metrics.
"""
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from random import choice
import utils
import numpy as np


#################################################################################################
#							Spectral-Normalized Convolution Layer 								#
#################################################################################################
class SpectralNormalizedConv2d(nn.Conv2d):
	"""
	Spectral-Normalized 2-D Convolution, using power iteration algorithm.

	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
				 padding=0, dilation=1, groups=1, bias=True, num_iter=1):
		super(SpectralNormalizedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride=1, padding=0, 
			dilation=1, groups=1, bias=True)

		self.num_iter = num_iter
		self.u = torch.rand(out_channels)

	def _l2(v):
		return v / (torch.norm(v, p=2) + 1e-10)

	def _power_iteration(W, u, num_iter):
		"""
		Power iteration to efficiently approximate the maximum eigenvalue of the Hessian matrix.  

		args:
			num_iter (int): number of iteration to perform
			u: current estimate of eigenvector 
			W: un-normalized weight (matrix of the subject) 

		"""
		assert num_iter > 0, '[num_iter] must be positive number.' 

		for _ in range(num_iter):
			v = _l2(torch.mv(torch.t(W), u))
			u = _l2(torch.mv(W, v))
		s = torch.dot(u, torch.mv(torch.t(W), u)) / torch.dot(u, u) # s = uWu/uu 

		return s, u 

	def spectral_normalize(self):
		"""
		See the section 4 second paragraph, footnote [3] of [Spectral Normalization for Generative 
		Adversarial Networks (Miyato et al.)]. "Note that, since we are conducting the convolution 
		discretely, the spectral norm will depend on the size of the stride and padding. However, 
		the answer will only differ by some predefined K." That is, treating the weight matrix as 
		2-D matris of dimension d_out x (d_in*h*w) is valid. 
        """
		weight_temp = self.weight.view(self.weight.size(0), -1)

		s, u = self._power_iteration(weight_temp, self.u, self.num_iter) 
		self.u = u
		self.weight = self.weight / s

	def forward(self, input):
		"""
		Weight is spectral-normalized at every forward execution while training. 
		"""
		if self.training:
			self.spectral_normalize()
		return F.conv2d(input, self.weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)



#################################################################################################
#					   				Deep Convolutional GAN		 								#
#################################################################################################

class SpectralNormalizedDiscriminator(nn.Module):

    def __init__(self, args):
        super(SpectralNormalizedDiscriminator, self).__init__()

        self.conv1 = SpectralNormalizedConv2d(args.nc, args.ndf, 4, 2, 1, bias=False)
        self.conv2 = SpectralNormalizedConv2d(args.ndf, 2 * args.ndf, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * args.ndf)
        self.conv3 = SpectralNormalizedConv2d(2 * args.ndf, 4 * args.ndf, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * args.ndf)
        self.conv4 = SpectralNormalizedConv2d(4 * args.ndf, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), .2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), .2)
        out = F.leaky_relu(self.bn3(self.conv3(out)), .2)
        out = F.sigmoid(self.conv4(out))

        return out


    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['dnet'])

class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngf = args.ngf

        self.proj = nn.Linear(args.nz, 4 * args.ngf * 4 * 4)
        self.bn0 = nn.BatchNorm1d(4 * args.ngf * 4 * 4)

        self.dconv1 = nn.ConvTranspose2d(4 * args.ngf, args.ngf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(args.ngf * 2)

        self.dconv2 = nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(args.ngf)

        self.dconv3 = nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False)

    def forward(self, z, c=None):
        out = F.relu(self.bn0(self.proj(z)))
        out = out.view(-1, 4 * args.ngf, 4, 4)
        out = F.relu(self.bn1(self.dconv1(out)))
        out = F.relu(self.bn2(self.dconv2(out)))
        out = F.tanh(self.dconv3(out))

        return out

    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['gnet'])

#################################################################################################
#								Losses and Training Functions 	 								#
#################################################################################################

def d_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Discriminator.
    """

    return - torch.mean(torch.log(dreal), dim=0) - torch.mean(torch.log(1 - dfake), dim=0)


def g_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Generator.
    """
    return - torch.mean(torch.log(dfake), dim=0)


def train_batch(input_data, g_net, d_net, g_opt, d_opt, sampler, args, writer=None):
    """Train the GAN for one batch iteration.
    Args:
        [input_data]    Input tensors (tuple). Should contain the images and the labels.
        [g_net]         The generator.
        [d_net]         The discriminator.
        [g_opt]         Optimizer that updates [g_net]'s parameters.
        [d_opt]         Optimizer that updates [d_net]'s parameters.
        [sampler]       Function that could output the noise vector for training.
        [args]          Commandline arguments.
        [writer]        Tensorboard writer.
    Rets:
        [L_d]   (float) Discriminator loss (before discriminator's update step).
        [L_g]   (float) Generator loss (before generator's update step)
    """
    if args.cuda:
        input_data[0] = input_data[0].cuda()
        input_data[1] = input_data[1].cuda()

    g_opt.zero_grad()
    d_opt.zero_grad()

    input_fake = sampler()
    dfake = d_net(g_net(input_fake))
    dreal = d_net(input_data[0])
    loss_g = g_loss(dfake, dreal)
    loss_g.backward()
    g_opt.step()

    input_fake = sampler()
    dfake = d_net(g_net(input_fake))
    dreal = d_net(input_data[0])
    loss_d = d_loss(dfake, dreal)
    loss_d.backward()
    d_opt.step()

    return loss_d, loss_g
def sample(model, n, sampler, args):
    """ Sample [n] images from [model] using noise created by the sampler.
    Args:
        [model]     Generator model that takes noise and output images.
        [n]         Number of images to sample.
        [sampler]   [sampler()] will return a batch of noise.
    Rets:
        [imgs]      (B, C, W, H) Float, numpy array.
    """
    images = np.zeros((n, args.nc, args.image_size, args.image_size))

    completed = 0

    while completed < n:

        curr_sample = sampler()

        results = model(curr_sample)

        temp_completed = completed + args.batch_size

        if temp_completed <= n:
            images[completed : temp_completed] = results.detach().cpu().numpy()
        else:
            diff = n - completed
            images[completed:] = results[:diff].detach().cpu().numpy()
        completed += args.batch_size

    return images


#################################################################################################
#												MAIN  											#
#################################################################################################
if __name__ == "__main__":
    args = utils.get_args()
    loader = utils.get_loader(args)['train']
    writer = SummaryWriter()

    d_net = SpectralNormalizedDiscriminator(args)
    g_net = Generator(args)

    if args.cuda:
        d_net = d_net.cuda()
        g_net = g_net.cuda()

    d_opt = torch.optim.Adam(
            d_net.parameters(), lr=args.lr_d, betas=(args.beta_1, args.beta_2))
    g_opt = torch.optim.Adam(
            g_net.parameters(), lr=args.lr_g, betas=(args.beta_1, args.beta_2))

    def get_z():
        z = torch.rand(args.batch_size, args.nz)
        if args.cuda:
            z = z.cuda(async=True)
        return z

    step = 0
    for epoch in range(args.nepoch):
        for input_data in loader:
            l_d, l_g = train_batch(
                input_data, g_net, d_net, g_opt, d_opt, get_z, args, writer=writer)

            step += 1
            print("Step:%d\tLossD:%2.5f\tLossG:%2.5f"%(step, l_d, l_g))

    utils.save_checkpoint('dcgan.pth.tar', **{
        'gnet' : g_net.state_dict(),
        'dnet' : d_net.state_dict(),
        'gopt' : g_opt.state_dict(),
        'dopt' : d_opt.state_dict()
    })

    gen_img = sample(g_net, 60000, get_z, args)
    np.save('dcgan_out.npy', gen_img)


