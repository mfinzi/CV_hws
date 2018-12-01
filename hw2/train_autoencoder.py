import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from random import choice
import utils


def recon_loss(g_out, labels, args):
	"""
	Args:
		[g_out]     FloatTensor, (B, C, W, H), Output of the generator.
		[labels]    FloatTensor, (B, C, W, H), Ground truth images.
	Rets:
		Reconstruction loss with both L1 and L2.
	"""
	lam1 = args.recon_l1_weight
	lam2 = arg.recon_l2_weight

	fst_term = lam2*np.power((g_out - labels), 2)
	snd_term = lam1*np.absolute(g_out - labels)

	N = g_out.size()[0]
	res = (1./N)*np.sum(fst_term + snd_term)

	return res





class Encoder(nn.Module):

	def __init__(self, args):
		super(Encoder, self).__init__()
		
		self.layer1 = nn.Sequential (
			nn.Conv2d(in_channels=args.nc, out_channels=args.nef, kernel_size=args.e_ksize, stride=2, padding=1, bias=False),
			nn.ReLU()
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=args.nef, out_channels=args.nef*2, kernel_size=args.e_ksize, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(args.nef*2),
			nn.ReLU()
			)

		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=args.nef*2, out_channels=args.nef*4, kernel_size=args.e_ksize, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(args.nef*4),
			nn.ReLU()
			)

		self.layer4 = nn.Sequential(
			nn.Conv2d(in_channels=args.nef*4, out_channels=args.nef*8, kernel_size=args.e_ksize, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(args.nef*8),
			nn.ReLU()
			)

		self.layer5 = nn.Linear(args.nef*8, args.nz)

	def forward(self, x):
		return self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))


	def load_model(self, filename):
		""" Load the pretrained weights stored in file [filename] into the model.
		Args:
			[filename]  The filename of the checkpoint saved from the main procedure
						(i.e. the 'autoencoder.pth.tar' file below.)
		Usage:
			enet = Encoder(args)
			enet.load_model('autoencoder.pth.tar')
			# Here [enet] should be loaded with weights from file 'autoencoder.pth.tar'
		"""
		self.load_state_dict(torch.load(filename))



class Decoder(nn.Module):

	def __init__(self, args):
		super(Decoder, self).__init__()
		self.decoder_layers = nn.Sequential(
			nn.BatchNorm1d(args.nz),
			nn.Linear(args.nz, args.ngf*4*4),
			nn.ReLU(),

			nn.Conv2d(in_channels=args.ngf*4, out_channels=args.ngf*2, kernel_size=args.g_ksize, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(args.ngf*2),
			nn.ReLU(),

			nn.Conv2d(in_channels=args.ngf*2, out_channels=args.ngf, kernel_size=args.g_ksize, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(args.ngf),
			nn.ReLU(),

			nn.Conv2d(in_channels=args.ngf, out_channels=args.nc, kernel_size=args.g_ksize, stride=2, padding=1, bias=False),
			nn.Tanh()
			)



	def forward(self, z, c=None):
		return self.decoder_layers(z)


	def load_model(self, filename):
		""" Load the pretrained weights stored in file [filename] into the model.
		Args:
			[filename]  The filename of the checkpoint saved from the main procedure
						(i.e. the 'autoencoder.pth.tar' file below.)
		Usage:
			dnet = Decoder(args)
			dnet.load_model('autoencoder.pth.tar')
			# Here [dnet] should be loaded with weights from file 'autoencoder.pth.tar'
		"""
		self.load_state_dict(torch.load(filename))


def train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=None):
	"""Train the AutoEncoder for one iteration (i.e. forward, backward, and update
	   weights for one batch of data)
	Args:
		[input_data]    Input tensors tuple from the data loader.
		[encoder]       Encoder module.
		[decoder]       Decoder module.
		[enc_opt]       Optimizer to update encoder's weights.
		[dec_opt]       Optimizer to update decoder's weights.
		[args]          Commandline arguments.
		[writer]        Tensorboard writer (optional)
	Rets:
		[loss]  (float) Reconstruction loss of the batch (before the update).
	"""
	
	en_res = encoder(input_data)
	de_res = decoder(en_res)
	enc_opt.zero_grad()
	dec_opt.zero_grad()
	enc_opt.step()
	dec_opt.step()

	return recon_loss(de_res, input_data)

def sample(model, n, sampler, args):
	""" Sample [n] images from [model] using noise created by the sampler.
	Args:
		[model]     Generator model that takes noise and output images.
		[n]         Number of images to sample.
		[sampler]   [sampler()] will return a batch of noise.
	Rets:
		[imgs]      (B, C, W, H) Float, numpy array.
	"""
	images = np.zeros((n))

	for i in range(n):
		curr_sample = sampler()
		images[i] = model(curr_sample)

	return images


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
	args = utils.get_args()
	loader = utils.get_loader(args)['train']
	writer = SummaryWriter()

	decoder = Decoder(args)
	encoder = Encoder(args)
	if args.cuda:
		decoder = decoder.cuda()
		encoder = encoder.cuda()

	dec_opt = torch.optim.Adam(
			decoder.parameters(), lr=args.lr_dec, betas=(args.beta_1, args.beta_2))
	enc_opt = torch.optim.Adam(
			encoder.parameters(), lr=args.lr_enc, betas=(args.beta_1, args.beta_2))

	step = 0
	for epoch in range(args.nepoch):
		for input_data in loader:
			l = train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=writer)
			step += 1
			if step % 50 == 0:
				print("Step:%d\tLoss:%2.5f"%(step, l))

	utils.save_checkpoint('autoencoder.pth.tar', **{
		'decoder' : decoder.state_dict(),
		'encoder' : encoder.state_dict(),
		'dec_opt' : dec_opt.state_dict(),
		'enc_opt' : enc_opt.state_dict()
	})


	def get_z():
		z = torch.rand(args.batch_size, args.nz)
		if args.cuda:
			z = z.cuda(async=True)
		return z

	gen_img = sample(decoder, 60000, get_z, args)
	np.save('autoencoder_out.npy', gen_img)


