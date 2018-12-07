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

	intermed = g_out - labels

	lam1 = args.recon_l1_weight
	fst_term = lam1*intermed.abs()


	lam2 = args.recon_l2_weight
	snd_term = lam2*(intermed.pow(2))

	interm_two = fst_term + snd_term

	N = g_out.size()[0]
	res = (1./N)*(interm_two.sum())
	return res





class Encoder(nn.Module):

	def __init__(self, args):
		super(Encoder, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=args.nc, out_channels=args.nef, kernel_size=args.e_ksize, stride=2, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=args.nef, out_channels=args.nef*2, kernel_size=args.e_ksize, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(args.nef*2)
		self.conv3 = nn.Conv2d(in_channels=args.nef*2, out_channels=args.nef*4, kernel_size=args.e_ksize, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(args.nef*4)
		self.conv4 = nn.Conv2d(in_channels=args.nef*4, out_channels=args.nef*8, kernel_size=args.e_ksize, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(args.nef*8)
		self.proj = nn.Linear(args.nef*8*2*2, args.nz)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = x.view(x.size(0), -1)
		x = self.proj(x)	
		return x


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
		checkpoint = torch.load(filename)
		print (checkpoint.keys())
		self.load_state_dict(checkpoint['encoder'])



class Decoder(nn.Module):

	def __init__(self, args):
		super(Decoder, self).__init__()

		self.proj = nn.Linear(args.nz, args.ngf*4*4*4)
		self.bn0 = nn.BatchNorm1d(args.ngf*4*4*4)
		self.dconv1 = nn.ConvTranspose2d(in_channels=args.ngf*4, out_channels=args.ngf*2, kernel_size=args.g_ksize, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(args.ngf*2)
		self.dconv2 = nn.ConvTranspose2d(in_channels=args.ngf*2, out_channels=args.ngf, kernel_size=args.g_ksize, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(args.ngf)
		self.dconv3 = nn.ConvTranspose2d(in_channels=args.ngf, out_channels=args.nc, kernel_size=args.g_ksize, stride=2, padding=1, bias=False)

		self.reshape_num = args.ngf

	def forward(self, z, c=None):
		x = F.relu(self.bn0(self.proj(z)))
		x = x.view(-1,self.reshape_num*4,4,4)
		x = F.relu(self.bn1(self.dconv1(x)))
		x = F.relu(self.bn2(self.dconv2(x)))
		x = F.tanh(self.dconv3(x))
		return x


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
		checkpoint = torch.load(filename)
		print (checkpoint.keys())

		self.load_state_dict(checkpoint['decoder'])



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

	if args.cuda:
		input_data[0] = input_data[0].cuda()
		input_data[1] = input_data[1].cuda()


	enc_opt.zero_grad()
	dec_opt.zero_grad()

	en_res = encoder(input_data[0])
	de_res = decoder(en_res)

	loss  = recon_loss(de_res, input_data[0], args)
	loss.backward()

	enc_opt.step()
	dec_opt.step()

	

	return loss

def sample(model, n, sampler, args):
	""" Sample [n] images from [model] using noise created by the sampler.
	Args:
		[model]     Generator model that takes noise and output images.
		[n]         Number of images to sample.
		[sampler]   [sampler()] will return a batch of noise.
	Rets:
		[imgs]      (n, C, W, H) Float, numpy array.
	"""
	images = np.zeros((n, C, W, H))

	completed = 0

	while completed < n:

		curr_sample = sampler()

		results = model(curr_sample)

		temp_completed = completed + curr_sample

		if completed <= n: 
			images[completed : temp_completed - 1] = results



	# for i in range(n):
	# 	curr_sample = sampler()
	# 	print (curr_sample)
	# 	print (curr_sample.size())
	# 	images[i] = model(curr_sample)

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
	for epoch in range(0):
		print (epoch)
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


