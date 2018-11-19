import torch
import torchvision
import numpy as np
from train_clf import resume_model
import tqdm
import scipy.linalg as la
import utils
from scipy.stats import entropy
from scipy.linalg import norm,sqrtm

def get_logits(imgs,model,bs=32):
    """ Extracts logits from a model, imgs should be numpy array (N, C, W, H),
        and model is the pytorch module on the gpu, returns a numpy array (N, K)
        where K is the number of classes"""
    N,C,W,H = imgs.shape
    mb_images = imgs.reshape(N//bs,bs,C,W,H)
    get_logits = lambda mb: model(torch.from_numpy(mb).cuda()).cpu().data.numpy()
    logits = np.concatenate([get_logits(minibatch) for minibatch in mb_images],axis=0)
    return logits

def IS(gen_imgs, model, args):
    """ Compute the Inception Score using [model] to replace the inception model
        NOTE: do not using batch/splits for this implementation.
    Aggs:
        [gen_imgs]  Generated images, numpy array (N, C, W, H)
        [model]     A model that could take a batch of image and produce a logits
                    NOTE: the logits is before softmax
        [args]       Commandline arguments.
    Rets:
        [score]     float, the inception score.
    """
    # E_z[KL(Pyz||Py)] = \mean_z [\sum_y (Pyz log(Pyz) - Pyz log(Py))]
    logits = get_logits(gen_imgs,model)
    Pyz = np.exp(logits).transpose() # Take softmax (up to a normalization constant)
    Py = Pyz.mean(-1)                # Average over z
    logIS = entropy(Pyz,Py).mean()   # Average over z
    raise np.exp(logIS)

def FID(gen_imgs, gtr_imgs, feat_extr, args, eps=1e-8):
    """ Compute FID scores for [gen_imgs] using [gtr_imgs] as ground truth and
        [feat_extr] as feature extractor.
    Args:
        [gen_imgs]   Generated images.
        [gtr_imgs]   Ground truth images.
        [feat_extr]  Feature extractor.
        [args]       Commandline arguments.
        [eps]        Small perturbation used for numerical stability.
    Rets:
        [score]     float, the FID score.
    """
    gen_logits = get_logits(gen_imgs,feat_extr)
    gtr_logits = get_logits(gtr_imgs,feat_extr)
    mu1 = np.mean(gen_logits,axis=0)
    sigma1 = np.cov(gen_logits, rowvar=False)
    mu2 = np.mean(gtr_logits,axis=0)
    sigma2 = np.cov(gtr_logits, rowvar=False)

    tr = np.trace(sigma1 + sigma2 - 2*sqrtm(sigma1@sigma2))
    distance = norm(mu1-mu2)**2 + tr
    return distance


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist only.')
    parser.add_argument('--dataroot', type=str, default='.', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--gen_img_file', type=str, default='autoencoder_out.npy',
            help='Numpy file that contains the generated images.')
    parser.add_argument('--score_type', type=str, default='IS', help='IS|FID.')
    parser.add_argument('--clf_ckpt', type=str, default='clf.pth.tar',
            help='Checkpoint filename for the classifier or feature extractor.')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()

    clf, _ = resume_model(args.clf_ckpt, args)
    gen_imgs = np.load(args.gen_img_file)
    if args.score_type == 'IS':
        score = IS(gen_imgs, clf, args)
    elif args.score_type == 'FID':
        loader = utils.get_loader(args)['train']
        gtr_imgs = []
        for x,_ in tqdm.tqdm(loader):
            gtr_imgs.append(x.detach().cpu().numpy())
        gtr_imgs = np.concatenate(gtr_imgs, axis=0)
        np.save('train_data.npy', gtr_imgs)
        score = FID(gen_imgs, gtr_imgs, clf, args, eps=1e-10)
    else:
        raise Exception("Invalid score type:%s"%(args.score_type))

    print(score)
