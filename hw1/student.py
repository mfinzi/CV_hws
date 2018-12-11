import numpy as np
import utils

def convolution(img, kernel, padding='fill'):
    """Convolve image [img] with [kernel].
    Args:
        [img]       Shape HxW, grayscale image.
        [kernel]    Shape hxw, grayscale image.
        [padding]   Please refer to utils.pad_image
    Rets:
        Shape HxW, image after convolving [kernel] over [img].

    """
    kernel = np.rot90(kernel, 2)
    h,w = kernel.shape[:2]
    t,b,l,r = (h-1)//2, h//2, (w-1)//2, w//2    # Use numpy padding because it works for >2d
    padshape = [(t,b),(l,r)]+[(0,0)]*(len(img.shape[2:]))
    padded_img = np.pad(img, padshape, mode={'fill':'constant','replicate':'edge'}[padding])
    conved_img = np.zeros_like(img)
    for i in 1+np.arange(-h//2,h//2):
        for j in 1+np.arange(-w//2,w//2):
            if kernel[t+i,l+j]==0: continue
            conved_img += kernel[t+i,l+j]*padded_img[t+i:-b+i or None,l+j:-r+j or None]
    return conved_img

def gaussian_kernel(k=3,sigma=.3):
    x,y = np.mgrid[-k//2:k//2,-k//2:k//2]+1
    kernel = np.exp(-(x**2 +y**2)/(2*sigma**2))
    return kernel/kernel.sum()

Dx = lambda: np.array([[1,0,-1]])
Dy = lambda: Dx().T

def convolve(*args,**kwargs):
    return convolution(*args,**kwargs)

def compute_Harris_response(img, patchsize, ksize=3, sigma=0.1, epsilon=1e-6):
    """ Computing the response for Harris corner detector. You need to complete
        the following steps in this function:
            1. Use Gaussian filter to smooth the image.
            2. Compute image gradients in both x-direction and y-direction.
            3. Compute M matrix for Harris Corner.
            4. Use det(M)/(trace(M) + epsilon) for the final response.
        NOTE: Though it's also a valid approach to use Sobel filter to do step-1
              and 2 together; in this assignment, please do these them separately.
        NOTE: An alternative way to compute the response is det(M)-k(trace(M))^2.
              In this assignment, we will use det(M)/(trace(M) + epsilon)
    Args:
        [img]           Shape HxW, grayscale image.
        [patchsize]     Size of the patch used for computing Harris response (i.e. M).
        [ksize]         Size of the Gaussian kernel used for smoothing.
        [sigma]         Variance of the Gaussian kernel.
    Rets:
        Shape HxW, matrix of response, where response is det/(trace+1e-6)
    """
    img = convolve(img,gaussian_kernel(ksize,sigma))
    g_x, g_y = convolve(img,Dx()), convolve(img,Dy())
    g = np.stack([g_x,g_y],axis=-1)

    gg_T = g[...,None,:]*g[...,:,None]
    avg_filter = np.ones((patchsize,patchsize))
    M = convolve(gg_T, avg_filter)

    det = M[...,0,0]*M[...,1,1] - M[...,0,1]*M[...,1,0]
    trace = M[...,0,0]+ M[...,1,1]
    return det/(trace+epsilon)


def compute_local_maxima(response, patchsize=3):
    """None-maximum suppression.
    Args:
        [response]      Shape HxW, containing the response from Harris Corner
                        detection. Assume responses >= 0
        [patchsize]     Size of the patch used to compute the local maxima.
    Rets:
        Shape HxW, value will be 0 if it's not the maximum value;
        if the value is local maxima, keep the original value
    """
    h,w = patchsize, patchsize
    t,b,l,r = (h-1)//2, h//2, (w-1)//2, w//2
    padded_response = utils.pad_image(response, t,b,l,r, padding='fill')
    # Convolution like Max pooling
    nbhds = np.zeros(response.shape+(h,w))
    for i in 1+np.arange(-h//2,h//2):
        for j in 1+np.arange(-w//2,w//2):
            nbhds[...,i,j] = padded_response[t+i:-b+i or None,l+j:-r+j or None]
    maxima = nbhds.max(axis=(-2,-1))

    return (response==maxima)*response


def compute_Harris_corners(img, patchsize, thresh, ksize=3, sigma=0.1, epsilon=1e-6):
    """Harris Corner Detection Function.
    Args:
        [img]           Shape:HxW   Grayscale image.
        [patchsize]     integer     Patch-size used for compute M matrix for Harris corner
                                    detector and the non-maximum supression.
        [thresh]        float       The localtion is a corner when the response > thresh
        [ksize]         int         Kernel size of the Gaussian filter used to smooth the image.
        [sigma]         float       Variance of the Gaussian filter used to smooth the image.
    Rets:
        [corners]       Shape Nx2   Localtions for all of the [N] detected corners.
        [R]             Shape HxW   Harris corner response
    """
    R = compute_Harris_response(img, patchsize, ksize=ksize, sigma=sigma, epsilon=epsilon)
    R = compute_local_maxima(R, patchsize=patchsize)
    corners = np.where(R>thresh)
    corners = np.concatenate((corners[0].reshape((-1,1)), corners[1].reshape((-1,1))),axis=1)
    return corners, R


def compute_mini_sift_desc(img, kp_locs, orientation_norm=False,
        patch_size=32, num_spatial_bins=4, num_ori_bins=8):
    """ Compute the mini-SIFT descriptor described in the homework write-up
        NOTE : Orientation normalization is computed in image patch.
        HINT : `utils.crop_patch` and `utils.compute_histogram` will be useful.
    Args:
        [img]                   Shape:HxW   Input image (in grayscale).
        [kp_locs]               Shape:Nx2   Localtion of the keypoints: (row, col)
        [orientation_norm]      Boolean     Whether do orientation normalization.
        [patch_size]            Int         Size of the image patch.
        [num_spatial_bins]      Int         #spatial bins.
        [num_ori_bins]          Int         #bins for the orientation histogram.
    Rets:
        Shape Nxd where d = [num_spatial_bins]x[num_spatial_bins]x[num_ori_bins].
        The default settings hould produce Nx128.
    """
    patches = key_points(img,kp_locs,patch_size)
    feature_vecs = []
    if orientation_norm == True:
        list_orientations, list_grad_norm = bin_norm(patches, num_ori_bins)
    else:
        list_orientations, list_grad_norm = grad_patches(patches)
    stride = int(patch_size/num_spatial_bins) #Could cause rounding problems
    for j, patch in enumerate(patches):
        grad_norms = list_grad_norm[j]
        orientations = list_orientations[j]
        vec = feature_vec(num_spatial_bins, grad_norms, orientations, stride, num_ori_bins)
        if np.linalg.norm(vec) > 0:
            vec = vec/np.linalg.norm(vec)
        feature_vecs.append(vec)
    return np.array(feature_vecs)

def feature_vec(num_spatial_bin, grad_norms, orientations, stride, num_ori_bins):
    feature_vec = []
    radians = 2*np.pi/num_ori_bins
    bins = np.zeros([num_ori_bins])
    for s in range(num_spatial_bin):
        for p in range(num_spatial_bin):
            gradient_patch = grad_norms[s:(s+1)*stride, p:(p+1)*stride]
            orientation_patch = orientations[s:(s+1)*stride, p:(p+1)*stride]
            for i in range(stride):
                for j in range(stride):
                    for k in range(num_ori_bins):
                        if orientations[i,j] < radians * (k-1) and orientations[i,j] < k*radians:
                            bins[k] += grad_norms[i,j]
            feature_vec.append(bins)

    return np.array(feature_vec).flatten()

def key_points(img, kp_locs, patch_size):
    odd = patch_size % 2
    patches = np.zeros([len(kp_locs),patch_size, patch_size])
    #Make patch short on edges
    for i, kp in enumerate(kp_locs):
        if odd:
            x_min = kp[1] - (patch_size-1)/2.0
            x_max = kp[1] + (patch_size+1)/2.0
            y_min = kp[0] - (patch_size-1)/2.0
            y_max = kp[0] + (patch_size+1)/2.0
        else: 
            x_min = kp[1] - np.round((patch_size)/2.0)
            x_max = kp[1] + np.trunc((patch_size)/2.0)
            y_min = kp[0] - np.round((patch_size)/2.0)
            y_max = kp[0] + np.trunc(patch_size/2.0)

        #patch = utils.crop_patch(img, y_min, x_min, y_max, x_max)
        patch = utils.crop_patch(img, x_min, y_min, x_max, y_max)
        patches[i,:,:] = patch
    return patches


def grad(img):
    x_filter = [[0,0,0],[-1,0,1],[0,0,0]]
    y_filter = [[0,-1,0],[0,0,0],[0,1,0]]
    #padded_img = utils.pad_image(img, 1,1,1,1)
    g_x, g_y = convolve(img,Dx(),'replicate'), convolve(img,Dy(),'replicate')
    orientations = np.arctan2(g_y, g_x)
    return g_x,g_y, orientations

#def norm_orientation():

def bin_norm(patches, num_ori_bins):
    radians = 2*np.pi/num_ori_bins
    bins = np.zeros([num_ori_bins])
    num_patches = patches.shape[2]
    list_orientations = []
    list_grad_norm = []
    for patch in patches:
        g_x, g_y, orientations = grad(patch)
        grad_norm = np.linalg.norm([g_x,g_y], axis = 0)
        height, width = patch.shape
        for i in range(height):
            for j in range(width):
                for k in range(num_ori_bins):
                    if orientations[i,j] < radians * (k) and orientations[i,j] < (k+1)*radians:
                        bins[k] += grad_norm[i,j]#np.norm([g_x[i,j],g_y[i,j]])
        max_orientation = np.argmax(bins)
        orientations = (orientations - radians*max_orientation)%(2*np.pi)
        print(radians*max_orientation)
        list_orientations.append(orientations)
        list_grad_norm.append(grad_norm)
    return list_orientations, list_grad_norm

def grad_patches(patches):
    list_orientations = []
    list_grad_norm = []
    for patch in patches:
        g_x, g_y, orientations = grad(patch)
        grad_norm = np.linalg.norm([g_x,g_y], axis = 0)
        list_orientations.append(orientations)
        list_grad_norm.append(grad_norm)
    return list_orientations, list_grad_norm

def find_correspondences(pts1, pts2, desc1, desc2, match_score_type='ratio'):
    """Given two list of key-point locations and descriptions, compute the correspondences.
    Args:
        [pts1]              (N,2)   Array of (row, col) from image 1, keypoints to be matched.
        [pts2]              (M,2)   Array of (row, col) from image 2, keypoints to be matched.
        [desc1]             (N,d)   Discriptor for keypoints at location in [pts1].
        [desc2]             (M,d)   Discriptor for keypoints at location in [pts2].
        [match_score_type]  str     How to compute the match score. Options include 'ssd'|'ratio'.
                                    'ssd'   - use sum of squared distance.
                                    'ratio' - use ratio test, the score will be the ratio.
    Rets:
        Return following three things: [corr], [min_idx], and [scores]
        [corr]              (N,4)   Array of (row_1, col_1, row_2, col_2), where (row_1, col_1) is
                                    a keypoint in [pts1] and (row_2, col_2) is a keypoint in [pts2].
                                    NOTE: you need to find the best match keypoints from [pts2]
                                          for all keypoints in [pts1].
        [min_idx]           (N,)    Index of the matched keypoints in [pts2]. [min_idx[i]] is the index
                                    of the keypoint that appears in [corr[i]].
        [scores]            (N,)    Match score of the correspondences. [scores[i]] is the score for
                                    correspondences [corr[i]]. This will be either SSD or ratio from
                                    the ratio test (i.e. minimum/second_minimum).
    """
    N = pts1.shape[0]
    X = np.sum(desc1**2, axis=1, keepdims=True)
    Y = np.sum(desc2**2, axis=1, keepdims=True).T
    XY = np.dot(desc1,desc2.T)
    L = X + Y - 2*XY

    D = (np.maximum(L, 0))
    scores = np.min(D, axis = 1)
    indices = np.argmin(D,axis = 1)
    corr = []
    for j,index in enumerate(indices):
        corr.append(np.hstack([pts1[j],pts2[index]]))
    if match_score_type=='ratio': 
        p = np.sort(D, axis = 1)
        scores = p[:,0]/p[:,1]
    return np.array(corr), indices, scores


def estimate_3D(point1, point2, P1, P2):
    """
    Args:
        [point1]    Shape:(3,)      3D array of homogenous coordinates from image 1.
        [point2]    Shape:(3,)      3D array of homogenous coordinates from image 2.
        [P1]        Shape:(3,4)     Projection matrix for image 1.
        [P2]        Shape:(3,4)     Projection matrix for image 2.
    Rets:
        Return 3D arrary, representing the coordinate of 3D point
        X such that [point1] ~ [P1]X and [point2] ~ [P2]X
    """
    A = np.zeros([4,4])
    #print(P1[2,:]*point1[0])
    point1
    A[0,:] = P1[0,:]-P1[2,:]*point1[1]
    A[1,:] = P1[1,:]-P1[2,:]*point1[0]

    A[2,:] = P2[0,:]-P2[2,:]*point2[1]
    A[3,:] = P2[1,:]-P2[2,:]*point2[0]
    _, vecs = np.linalg.eig(A.T @ A)
    index = np.argsort(_)[0]
    point = vecs[:,index]
    return point/float(point[-1])

def estimate_F(corrs):
    """ Eight Point Algorithm with Hartley Normalization.
    Args:
        [corrs]     Nx4     Correspondences between two images, organized
                            in the following way: (row1, col1, row2, col2).
                            Assume N >= 8, raise exception if N < 8.
    Rets:
        The estimated F-matrix, which is (3,3) numpy array.
    """
    N, _ = corrs.shape
    print(corrs[0,:])
    corrs_temp = np.zeros([N,4])
    corrs_temp[:,1] = corrs[:,0]
    corrs_temp[:,0] = corrs[:,1]
    corrs_temp[:,2] = corrs[:,3]
    corrs_temp[:,3] = corrs[:,2]

    corrs = corrs_temp

    for i in range(4):
        mean = np.mean(corrs[:,i])
        std = np.std(corrs[:,i])
        corrs[:,i] -= mean
        corrs[:,i] /= std
    Y = []
    for j in range(N):
        Y.append(np.outer(np.hstack([corrs[j,:2],1]),np.hstack([corrs[j,2:],1])).flatten())
    Y = np.array(Y)
    u, s, v = np.linalg.svd(Y, full_matrix = 0)
    indices = np.argsort(abs(s))
    if s[-1] != 0:
        F = v[indices[-1]]#check this because it's second largest
    else:
        F = v[indices[-2]]
    F = F.reshape([3,3])
    u, s, v = np.linalg.svd(F)
    s[-1] = 0
    print(s)
    F = u * np.diag(s) * v 
    #index = np
    return F


def sym_epipolar_dist(corr, F):
    """Compute the Symmetrical Epipolar Distance.
    Args:
        [corr]  (2,)    (row_1, col_1, row_2, col_2), where row_1, col_1 are points
                        from image 1, and row_2, col_2 are points from image 2.
        [F]     (3,3)   Fundamental matrix from image 1 to image 2.
    Rets:
        Return the symetrical epipolar distance (float)
    """
    raise NotImplementedError()


def ransac(data, hypothesis, metric, sample_size, num_iter, inlier_thresh):
    """ Implement the general RANSAC framework.
    Args:
        [data]          (N,d) numpy array, representing the data to fit.
        [hypothesis]    Function that takes a (m,d) numpy array, return a model
                        (represented as a numpy array). For the case of F-matrix
                        estimation, hypothesis takes Nx4 data (i.e. the
                        correspondences) and return the 3x3 F-matrix.
        [metric]        Function that take an entry from [data] and an output
                        of [hypothesis]; it returns a score (float) mesuring how
                        well the data entry fits the output hypothesis.
                        ex. metric(data[i], hypothesis(data)) -> score (float).
        [sample_size]   Number of entries to sample for each iteration.
        [num_iter]      Number of iterations we run RANSAC.
        [inlier_thres]  The threshold to decide whether a data point is inliner.
    Rets:
        Returning the best fit model [model] and the inliner mask [mask].
        [model]         The best fit model (i.e. having fewest outliner ratio).
        [mask]          Mask for inliners. [mask[i]] is 1 if data[i] is an inliner
                        for the output model [model], 0 otherwise.
    """
    metric = np.vectorize(metric)
    N,d = data.shape
    best_score, best_hypothesis = 0, None
    for i in range(num_iter):
        js = np.random.choice(N,size=sample_size,replace=False)
        hypothesis_elements = data[js,:]
        H = hypothesis(hypothesis_elements)
        scores = metric(data,H)
        inlier_frac = (scores<inlier_thresh).mean()
        if inlier_frac>best_score:
            best_score, best_hypothesis = inlier_frac, H
    return H


def estimate_F_ransac(corr, num_iter, inlier_thresh):
    """Use normalized 8-point algorithm, symetrical epipolar distance, and
       RANSAC to estimate F-matrix.
       NOTE: Please reuse the `ransac`, `sym_epipolar_dist`, and `estimate_F`
             functions implemented above.
    Args:
        [corrs]         Nx4     Correspondences between two images, organized
                                in the following way: (row1, col1, row2, col2).
        [num_iter]      Number of iterations we run RANSAC.
        [inlier_thres]  The threshold to determine whether the data point is inliner.
    Rets:
        The estimated F-matrix, which is (3,3) numpy array.
    """
    raise NotImplementedError()


