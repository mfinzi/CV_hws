from PIL import Image
import unittest
import numpy as np
import student

class ConvolutionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = np.load('tests_release/convolution.npz')
        self.img = tmp['img']
        self.filt = tmp['filt']
        self.res_noreplicate = tmp['res_noreplicate']
        self.res_replicate = tmp['res_replicate']

    def test_without_replication(self):
        output = student.convolution(self.img, self.filt, padding='fill')
        self.assertTrue(np.allclose(output, self.res_noreplicate))

    def test_with_replication(self):
        output = student.convolution(self.img, self.filt, padding='replicate')
        self.assertTrue(np.allclose(output, self.res_replicate))


class HarrisCornerTestCase(unittest.TestCase):
    def setUp(self):
        tmp = np.load('tests_release/corner.npz')
        self.img = tmp['img']
        self.num = len(tmp['gts_response'])
        self.gts_response = tmp['gts_response']
        self.gts_nonmaxed_response = tmp['gts_nonmaxed_response']
        self.threshs = tmp['threshs']
        self.patch_sizes = tmp['patch_sizes']

    def test_response_value(self):
        for i in range(self.num):
            response = student.compute_Harris_response(
                    self.img, self.patch_sizes[i])
            self.assertTrue(np.allclose(response, self.gts_response[i]))

    def test_local_maxima(self):
        for i in range(self.num):
            nonmaxed_response = student.compute_local_maxima(self.gts_response[i])
            self.assertTrue(np.allclose(nonmaxed_response, self.gts_nonmaxed_response[i]))


class SiftDescriptorTestCase(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((100,100))
        self.img2 = np.zeros((100,100))
        self.img[50,40:60] = 1
        self.img2[55,40:60] = 1

        self.img3 = np.zeros((100,100))
        self.img4 = np.zeros((100,100))
        self.img3[49:51,50] = 1
        self.img4[50,49:51] = 1
        self.corners1 = np.array([[50,40], [50,50], [50,60]])
        self.corners2 = np.array([[65,24]])
        self.corners3 = np.array([[50,50]])


    def test_empty_patch(self):
        d1 = student.compute_mini_sift_desc(
                self.img, np.array([[3,3]]), orientation_norm=False)
        self.assertTrue(np.allclose(d1,np.zeros_like(d1)))

        d1 = student.compute_mini_sift_desc(
                self.img, np.array([[3,3]]), orientation_norm=True)
        self.assertTrue(np.allclose(d1,np.zeros_like(d1)))


    def test_nonnormalized_intensity_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img, self.corners1[:1], orientation_norm=False)
        d2 = student.compute_mini_sift_desc(
                self.img + 5, self.corners1[:1], orientation_norm=False)

        self.assertTrue(np.allclose(d1,d2))


    def test_normalized_intensity_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img, self.corners1, orientation_norm=True)
        d2 = student.compute_mini_sift_desc(
                self.img + 5, self.corners1, orientation_norm=True)
        self.assertTrue(np.allclose(d1,d2))


    def test_unnormalized_translation_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img, self.corners2, orientation_norm=False,
                patch_size=32, num_spatial_bins=1)
        d2 = student.compute_mini_sift_desc(
                self.img2, self.corners2, orientation_norm=False,
                patch_size=32, num_spatial_bins=1)
        self.assertTrue(np.allclose(d1,d2))


    def test_normalized_translation_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img, self.corners2, orientation_norm=True,
                patch_size=32, num_spatial_bins=1)
        d2 = student.compute_mini_sift_desc(
                self.img2, self.corners2, orientation_norm=True,
                patch_size=32, num_spatial_bins=1)
        self.assertTrue(np.allclose(d1,d2))


    def test_normalized_rotational_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img3, self.corners3, orientation_norm=True,
                patch_size=32, num_spatial_bins=1)
        d2 = student.compute_mini_sift_desc(
                self.img4, self.corners3, orientation_norm=True,
                patch_size=32, num_spatial_bins=1)
        self.assertTrue(np.allclose(d1,d2))


    def test_nonormalized_rotational_inv(self):
        d1 = student.compute_mini_sift_desc(
                self.img3, self.corners3, orientation_norm=False,
                patch_size=32, num_spatial_bins=1)
        d2 = student.compute_mini_sift_desc(
                self.img4, self.corners3, orientation_norm=False,
                patch_size=32, num_spatial_bins=1)
        self.assertTrue(np.allclose(d1,d2))


class CorrespondenceTestCase(unittest.TestCase):
    def setUp(self):
        tmp = np.load('tests_release/corr.npz')
        self.corners1 = tmp['corners1']
        self.corners2 = tmp['corners2']
        self.sift1 = tmp['sift1']
        self.sift2 = tmp['sift2']
        self.gt_match_idx_ssd = tmp['gt_match_idx_ssd']
        self.gt_match_scores_ssd = tmp['gt_match_scores_ssd']
        self.gt_corr_ssd = tmp['gt_corr_ssd']
        self.gt_match_idx_rt = tmp['gt_match_idx_rt']
        self.gt_match_scores_rt = tmp['gt_match_scores_rt']
        self.gt_corr_rt = tmp['gt_corr_rt']

    def test_ssd(self):
        correspondences, match_idx, match_scores = \
                student.find_correspondences(
                        self.corners1, self.corners2, \
                        self.sift1, self.sift2, match_score_type='ssd')
        self.assertTrue(np.allclose(match_idx, self.gt_match_idx_ssd))
        self.assertTrue(np.allclose(match_scores, self.gt_match_scores_ssd))
        self.assertTrue(np.allclose(correspondences, self.gt_corr_ssd))

    def test_ratio(self):
        correspondences, match_idx, match_scores = \
                student.find_correspondences(
                        self.corners1, self.corners2,self.sift1, self.sift2)
        self.assertTrue(np.allclose(match_idx, self.gt_match_idx_rt))
        self.assertTrue(np.allclose(match_scores, self.gt_match_scores_rt))
        self.assertTrue(np.allclose(correspondences, self.gt_corr_rt))


class FundamentalMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.num_iter = 5000
        self.inlier_thresh = 2
        self.atol = 1e-3

        tmp = np.load('tests_release/F_only.npz')
        self.corr = tmp['corr']
        self.gt_F = tmp['gt_F']
        self.gt_F_8pts = tmp['gt_F_8pts']
        self.sym_epi_dists = tmp['sym_epi_dists']
        self.ransac_8pts_gt_F = tmp['ransac_8pts_gt_F']
        self.ransac_9pts_gt_F = tmp['ransac_9pts_gt_F']
        self.corr_outliers = tmp['corr_outliers']


    def test_F_rank(self):
        F = student.estimate_F(self.corr)
        self.assertTrue(np.linalg.matrix_rank(F), 2)

    def test_F_norm(self):
        F = student.estimate_F(self.corr)
        self.assertTrue(np.linalg.norm(F), 1)

    def test_F_transpose(self):
        F = student.estimate_F(self.corr)
        corr_T = np.zeros_like(self.corr)
        corr_T[:,:2] = self.corr[:,2:]
        corr_T[:,2:] = self.corr[:,:2]
        F_T = student.estimate_F(corr_T)
        self.assertTrue(np.allclose(F.T, F_T))

    def test_F_deterministic(self):
        F = student.estimate_F(self.corr)
        self.assertTrue(
                (np.allclose(F,  self.gt_F, atol=self.atol)) \
             or (np.allclose(-F, self.gt_F, atol=self.atol)))


    def test_F_deterministic_8points(self):
        F = student.estimate_F(self.corr[:8,:])
        self.assertTrue(
                (np.allclose(F,  self.gt_F_8pts, atol=self.atol)) \
             or (np.allclose(-F, self.gt_F_8pts, atol=self.atol)))


    def test_sym_epi_dist(self):
        for i in range(self.corr.shape[0]):
            d_student  = student.sym_epipolar_dist(self.corr[i,:], self.gt_F)
            self.assertTrue(abs(d_student - self.sym_epi_dists[i]) < self.atol)


    def test_F_ransac8points(self):
        F = student.estimate_F_ransac(
                self.corr[:8, :], 1, self.inlier_thresh)
        self.assertTrue(
                (np.allclose(F,  self.ransac_8pts_gt_F, atol=self.atol)) \
             or (np.allclose(-F, self.ransac_8pts_gt_F, atol=self.atol)))


    def test_F_ransac9points(self):
        F = student.estimate_F_ransac(
                self.corr_outliers, 100, self.inlier_thresh)
        self.assertTrue(
                (np.allclose(F,  self.ransac_9pts_gt_F, atol=self.atol)) \
             or (np.allclose(-F, self.ransac_9pts_gt_F, atol=self.atol)))



class RANSACTestCase(unittest.TestCase):

    def setUp(self):
        self.inlier_thresh = 2
        self.atol = 1e-3
        self.num_iters = 1000

        tmp = np.load('tests_release/ransac.npz')
        self.corr_input = tmp['corr']
        self.ans_inliers = tmp['ans_inliers']


    def test_F_ransacAllpoints(self):
        _, inliers = student.ransac(
            self.corr_input.copy(), student.estimate_F, student.sym_epipolar_dist,
            8, self.num_iters, self.inlier_thresh)
        self.assertTrue(np.allclose(inliers, self.ans_inliers))



class TriangulationTestCase(unittest.TestCase):
    def setUp(self):
        tmp = np.load('tests_release/3Dgt.npz')
        self.X = tmp['X']
        self.x1 = tmp['x1']
        self.x2 = tmp['x2']
        self.P1 = tmp['P1']
        self.P2 = tmp['P2']

    def test3D(self):
        X = np.zeros_like(self.X)
        for i in range(self.X.shape[1]):
            X[:,i] = student.estimate_3D(
                    self.x1[:,i],self.x2[:,i],self.P1,self.P2)
        X = X / X[3,:]
        self.assertTrue(np.allclose(X, self.X))



if __name__ == '__main__':
    unittest.main()
