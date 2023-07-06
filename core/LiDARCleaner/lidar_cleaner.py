import copy
import torch
import occdtc_ops
import numpy as np
import torch.nn as nn

import PIL.Image as Image

from kornia.geometry.epipolar.numeric import cross_product_matrix
from kornia.geometry.epipolar import compute_correspond_epilines
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator

class LiDARCleaner(nn.Module):
    """
    Function to Clean LiDAR Scan Data.
    """
    def __init__(self, intrinsic_cam, extrinsic_LiDAR2Cam, LiDARPoints3D, height, width, rszh=0.5, rszw=1.0):
        super().__init__()
        self.intrinsic_cam = self.check_intrinsic(copy.deepcopy(intrinsic_cam))
        self.extrinsic_LiDAR2Cam = self.check_extrinsic(copy.deepcopy(extrinsic_LiDAR2Cam))
        self.LiDARPoints3D = self.check_LiDARPoints3D(copy.deepcopy(LiDARPoints3D))

        # Resize the Image to desired scale, we do not test when image enlarged
        assert rszh <= 1.0 and rszw <= 1.0
        self.height_rz, self.width_rz = int(height * rszh), int(width * rszw)
        self.height, self.width = int(height), int(width)
        self.rszh, self.rszw = float(self.height_rz / self.height), float(self.width_rz / self.width)
        self.resizeM = self.acquire_resizeM(self.rszh, self.rszw)

        self.debug = False

    def acquire_resizeM(self, rszh, rszw):
        resizeM = torch.eye(3)
        resizeM[0, 0] = rszw
        resizeM[1, 1] = rszh
        return resizeM

    def check_intrinsic(self, intrinsic_cam):
        if isinstance(intrinsic_cam, np.ndarray):
            intrinsic_cam = torch.from_numpy(intrinsic_cam)
        assert intrinsic_cam.shape[-1] == 3 and intrinsic_cam.shape[-2] == 3 and intrinsic_cam.device == torch.device("cpu")
        return intrinsic_cam.float()

    def check_extrinsic(self, extrinsic_LiDAR2Cam):
        if isinstance(extrinsic_LiDAR2Cam, np.ndarray):
            extrinsic_LiDAR2Cam = torch.from_numpy(extrinsic_LiDAR2Cam)
        assert extrinsic_LiDAR2Cam.shape[-2] == 3 and extrinsic_LiDAR2Cam.shape[-1] == 4 and extrinsic_LiDAR2Cam.device == torch.device("cpu")
        return extrinsic_LiDAR2Cam.float()

    def check_LiDARPoints3D(self, LiDARPoints3D):
        if isinstance(LiDARPoints3D, np.ndarray):
            LiDARPoints3D = torch.from_numpy(LiDARPoints3D)
        assert LiDARPoints3D.shape[-2] == 3 and LiDARPoints3D.device == torch.device("cpu")
        npts = LiDARPoints3D.shape[-1]
        LiDARPoints3D = torch.cat([LiDARPoints3D, torch.ones([1, npts])], dim=0).contiguous()
        return LiDARPoints3D.float()

    def interpolated_depth(self, depth, querylocation):
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        if isinstance(querylocation, np.ndarray):
            querylocation = torch.from_numpy(querylocation)

        h, w = depth.shape[-2::]
        depth = depth.view([1, 1, h, w])

        nquery, nsample, _ = querylocation.shape
        qx, qy = torch.split(querylocation, 1, dim=2)
        qx, qy = ((qx / w) - 0.5) * 2, ((qy / h) - 0.5) * 2
        querylocation = torch.cat([qx, qy], dim=2)
        querylocation = querylocation.view([1, nquery, nsample, 2])

        querydepth = torch.nn.functional.grid_sample(depth.cuda().float(), querylocation.cuda().float(), mode='nearest')
        oodselector = torch.nn.functional.grid_sample(torch.ones_like(depth).cuda().float(), querylocation.cuda().float(), mode='nearest')
        querydepth[oodselector == 0] = 1e5

        querydepth = querydepth.view([nquery, nsample, 1])
        return querydepth

    def inpainting_depth(self, rgb=None):
        pure_rotation = copy.deepcopy(self.extrinsic_LiDAR2Cam)
        pure_rotation[0:3, 3:4] = 0.0

        # Operate on Resized Depthmap
        grid_x, grid_y = np.meshgrid(range(self.width_rz), range(self.height_rz))
        prjpc, depths, visible_points = self.prj(
            self.resizeM @ self.intrinsic_cam,
            pure_rotation,
            self.LiDARPoints3D,
            height=self.height_rz, width=self.width_rz
        )

        prjpc_val = prjpc[:, visible_points].cpu().numpy()
        depths_val = depths[visible_points].cpu().numpy()

        assert prjpc_val.shape[1] > 100

        # tri = Delaunay(prjpc_val.T)
        # tri_function = LinearNDInterpolator(tri, depths_val, fill_value=1e5)
        # inpait_d = tri_function(grid_x, grid_y)

        nearest_func = NearestNDInterpolator(prjpc_val.T, depths_val)
        inpait_d = nearest_func(grid_x, grid_y)

        # inpait_d_ = self.interpolated_depth(inpait_d, prjpc_val)
        # diff = depths_val - inpait_d_.cpu().numpy()
        # diff = np.abs(diff)
        # acc01 = (diff < 0.1).mean()
        # acc05 = (diff < 0.5).mean()

        # Change to KNN in 3D Space?
        if rgb is not None:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(rgb.resize((self.width_rz, self.height_rz)))
            plt.scatter(prjpc_val[0, :], prjpc_val[1, :], c=1 / depths_val, cmap=plt.cm.get_cmap('magma'))
            plt.show()

            from core.utils.visualization import tensor2disp
            tensor2disp(1 / torch.from_numpy(inpait_d).unsqueeze(0).unsqueeze(0), vmax=0.3, viewind=0).show()

        return inpait_d, prjpc, depths, visible_points

    def prj(self, intrinsic, extrinsic, pc3D, height, width, min_dist=0.1):
        prjpc = intrinsic @ extrinsic @ pc3D
        prjpc[0, :] = prjpc[0, :] / (prjpc[2, :] + 1e-8)
        prjpc[1, :] = prjpc[1, :] / (prjpc[2, :] + 1e-8)
        depth = prjpc[2, :]

        visible_sel = (depth > min_dist)
        visible_sel = torch.logical_and(visible_sel, prjpc[0, :] > 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[0, :] < width - 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[1, :] > 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[1, :] < height - 0.5)
        return prjpc[0:2, :], depth, visible_sel

    def pad_pose44(self, extrinsic):
        pose = torch.eye(4)
        h, w = extrinsic.shape
        pose[0:h, 0:w] = extrinsic
        return pose

    def epplinedir(self, prjpc_vlidar, selector=None):
        intrinsic_cam_scaled = self.resizeM @ self.intrinsic_cam
        pure_rotation = self.pad_pose44(copy.deepcopy(self.extrinsic_LiDAR2Cam))
        pure_rotation[0:3, 3:4] = 0
        pure_translation = self.pad_pose44(copy.deepcopy(self.extrinsic_LiDAR2Cam)) @ torch.linalg.inv(pure_rotation)
        epipole = -intrinsic_cam_scaled @ pure_translation[0:3, 3:4]
        eppx, eppy = (epipole[0, 0] / epipole[2, 0]).item(), (epipole[1, 0] / epipole[2, 0]).item()

        eppdir = torch.stack([eppx - prjpc_vlidar[0, :], eppy - prjpc_vlidar[1, :]], dim=0).contiguous()
        eppdir = torch.nn.functional.normalize(eppdir, dim=0)
        eppdir = eppdir.T

        # eppdir = eppdir[:, selector]
        # E_mat = cross_product_matrix(pure_translation[0:3, 3])
        # F_mat = intrinsic_cam_scaled.inverse().transpose(-2, -1) @ E_mat @ intrinsic_cam_scaled.inverse()
        # eppline = compute_correspond_epilines(prjpc_vlidar.T, F_mat)
        # eppdir_val = eppline / torch.sqrt(torch.sum(eppline[:, 0:2] ** 2, dim=1, keepdim=True))
        # eppdir_val = eppdir_val[selector, 0:2]
        return eppdir, pure_translation[0:3, :], (eppx, eppy)

    def backprj_prj(self, intrinsic, pure_translation, enumlocation, depthinterp):
        intrinsic = self.pad_pose44(intrinsic).cuda()
        pure_translation = self.pad_pose44(pure_translation).cuda()
        prjM = intrinsic @ pure_translation @ intrinsic.inverse()
        prjM = prjM.view([1, 1, 4, 4])

        nquery, nsample, _ = enumlocation.shape
        qx, qy = torch.split(enumlocation, 1, dim=2)
        pts3D = torch.cat([qx * depthinterp, qy * depthinterp, depthinterp, torch.ones_like(depthinterp)], dim=2)
        pts3D = pts3D.view([nquery, nsample, 4, 1])

        pts3Dprj = prjM @ pts3D
        pts3Dprjx = pts3Dprj[:, :, 0, 0] / pts3Dprj[:, :, 2, 0]
        pts3Dprjy = pts3Dprj[:, :, 1, 0] / pts3Dprj[:, :, 2, 0]
        return torch.stack([pts3Dprjx, pts3Dprjy], dim=-1)

    def clean_python(self, intrinsic, pure_translation, depthmap, prjpc_lidar, prjpc_cam, eppdir, selector, srch_resolution=0.5):
        prjpc_lidar_, prjpc_cam_ = prjpc_lidar[selector, :], prjpc_cam[selector, :]
        eppdir_ = eppdir[selector, :]

        mindist = 1
        maxdist = 100
        samplenum = int(np.ceil((maxdist - mindist) / srch_resolution).item() + 1)
        sampled_range = torch.linspace(mindist, maxdist, samplenum).cuda()

        nanchor = len(eppdir_)
        enumlocation = prjpc_lidar_.view([nanchor, 1, 2]) + sampled_range.view([1, samplenum, 1]) * eppdir_.view([nanchor, 1, 2])

        depthinterp = self.interpolated_depth(depthmap, enumlocation)
        pts3Dprj = self.backprj_prj(intrinsic, pure_translation, enumlocation, depthinterp)

        nquery = len(prjpc_lidar_)
        prj_dir = pts3Dprj - prjpc_cam_.view([nquery, 1, 2])
        prj_dir = torch.nn.functional.normalize(prj_dir, dim=2)
        cosdiff = torch.sum(prj_dir * eppdir_.view([nquery, 1, 2]), dim=2)
        cosdiffmax, _ = torch.min(cosdiff, dim=1)
        occluded = cosdiffmax < 0

        if self.debug:
            h, w = depthmap.shape

            from core.utils.visualization import tensor2disp
            depthmapvls = tensor2disp(1 / depthmap.unsqueeze(0).unsqueeze(0), vmax=0.3, viewind=0)

            sampleidx = 1793

            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(depthmapvls)
            plt.xlim([0, w])
            plt.ylim([h, 0])
            plt.scatter(prjpc_lidar_[sampleidx, 0].cpu().numpy(), prjpc_lidar_[sampleidx, 1].cpu().numpy())
            plt.scatter(prjpc_cam_[sampleidx, 0].cpu().numpy(), prjpc_cam_[sampleidx, 1].cpu().numpy())
            plt.scatter(enumlocation[sampleidx, :, 0].cpu().numpy(), enumlocation[sampleidx, :, 1].cpu().numpy(), 1)
            plt.scatter(pts3Dprj[sampleidx, :, 0].cpu().numpy(), pts3Dprj[sampleidx, :, 1].cpu().numpy(), 1)
            plt.show()

        return occluded

    def forward(self, rgb=None):
        # Inpainting Depthmap in Virtual LiDAR View (Pure Rotation Movement) on Resized Depthmap
        inpait_d, prjpc_vlidar, depths_vlidar, visible_sel_vlidar = self.inpainting_depth(rgb=None)

        # Compute Essential Matrix Between Interpolated Depthmap and Camera View, as a pure translation
        eppdir, pure_translation, epppole = self.epplinedir(prjpc_vlidar, visible_sel_vlidar)

        # Clean up
        prjpc_cam, _, _ = self.prj(
            self.resizeM @ self.intrinsic_cam,
            self.extrinsic_LiDAR2Cam,
            self.LiDARPoints3D,
            height=self.height_rz, width=self.width_rz
        )
        inpait_d = torch.from_numpy(inpait_d).cuda().float()
        prjpc_vlidar, prjpc_cam = prjpc_vlidar.T.cuda().float(), prjpc_cam.T.cuda().float()
        eppdir = eppdir.cuda().float()
        occluded = self.clean_python(self.resizeM @ self.intrinsic_cam, pure_translation, inpait_d, prjpc_vlidar, prjpc_cam, eppdir, visible_sel_vlidar, srch_resolution=1.0)

        # Apply Clean
        tomask = torch.zeros_like(occluded.cpu())
        tomask[occluded] = 1
        tomask_all = torch.zeros_like(visible_sel_vlidar)
        tomask_all[visible_sel_vlidar] = tomask

        # Acquire Visible LiDAR Points in Camera View
        camprj_vls, camdepth_vls, visible_points = self.prj(
            self.intrinsic_cam,
            self.extrinsic_LiDAR2Cam,
            self.LiDARPoints3D,
            height=self.height, width=self.width
        )
        visible_points_filtered = torch.clone(visible_points)
        visible_points_filtered[tomask_all] = 0

        x, y = 909, 826
        debugindex = (camprj_vls[0, visible_sel_vlidar] - x).abs() + (camprj_vls[1, visible_sel_vlidar] - y).abs()
        min_idx = torch.argmin(debugindex).item()

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(rgb)
        plt.scatter(camprj_vls[0, visible_points].numpy(), camprj_vls[1, visible_points].numpy(), c=1 / camdepth_vls[visible_points], cmap=plt.cm.get_cmap('magma'))
        plt.savefig('tmp1.jpg', transparent=True)
        # plt.show()

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(rgb)
        plt.scatter(camprj_vls[0, visible_points_filtered].numpy(), camprj_vls[1, visible_points_filtered].numpy(), c=1 / camdepth_vls[visible_points_filtered], cmap=plt.cm.get_cmap('magma'))
        plt.savefig('tmp2.jpg', transparent=True)
        # plt.show()

        im1 = Image.open('tmp1.jpg')
        im2 = Image.open('tmp2.jpg')
        imcombined = np.concatenate([np.array(im1), np.array(im2)], axis=0)
        imcombined = Image.fromarray(imcombined)

        if self.debug:
            prjpc_ck, depth_ck, sel_ck = self.prj(
                self.resizeM @ self.intrinsic_cam,
                self.extrinsic_LiDAR2Cam,
                self.LiDARPoints3D,
                height=self.height_rz, width=self.width_rz
            )

            eppdir_ck = -(prjpc_ck - prjpc_vlidar)
            eppdir_ck = eppdir_ck / torch.sqrt(torch.sum(eppdir_ck ** 2, dim=0, keepdim=True))
            eppdir_ck = eppdir_ck.T
            eppdir_ck = eppdir_ck[visible_sel_vlidar, :]
            eppdir = eppdir[visible_sel_vlidar, :]

            import matplotlib.pyplot as plt
            plt.figure()
            plt.xlim([0, self.width_rz])
            plt.ylim([self.height_rz, 0])
            plt.axis('equal')
            plt.scatter(prjpc_ck[0, sel_ck], prjpc_ck[1, sel_ck], c=1 / depth_ck[sel_ck], cmap=plt.cm.get_cmap('magma'))
            plt.show()

            import matplotlib.pyplot as plt
            plt.figure()
            plt.xlim([0, self.width_rz])
            plt.ylim([self.height_rz, 0])
            plt.axis('equal')
            plt.scatter(prjpc_vlidar[0, sel_ck], prjpc_vlidar[1, sel_ck], c=1 / depths_vlidar[sel_ck], cmap=plt.cm.get_cmap('magma'))
            plt.quiver(prjpc_vlidar[0, sel_ck], prjpc_vlidar[1, sel_ck], -eppdir[sel_ck, 0], eppdir[sel_ck, 1])
            plt.scatter(epppole[0], epppole[1], 20, 'r')
            plt.show()

        return imcombined