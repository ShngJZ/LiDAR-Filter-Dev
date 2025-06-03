"""
LiDAR Point Cloud Cleaning Module

This module provides functionality for cleaning LiDAR point cloud data by identifying and removing
occluded points when projecting 3D LiDAR data onto 2D camera images. It uses geometric constraints
and epipolar geometry to determine occlusions.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import PIL.Image as Image
from scipy.interpolate import NearestNDInterpolator

class LiDARCleaner(nn.Module):
    """
    A PyTorch module for cleaning LiDAR point cloud data by detecting and removing occluded points.

    This class implements a method to identify and filter out LiDAR points that are occluded
    when projected onto a camera image. It uses camera intrinsics, extrinsics, and 3D LiDAR
    points to perform the cleaning operation.

    Args:
        intrinsic_cam (numpy.ndarray or torch.Tensor): Camera intrinsic matrix (3x3)
        extrinsic_LiDAR2Cam (numpy.ndarray or torch.Tensor): LiDAR to camera extrinsic matrix (3x4)
        LiDARPoints3D (numpy.ndarray or torch.Tensor): 3D LiDAR points (3xN)
        height (int): Image height in pixels
        width (int): Image width in pixels
        rszh (float, optional): Height resize factor (0-1). Defaults to 0.5
        rszw (float, optional): Width resize factor (0-1). Defaults to 1.0
        plotmarker_size (float, optional): Size of markers for visualization. Defaults to 5.0
        showimage (bool, optional): Whether to show background image in visualization. Defaults to False
    """
    def __init__(self,
                 intrinsic_cam,
                 extrinsic_LiDAR2Cam,
                 LiDARPoints3D,
                 height, width,
                 rszh=0.5, rszw=1.0,
                 plotmarker_size=5.0,
                 showimage=False
                 ):
        super().__init__()

        # Initialize and validate input matrices
        self.intrinsic_cam = self.check_intrinsic(copy.deepcopy(intrinsic_cam))
        self.extrinsic_LiDAR2Cam = self.check_extrinsic(copy.deepcopy(extrinsic_LiDAR2Cam))
        self.LiDARPoints3D = self.check_LiDARPoints3D(copy.deepcopy(LiDARPoints3D))

        # Validate and set image resize parameters
        assert rszh <= 1.0 and rszw <= 1.0, "Resize factors must be <= 1.0"
        self.height_rz, self.width_rz = int(height * rszh), int(width * rszw)
        self.height, self.width = int(height), int(width)
        self.rszh, self.rszw = float(self.height_rz / self.height), float(self.width_rz / self.width)
        self.resizeM = self.acquire_resizeM(self.rszh, self.rszw)

        # Visualization parameters
        self.plotmarker_size = plotmarker_size
        self.showimage = showimage

    def acquire_resizeM(self, rszh, rszw):
        """
        Creates a resize transformation matrix for 2D coordinates.

        Args:
            rszh (float): Height resize factor
            rszw (float): Width resize factor

        Returns:
            torch.Tensor: 3x3 resize transformation matrix
        """
        resizeM = torch.eye(3)
        resizeM[0, 0] = rszw
        resizeM[1, 1] = rszh
        return resizeM

    def check_intrinsic(self, intrinsic_cam):
        """
        Validates and converts camera intrinsic matrix to the required format.

        Args:
            intrinsic_cam: Input intrinsic matrix (numpy array or torch tensor)

        Returns:
            torch.Tensor: Validated intrinsic matrix as float tensor
        """
        if isinstance(intrinsic_cam, np.ndarray):
            intrinsic_cam = torch.from_numpy(intrinsic_cam)
        assert intrinsic_cam.shape[-1] == 3 and intrinsic_cam.shape[-2] == 3 and intrinsic_cam.device == torch.device("cpu")
        return intrinsic_cam.float()

    def check_extrinsic(self, extrinsic_LiDAR2Cam):
        """
        Validates and converts LiDAR to camera extrinsic matrix to the required format.

        Args:
            extrinsic_LiDAR2Cam: Input extrinsic matrix (numpy array or torch tensor)

        Returns:
            torch.Tensor: Validated extrinsic matrix as float tensor
        """
        if isinstance(extrinsic_LiDAR2Cam, np.ndarray):
            extrinsic_LiDAR2Cam = torch.from_numpy(extrinsic_LiDAR2Cam)
        assert extrinsic_LiDAR2Cam.shape[-2] == 3 and extrinsic_LiDAR2Cam.shape[-1] == 4 and extrinsic_LiDAR2Cam.device == torch.device("cpu")
        return extrinsic_LiDAR2Cam.float()

    def check_LiDARPoints3D(self, LiDARPoints3D):
        """
        Validates and converts 3D LiDAR points to homogeneous coordinates.

        Args:
            LiDARPoints3D: Input 3D points (numpy array or torch tensor)

        Returns:
            torch.Tensor: Validated points in homogeneous coordinates as float tensor
        """
        if isinstance(LiDARPoints3D, np.ndarray):
            LiDARPoints3D = torch.from_numpy(LiDARPoints3D)
        assert LiDARPoints3D.shape[-2] == 3 and LiDARPoints3D.device == torch.device("cpu")
        npts = LiDARPoints3D.shape[-1]
        # Convert to homogeneous coordinates by adding ones
        LiDARPoints3D = torch.cat([LiDARPoints3D, torch.ones([1, npts])], dim=0).contiguous()
        return LiDARPoints3D.float()

    def interpolated_depth(self, depth, querylocation):
        """
        Interpolates depth values at query locations using grid sampling.

        Args:
            depth (torch.Tensor): Input depth map
            querylocation (torch.Tensor): Query locations for interpolation

        Returns:
            torch.Tensor: Interpolated depth values at query locations
        """
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        if isinstance(querylocation, np.ndarray):
            querylocation = torch.from_numpy(querylocation)

        h, w = depth.shape[-2::]
        depth = depth.view([1, 1, h, w])

        # Normalize query locations to [-1, 1] range for grid_sample
        nquery, nsample, _ = querylocation.shape
        qx, qy = torch.split(querylocation, 1, dim=2)
        qx, qy = ((qx / w) - 0.5) * 2, ((qy / h) - 0.5) * 2
        querylocation = torch.cat([qx, qy], dim=2)
        querylocation = querylocation.view([1, nquery, nsample, 2])

        # Perform grid sampling and handle out-of-bounds points
        querydepth = torch.nn.functional.grid_sample(depth.cuda().float(), querylocation.cuda().float(), mode='nearest')
        oodselector = torch.nn.functional.grid_sample(torch.ones_like(depth).cuda().float(), querylocation.cuda().float(), mode='nearest')
        querydepth[oodselector == 0] = 1e5

        querydepth = querydepth.view([nquery, nsample, 1])
        return querydepth

    def inpainting_depth(self, visible_cam, rgb=None):
        """
        Performs depth map inpainting using nearest neighbor interpolation.

        Args:
            visible_cam (torch.Tensor): Visibility mask for camera points
            rgb (PIL.Image, optional): RGB image for visualization

        Returns:
            tuple: Inpainted depth map and associated point cloud information
        """
        # Create pure rotation matrix by removing translation
        pure_rotation = copy.deepcopy(self.extrinsic_LiDAR2Cam)
        pure_rotation[0:3, 3:4] = 0.0

        # Create coordinate grid for resized image
        grid_x, grid_y = np.meshgrid(range(self.width_rz), range(self.height_rz))

        # Project points using resized intrinsics
        prjpc, depths, visible_points = self.prj(
            self.resizeM @ self.intrinsic_cam,
            pure_rotation,
            self.LiDARPoints3D,
            height=self.height_rz, width=self.width_rz
        )
        visible_points = visible_points * visible_cam

        # Extract valid points for interpolation
        prjpc_val = prjpc[:, visible_points].cpu().numpy()
        depths_val = depths[visible_points].cpu().numpy()

        assert prjpc_val.shape[1] > 100, "Insufficient valid points for interpolation"

        # Perform nearest neighbor interpolation
        nearest_func = NearestNDInterpolator(prjpc_val.T, depths_val)
        inpait_d = nearest_func(grid_x, grid_y)

        return inpait_d, prjpc, depths, visible_points

    def prj(self, intrinsic, extrinsic, pc3D, height, width, min_dist=0.1):
        """
        Projects 3D points onto 2D image plane and checks visibility.

        Args:
            intrinsic (torch.Tensor): Camera intrinsic matrix
            extrinsic (torch.Tensor): Extrinsic transformation matrix
            pc3D (torch.Tensor): 3D points to project
            height (int): Image height
            width (int): Image width
            min_dist (float, optional): Minimum valid distance. Defaults to 0.1

        Returns:
            tuple: Projected points, depths, and visibility mask
        """
        # Project 3D points to 2D
        prjpc = intrinsic @ extrinsic @ pc3D
        prjpc[0, :] = prjpc[0, :] / (prjpc[2, :] + 1e-8)  # X coordinates
        prjpc[1, :] = prjpc[1, :] / (prjpc[2, :] + 1e-8)  # Y coordinates
        depth = prjpc[2, :]  # Depth values

        # Check visibility conditions
        visible_sel = (depth > min_dist)
        visible_sel = torch.logical_and(visible_sel, prjpc[0, :] > 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[0, :] < width - 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[1, :] > 0.5)
        visible_sel = torch.logical_and(visible_sel, prjpc[1, :] < height - 0.5)

        return prjpc[0:2, :], depth, visible_sel

    def pad_pose44(self, extrinsic):
        """
        Pads a 3x4 transformation matrix to 4x4 homogeneous form.

        Args:
            extrinsic (torch.Tensor): 3x4 transformation matrix

        Returns:
            torch.Tensor: 4x4 homogeneous transformation matrix
        """
        pose = torch.eye(4)
        h, w = extrinsic.shape
        pose[0:h, 0:w] = extrinsic
        return pose

    def epplinedir(self, prjpc_vlidar, selector=None):
        """
        Computes epipolar line directions for projected points.

        Args:
            prjpc_vlidar (torch.Tensor): Projected LiDAR points
            selector (torch.Tensor, optional): Point selection mask

        Returns:
            tuple: Epipolar directions, pure translation, and epipole coordinates
        """
        # Scale intrinsic matrix and compute pure rotation/translation
        intrinsic_cam_scaled = self.resizeM @ self.intrinsic_cam
        pure_rotation = self.pad_pose44(copy.deepcopy(self.extrinsic_LiDAR2Cam))
        pure_rotation[0:3, 3:4] = 0
        pure_translation = self.pad_pose44(copy.deepcopy(self.extrinsic_LiDAR2Cam)) @ torch.linalg.inv(pure_rotation)

        # Compute epipole
        epipole = -intrinsic_cam_scaled @ pure_translation[0:3, 3:4]
        eppx, eppy = (epipole[0, 0] / epipole[2, 0]).item(), (epipole[1, 0] / epipole[2, 0]).item()

        # Compute and normalize epipolar directions
        eppdir = torch.stack([eppx - prjpc_vlidar[0, :], eppy - prjpc_vlidar[1, :]], dim=0).contiguous()
        eppdir = torch.nn.functional.normalize(eppdir, dim=0)
        eppdir = eppdir.T

        return eppdir, pure_translation[0:3, :], (eppx, eppy)

    def backprj_prj(self, intrinsic, pure_translation, enumlocation, depthinterp):
        """
        Back-projects and re-projects points using pure translation.

        Args:
            intrinsic (torch.Tensor): Camera intrinsic matrix
            pure_translation (torch.Tensor): Pure translation matrix
            enumlocation (torch.Tensor): Point locations
            depthinterp (torch.Tensor): Interpolated depth values

        Returns:
            torch.Tensor: Re-projected points
        """
        # Prepare transformation matrices
        intrinsic = self.pad_pose44(intrinsic).cuda()
        pure_translation = self.pad_pose44(pure_translation).cuda()
        prjM = intrinsic @ pure_translation @ intrinsic.inverse()
        prjM = prjM.view([1, 1, 4, 4])

        # Create 3D points from depth and locations
        nquery, nsample, _ = enumlocation.shape
        qx, qy = torch.split(enumlocation, 1, dim=2)
        pts3D = torch.cat([qx * depthinterp, qy * depthinterp, depthinterp, torch.ones_like(depthinterp)], dim=2)
        pts3D = pts3D.view([nquery, nsample, 4, 1])

        # Project points
        pts3Dprj = prjM @ pts3D
        pts3Dprjx = pts3Dprj[:, :, 0, 0] / pts3Dprj[:, :, 2, 0]
        pts3Dprjy = pts3Dprj[:, :, 1, 0] / pts3Dprj[:, :, 2, 0]

        return torch.stack([pts3Dprjx, pts3Dprjy], dim=-1)

    def clean_python(self, intrinsic, pure_translation, depthmap, prjpc_lidar, prjpc_cam, eppdir, selector, srch_resolution=0.5):
        """
        Identifies occluded points using epipolar geometry.

        Args:
            intrinsic (torch.Tensor): Camera intrinsic matrix
            pure_translation (torch.Tensor): Pure translation matrix
            depthmap (torch.Tensor): Depth map
            prjpc_lidar (torch.Tensor): Projected LiDAR points
            prjpc_cam (torch.Tensor): Projected camera points
            eppdir (torch.Tensor): Epipolar directions
            selector (torch.Tensor): Point selection mask
            srch_resolution (float, optional): Search resolution. Defaults to 0.5

        Returns:
            torch.Tensor: Boolean mask of occluded points
        """
        # Select valid points
        prjpc_lidar_, prjpc_cam_ = prjpc_lidar[selector, :], prjpc_cam[selector, :]
        eppdir_ = eppdir[selector, :]

        # Create depth range for search
        mindist, maxdist = 1, 100
        samplenum = int(np.ceil((maxdist - mindist) / srch_resolution).item() + 1)
        sampled_range = torch.linspace(mindist, maxdist, samplenum).cuda()

        # Generate search locations along epipolar lines
        nanchor = len(eppdir_)
        enumlocation = prjpc_lidar_.view([nanchor, 1, 2]) + sampled_range.view([1, samplenum, 1]) * eppdir_.view([nanchor, 1, 2])

        # Interpolate depths and compute projections
        depthinterp = self.interpolated_depth(depthmap, enumlocation)
        pts3Dprj = self.backprj_prj(intrinsic, pure_translation, enumlocation, depthinterp)

        # Compute projection directions and identify occlusions
        nquery = len(prjpc_lidar_)
        prj_dir = pts3Dprj - prjpc_cam_.view([nquery, 1, 2])
        prj_dir = torch.nn.functional.normalize(prj_dir, dim=2)
        cosdiff = torch.sum(prj_dir * eppdir_.view([nquery, 1, 2]), dim=2)
        cosdiffmax, _ = torch.min(cosdiff, dim=1)
        occluded = cosdiffmax < 0

        return occluded

    def forward(self, rgb=None, debug=True):
        """
        Forward pass of the LiDAR cleaning process.

        Args:
            rgb (PIL.Image, optional): RGB image for visualization
            debug (bool, optional): Whether to generate debug visualizations. Defaults to True

        Returns:
            tuple: Filtered point visibility mask and visualization (if debug=True)
        """
        # Project LiDAR points to camera view
        camprj_vls, camdepth_vls, visible_points = self.prj(
            self.intrinsic_cam,
            self.extrinsic_LiDAR2Cam,
            self.LiDARPoints3D,
            height=self.height, width=self.width
        )

        # Perform depth inpainting
        inpait_d, prjpc_vlidar, depths_vlidar, visible_sel_vlidar = self.inpainting_depth(visible_cam=visible_points, rgb=None)

        # Compute epipolar geometry
        eppdir, pure_translation, epppole = self.epplinedir(prjpc_vlidar, visible_sel_vlidar)

        # Project points for cleaning
        prjpc_cam, _, _ = self.prj(
            self.resizeM @ self.intrinsic_cam,
            self.extrinsic_LiDAR2Cam,
            self.LiDARPoints3D,
            height=self.height_rz, width=self.width_rz
        )

        # Prepare tensors for cleaning
        inpait_d = torch.from_numpy(inpait_d).cuda().float()
        prjpc_vlidar, prjpc_cam = prjpc_vlidar.T.cuda().float(), prjpc_cam.T.cuda().float()
        eppdir = eppdir.cuda().float()

        # Identify occluded points
        occluded = self.clean_python(
            self.resizeM @ self.intrinsic_cam,
            pure_translation,
            inpait_d,
            prjpc_vlidar,
            prjpc_cam,
            eppdir,
            visible_sel_vlidar,
            srch_resolution=1.0
        )

        # Create visibility masks
        tomask = torch.zeros_like(occluded.cpu())
        tomask[occluded] = 1
        tomask_all = torch.zeros_like(visible_sel_vlidar)
        tomask_all[visible_sel_vlidar] = tomask

        # Apply cleaning
        visible_points_filtered = torch.clone(visible_points)
        visible_points_filtered[tomask_all] = 0

        if debug:
            # Get image dimensions for visualization
            vlsw, vlsh = rgb.size

            # Sort points by depth for proper visualization (far points first)
            sort_ind = camdepth_vls.argsort()
            camprj_vls = camprj_vls[:, sort_ind]

            # Configure matplotlib for non-interactive backend
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('Agg')  # Use non-GUI backend

            # Create visualization of original points
            plt.figure(figsize=(16, 9))
            if self.showimage:
                plt.imshow(rgb, alpha=0.85)  # Show background image if enabled
            plt.scatter(
                camprj_vls[0, visible_points[sort_ind]].numpy(),  # X coordinates
                camprj_vls[1, visible_points[sort_ind]].numpy(),  # Y coordinates
                c=1 / camdepth_vls[sort_ind][visible_points[sort_ind]],  # Color by inverse depth
                cmap=plt.cm.get_cmap('magma'),  # Use magma colormap
                s=self.plotmarker_size,  # Marker size
                vmin=0, vmax=0.28)  # Color range
            plt.axis('off')  # Hide axes
            plt.xlim([0, vlsw])  # Set X limits
            plt.ylim([vlsh, 0])  # Set Y limits (inverted)
            plt.savefig('tmp1.jpg', transparent=True, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()

            # Create visualization of filtered points
            plt.figure(figsize=(16, 9))
            if self.showimage:
                plt.imshow(rgb, alpha=0.85)
            plt.scatter(
                camprj_vls[0, visible_points_filtered[sort_ind]].numpy(),
                camprj_vls[1, visible_points_filtered[sort_ind]].numpy(),
                c=1 / camdepth_vls[sort_ind][visible_points_filtered[sort_ind]],
                cmap=plt.cm.get_cmap('magma'),
                s=self.plotmarker_size,
                vmin=0, vmax=0.28)
            plt.axis('off')
            plt.xlim([0, vlsw])
            plt.ylim([vlsh, 0])
            plt.savefig('tmp2.jpg', transparent=True, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()

            # Combine both visualizations side by side
            im1 = Image.open('tmp1.jpg')
            im2 = Image.open('tmp2.jpg')
            imcombined = np.concatenate([np.array(im1), np.array(im2)], axis=1)
            imcombined = Image.fromarray(imcombined)

            # Clean up temporary files
            os.remove('tmp1.jpg')
            os.remove('tmp2.jpg')

            return visible_points_filtered, tomask_all, imcombined

        else:
            # Return results without visualization in non-debug mode
            return visible_points_filtered, tomask_all, None