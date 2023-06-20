from __future__ import print_function, division
import os, sys
prj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, prj_root)
sys.path.append('core')

from core.utils.utils import tensor2disp, tensor2rgb
from core import occlusion_detector

if __name__ == '__main__':
    odter = occlusion_detector.apply

    # intrinsic : Camera Intrinsic
    # relpose: Relative Camrea Pose Between Camera and LiDAR
    # monocdepth: Monocular Depthmap, produced by image filling
    occ_selector = odter(intrinsic.cuda(), relpose.cuda(), monocdepth.cuda(), float(1e10))

    fig1 = tensor2rgb(image1, viewind=0)
    fig4 = tensor2disp(occ_selector.float(), vmax=1, viewind=0)
    fig5 = tensor2disp(1 / mD_pred, vmax=0.15, viewind=0)