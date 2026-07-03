import cv2
import random
import warnings
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter, Compose, functional

class AdjustGamma:
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gain_min  = gain_min
        self.gain_max  = gain_max

    def __call__(self, sample):
        gain  = random.uniform(self.gain_min,  self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return (f"AdjustGamma(gamma=({self.gamma_min},{self.gamma_max}), "
                f"gain=({self.gain_min},{self.gain_max}))")


class FlowAugmentor:
    """
    Data augmentor for dense stereo datasets.

    Applies, in order:
        1. Photometric augmentation  (color jitter + gamma)
        2. Eraser / occlusion augmentation
        3. Spatial augmentation      (scale + crop, optional flip)

    All spatial operations keep disparity and mask_occ in sync with the images.

    Args:
        crop_size        (tuple[int,int]): Output (H, W) after cropping.
        spatial_scale    (bool):          Enable random scale augmentation.
        min_scale        (float):         log2 lower bound for scale factor.
        max_scale        (float):         log2 upper bound for scale factor.
        yjitter          (bool):          Apply independent vertical jitter to the
                                          right image crop origin.
        saturation_range (list[float]):   [min, max] saturation multiplier.
        gamma            (list[float]):   [gamma_min, gamma_max, gain_min, gain_max].
    """

    def __init__(self, crop_size, spatial_scale=True, min_scale=-0.2, max_scale=0.5, yjitter= False, saturation_range=(0.6, 1.4), gamma=(1, 1, 1, 1)):
        self.crop_size = crop_size
        self.spatial_scale = spatial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.yjitter = yjitter

        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        self.photo_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4,
                        saturation=list(saturation_range), hue=0.5 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.occlusion_aug_prob = 0.5

    def color_transform(self, img1: np.ndarray, img2: np.ndarray):
        """
        Photometric Data Augmentation.
        Applies Photometric Data Augmenations asymmetrically with probability ``p=0.2``.
        """
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            stack = np.concatenate([img1, img2], axis=0)
            stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
            img1, img2 = np.split(stack, 2, axis=0)
        return img1, img2

    def occlusion_transform(self, img1: np.ndarray, img2: np.ndarray,bounds=(50, 100)):
        """
        Performs in order:
            1) Compute Average RGB Value in ``image2``.
            2) Replaces Rectangle with size ``bounds`` in ``image2`` with average
            RGB-Value to simulate occlusion.
        """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.occlusion_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2


    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray | None, mask_occ: np.ndarray | None):
        """
        Applies in order:
            1. Random Isotropic Scaling with ``p=0.2``. Random Anisotropic Scaling with ``p=0.8``.
            2. Jitter if ``args.jitter`` and crop according to ``self.crop_size``.
        """
        if self.spatial_scale:
            ht, wd = img1.shape[:2]

            min_scale = max(
                (self.crop_size[0] + 8) / ht,
                (self.crop_size[1] + 8) / wd,
            )
            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = max(scale_x, min_scale)
            scale_y = max(scale_y, min_scale)

            if np.random.rand() < self.spatial_aug_prob:
                img1, img2, disp, mask_occ = self._resize_all(img1, img2, disp, mask_occ, scale_x, scale_y)

        ht, wd = img1.shape[:2]

        # Jitter if args.jitter and crop.
        if self.yjitter:
            y0 = np.random.randint(2, ht - self.crop_size[0] - 2)
            x0 = np.random.randint(2, wd - self.crop_size[1] - 2)
            y1 = y0 + np.random.randint(-2, 3)

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] if disp is not None else None
            mask_occ = mask_occ[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] if mask_occ is not None else None
        else:
            y0 = np.random.randint(0, ht - self.crop_size[0])
            x0 = np.random.randint(0, wd - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] if disp is not None else None
            mask_occ = mask_occ[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] if mask_occ is not None else None

        return img1, img2, disp, mask_occ


    @staticmethod
    def _resize_all(img1, img2, disp, mask_occ, fx, fy):
        img1 = cv2.resize(img1, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        if disp is not None:
            disp = cv2.resize(disp, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST) * fx  # scale disparity
        if mask_occ is not None:
            mask_occ = cv2.resize(mask_occ, None, fx=fx, fy=fy,interpolation=cv2.INTER_NEAREST)
        return img1, img2, disp, mask_occ


    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray | None = None, mask_occ: np.ndarray | None = None):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.occlusion_transform(img1, img2)
        img1, img2, disp, mask_occ = self.spatial_transform(img1, img2, disp, mask_occ)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp) if disp is not None else None
        mask_occ = np.ascontiguousarray(mask_occ) if mask_occ is not None else None

        return img1, img2, disp, mask_occ


# ──────────────────────────────────────────────────────────────
#  Sparse augmentor for KITTI, MiddleBury, ETH3D and Booster
# ──────────────────────────────────────────────────────────────

class SparseFlowAugmentor:
    """
    Data augmentor for sparse stereo datasets (e.g. KITTI).

    Keeps a ``valid`` mask in sync with all spatial operations.
    mask_occ is also kept in sync where provided.

    Args:
        crop_size        (tuple[int,int]): Output (H, W) after cropping.
        min_scale        (float):         log2 lower bound for scale factor.
        max_scale        (float):         log2 upper bound for scale factor.
        yjitter          (bool):          Not used for sparse (kept for API compat).
        saturation_range (list[float]):   [min, max] saturation multiplier.
        gamma            (list[float]):   [gamma_min, gamma_max, gain_min, gain_max].
    """

    def __init__(self, crop_size, spatial_scale=True, min_scale=-0.2, max_scale=0.5, yjitter=False, saturation_range=(0.7, 1.3), gamma=(1, 1, 1, 1)):
        self.crop_size = crop_size
        self.spatial_scale = spatial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        self.photo_aug = Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=list(saturation_range), hue=0.3 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.occlusion_aug_prob = 0.5


    def color_transform(self, img1: np.ndarray, img2: np.ndarray):
        """Symmetric photometric augmentation (sparse datasets)."""
        #if np.random.rand() < self.asymmetric_color_aug_prob:
        #    img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
        #    img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        #else:
        stack = np.concatenate([img1, img2], axis=0)
        stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
        img1, img2 = np.split(stack, 2, axis=0)
        return img1, img2


    def occlusion_transform(self, img1: np.ndarray, img2: np.ndarray):
        """
        Performs in order:
            1) Compute Average RGB Value in ``image2``.
            2) Replaces Rectangle with size ``bounds`` in ``image2`` with average
            RGB-Value to simulate occlusion.
        """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.occlusion_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    @staticmethod
    def _resize_sparse(disp: np.ndarray, valid: np.ndarray, fx=1.0, fy=1.0):
        """Resize a sparse disparity map and its validity mask."""
        ht, wd = disp.shape[:2]
        ys, xs = np.meshgrid(np.arange(ht), np.arange(wd), indexing='ij')

        coords = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
        disp_v = disp.ravel().astype(np.float32)
        valid_v = valid.ravel().astype(np.float32)

        sel_coords = coords[valid_v >= 1]
        sel_disp = disp_v[valid_v >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        new_coords = sel_coords * np.array([fx, fy], dtype=np.float32)
        new_disp = sel_disp * fx   # disparity scales with x

        xx = np.round(new_coords[:, 0]).astype(np.int32)
        yy = np.round(new_coords[:, 1]).astype(np.int32)
        v = (xx >= 0) & (xx < wd1) & (yy >= 0) & (yy < ht1)
        xx, yy = xx[v], yy[v]
        new_disp = new_disp[v]

        disp_out = np.zeros([ht1, wd1], dtype=np.float32)
        valid_out = np.zeros([ht1, wd1], dtype=np.int32)
        disp_out[yy, xx]  = new_disp
        valid_out[yy, xx] = 1
        return disp_out, valid_out


    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray | None, valid: np.ndarray | None, mask_occ: np.ndarray | None):
        """
        Applies in order:
            1. Random Isotropic Scaling with ``p=0.2``.
            2. Crop according to ``self.crop_size``.
        """
        ht, wd = img1.shape[:2]

        if self.spatial_scale:
            # For Sparse Ground-Truth Data we only scale isotropically.
            min_scale = max(
                (self.crop_size[0] + 1) / ht,
                (self.crop_size[1] + 1) / wd,
            )
            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = max(scale, min_scale)
            scale_y = max(scale, min_scale)
            
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            if disp is not None and valid is not None:
                disp, valid = self._resize_sparse(disp, valid, fx=scale_x, fy=scale_y)
            if mask_occ is not None:
                mask_occ = cv2.resize(mask_occ, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        ht, wd = img1.shape[:2]
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, max(1, ht - self.crop_size[0] + margin_y))
        x0 = np.random.randint(-margin_x, max(1, wd - self.crop_size[1] + margin_x))
        y0 = np.clip(y0, 0, ht - self.crop_size[0])
        x0 = np.clip(x0, 0, wd - self.crop_size[1])

        def _crop(arr):
            return arr[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] if arr is not None else None

        img1 = _crop(img1)
        img2 = _crop(img2)
        disp = _crop(disp)
        valid = _crop(valid)
        mask_occ = _crop(mask_occ)

        return img1, img2, disp, valid, mask_occ
    
    
    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray | None = None, valid: np.ndarray | None = None, mask_occ: np.ndarray | None = None):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.occlusion_transform(img1, img2)
        img1, img2, disp, valid, mask_occ = self.spatial_transform(img1, img2, disp, valid, mask_occ)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp) if disp is not None else None
        valid = np.ascontiguousarray(valid) if valid is not None else None
        mask_occ = np.ascontiguousarray(mask_occ) if mask_occ is not None else None

        return img1, img2, disp, valid, mask_occ