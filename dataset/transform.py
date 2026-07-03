import torchvision.transforms.functional as F
import torch.nn.functional
import torchvision.transforms
import torch

class ToTensor:
    def __call__(self, sample):
        left_image  = sample['left_image']
        right_image = sample['right_image']
        disparity   = sample['disparity']

        # Pytorch expects Tensors in order [Channels, Height, Width]
        left_image  = torch.from_numpy(left_image).permute(2, 0, 1).contiguous().float()
        right_image = torch.from_numpy(right_image).permute(2, 0, 1).contiguous().float()

        if disparity is not None:
            disparity = torch.from_numpy(disparity).contiguous()

        sample['left_image']  = left_image
        sample['right_image'] = right_image
        sample['disparity']   = disparity

        if sample.get('mask_occ') is not None:
            sample['mask_occ'] = torch.from_numpy(sample['mask_occ']).contiguous()

        return sample

class RandomCrop:
    """Crops images to enforce uniform dimensions."""
    def __init__(self, patch_size):
        self.crop_height = patch_size[0]
        self.crop_width  = patch_size[1]

    def __call__(self, sample):
        left_image  = sample['left_image']
        right_image = sample['right_image']
        disparity   = sample['disparity']

        h, w = left_image.shape[1:]
        if h < self.crop_height or w < self.crop_width:
            raise ValueError(
                f"Image size {(h, w)} is smaller than crop size {(self.crop_height, self.crop_width)}"
            )

        top  = torch.randint(0, h - self.crop_height + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_width  + 1, (1,)).item()

        sample['left_image']  = left_image[:,  top:top+self.crop_height, left:left+self.crop_width]
        sample['right_image'] = right_image[:, top:top+self.crop_height, left:left+self.crop_width]
        sample['disparity']   = disparity[top:top+self.crop_height, left:left+self.crop_width] \
                                if disparity is not None else None

        if sample.get('mask_occ') is not None:
            sample['mask_occ'] = sample['mask_occ'][top:top+self.crop_height, left:left+self.crop_width]

        return sample


class CenterCrop:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        left_image  = sample['left_image']
        right_image = sample['right_image']
        disparity   = sample['disparity']

        sample['left_image']  = F.center_crop(left_image,  self.patch_size)
        sample['right_image'] = F.center_crop(right_image, self.patch_size)

        if disparity is not None:
            # F.center_crop requires a channel dim — squeeze back to [H, W] afterwards
            sample['disparity'] = F.center_crop(disparity.unsqueeze(0), self.patch_size).squeeze(0)
        else:
            sample['disparity'] = None

        if sample.get('mask_occ') is not None:
            sample['mask_occ'] = F.center_crop(
                sample['mask_occ'].unsqueeze(0), self.patch_size
            ).squeeze(0)

        return sample