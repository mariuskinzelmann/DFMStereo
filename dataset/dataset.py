from __future__ import annotations
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import io
import imageio.v2 as imageio
import dataset.dataset_io
from dataset.augmentor import FlowAugmentor, SparseFlowAugmentor
import zipfile
import json
import re


class StereoDataset(data.Dataset):
    """
    Abstract base class for stereo datasets.

    Subclasses must implement ``_load_data()``, which populates:
        self.left_image_paths  : list[Path]
        self.right_image_paths : list[Path]
        self.disparity_paths   : list[Path | None]
        self.mask_occ_paths    : list[Path | None]  (only if self.load_mask)

    Args:
        root_dir    (str | Path):  Root directory of the dataset.
        aug_params  (dict | None): Augmentation parameters.
        mode        (str):         Dataset split, e.g. 'training' / 'testing'.
        load_mask   (bool):        Whether to load occlusion masks.
        sparse      (bool):        Use SparseFlowAugmentor instead of the dense
                                   one.  Ignored when aug_params is None.
    """
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str, load_mask: bool = False, sparse: bool = False):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.load_mask = load_mask
        self.sparse = sparse

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # --- Data Augmentor ---
        self.augmentor: FlowAugmentor | SparseFlowAugmentor | None = None
        if aug_params is not None:
            params = {k: v for k, v in aug_params.items() if k != "sparse"}
            if sparse:
                self.augmentor = SparseFlowAugmentor(**params)
            else:
                self.augmentor = FlowAugmentor(**params)

        self.left_image_paths: list[Path] = []
        self.right_image_paths: list[Path] = []
        self.disparity_paths: list[Path | None] = []
        self.mask_occ_paths: list[Path | None] = []

        self._load_data()

    def _load_data(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _load_data()")

    def _supports_masks(self) -> bool:
        """Override in subclasses that provide masks."""
        return False

    @staticmethod
    def _assert_exists(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    def _read_image(self, path: Path) -> np.ndarray:
        return np.array(dataset.dataset_io.read(path), dtype=np.uint8)

    def _read_disparity(self, path: Path) -> np.ndarray:
        return np.array(dataset.dataset_io.read(path), dtype=np.float32)

    def _read_mask(self, path: Path) -> np.ndarray:
        return np.array(dataset.dataset_io.readMask(path), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.left_image_paths)

    def __getitem__(self, idx: int) -> dict:
        img1 = self._read_image(self.left_image_paths[idx])
        img2 = self._read_image(self.right_image_paths[idx])

        disp_path = self.disparity_paths[idx] if self.disparity_paths else None
        disp = self._read_disparity(disp_path) if disp_path is not None else None

        mask_occ = None
        if self.load_mask:
            mask_path = self.mask_occ_paths[idx] if self.mask_occ_paths else None
            mask_occ  = self._read_mask(mask_path) if mask_path is not None else None

        valid = None
        if self.augmentor is not None:
            if self.sparse:
                valid = (disp < 1024) if disp is not None else None
                img1, img2, disp, valid, mask_occ = self.augmentor(img1, img2, disp, valid, mask_occ)
            else:
                img1, img2, disp, mask_occ = self.augmentor(img1, img2, disp, mask_occ)
                valid = (disp < 1024) if disp is not None else None

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        sample: dict = {"left_image": img1, "right_image": img2}

        if disp is not None:
            sample["disparity"] = torch.from_numpy(disp).float()

        if mask_occ is not None:
            sample["mask_occ"] = torch.from_numpy(mask_occ).float()

        if self.sparse and valid is not None:
            sample["valid"] = torch.from_numpy(valid.astype(np.float32))
            
        return sample

class KITTI12(StereoDataset):
    """
    KITTI 2012 Stereo Dataset.

    Args:
        root_dir    (str | Path): Root directory containing KITTI 2012 data.
        aug_params  (dict|None):  Augmentation parameters.
        mode        (str):        'training' or 'testing'.
        occ         (bool):       Use occluded disparities instead of non-occluded.
        load_mask   (bool):       Not supported — raises Error if True.
    """
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "training", occ: bool = False, load_mask: bool = False):
        assert not load_mask, "KITTI 2012 does not provide occlusion masks."
        assert mode in ("training", "testing"), f"Unknown mode '{mode}'. Expected 'training' or 'testing'."

        self.occ = occ
        # KITTI uses sparse ground-truth
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=True)

        if len(self.left_image_paths) != 194:
            raise RuntimeError(
                f"Expected 194 samples, but loaded {len(self.left_image_paths)}.")

    def _load_data(self) -> None:
        left_dir  = self.root_dir / self.mode / "colored_0"
        right_dir = self.root_dir / self.mode / "colored_1"
        disp_dir  = self.root_dir / self.mode / ("disp_occ" if self.occ else "disp_noc")

        left_paths = sorted(left_dir.glob("*_10.png"))
        if not left_paths:
            raise FileNotFoundError(f"No images matched '*_10.png' in {left_dir}")

        for lp in left_paths:
            rp = right_dir / lp.name
            dp = disp_dir  / lp.name
            self._assert_exists(lp, "Left image")
            self._assert_exists(rp, "Right image")
            self.left_image_paths.append(lp)
            self.right_image_paths.append(rp)
            self.disparity_paths.append(dp if self.mode == "training" else None)

    def _read_disparity(self, path: Path) -> np.ndarray:
        return (np.array(dataset.dataset_io.readDisparity_png(path), dtype=np.float32) / 256.0)
    

class KITTI15(StereoDataset):
    """
    KITTI 2015 Stereo Dataset.

    Args:
        root_dir    (str | Path): Root directory containing KITTI 2015 data.
        aug_params  (dict|None):  Augmentation parameters.
        mode        (str):        'training' or 'testing'.
        occ         (bool):       Use occluded disparities.
        load_mask   (bool):       Not supported — raises if True.
    """
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "training", occ: bool = False, load_mask: bool = False):
        assert not load_mask, "KITTI 2015 does not provide occlusion masks."
        assert mode in ("training", "testing"), f"Unknown mode '{mode}'. Expected 'training' or 'testing'."

        self.occ = occ
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=True)

        if len(self.left_image_paths) != 200:
            raise RuntimeError(
                f"Expected 200 samples, but loaded {len(self.left_image_paths)}.")

    def _load_data(self) -> None:
        left_dir  = self.root_dir / self.mode / "image_2"
        right_dir = self.root_dir / self.mode / "image_3"
        disp_dir  = self.root_dir / self.mode / ("disp_occ_0" if self.occ else "disp_noc_0")

        left_paths = sorted(left_dir.glob("*_10.png"))
        if not left_paths:
            raise FileNotFoundError(f"No images matched '*_10.png' in {left_dir}")

        for lp in left_paths:
            rp = right_dir / lp.name
            dp = disp_dir  / lp.name
            self._assert_exists(lp, "Left image")
            self._assert_exists(rp, "Right image")
            self.left_image_paths.append(lp)
            self.right_image_paths.append(rp)
            self.disparity_paths.append(dp if self.mode == "training" else None)

    def _read_disparity(self, path: Path) -> np.ndarray:
        return (np.array(dataset.dataset_io.readDisparity_png(path), dtype=np.float32) /  256.0)


class SceneFlow(StereoDataset):
    """
    SceneFlow benchmark dataset (dense ground truth).

    Args:
        root_dir    (str | Path): Root directory containing SceneFlow data.
        aug_params  (dict|None):  Augmentation parameters.
        mode        (str):        'training' or 'testing'.
        load_mask   (bool):       Not supported — raises if True.
    """
    FILENAMES = {
        "training": "SceneFlow_GANet_file_names.txt",
        "testing":  "SceneFlow_Test_GANet_file_names.txt",
    }

    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "training", load_mask: bool = False):
        if load_mask:
            raise ValueError("SceneFlow does not provide occlusion masks.")
        if mode not in self.FILENAMES:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {list(self.FILENAMES)}.")

        self.file_list_path = Path(__file__).parent / "filenames" / self.FILENAMES[mode]
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=False)

    def _load_data(self) -> None:
        with open(self.file_list_path) as f:
            all_paths = {line.strip() for line in f}

        for path in sorted(all_paths):
            if "frames_finalpass" in path and "/left/" in path and path.endswith(".png"):
                lp = path
                rp = lp.replace("/left/", "/right/")
                dp = lp.replace("frames_finalpass", "disparity").replace(".png", ".pfm")

                if rp in all_paths and dp in all_paths:
                    self._assert_exists(self.root_dir / lp, "Left image")
                    self._assert_exists(self.root_dir / rp, "Right image")
                    self._assert_exists(self.root_dir / dp, "Disparity map")
                    self.left_image_paths.append(self.root_dir / lp)
                    self.right_image_paths.append(self.root_dir / rp)
                    self.disparity_paths.append(self.root_dir / dp)


class Middlebury(StereoDataset):
    """
    Middlebury Stereo Dataset (MiddleEval).

    Args:
        root_dir    (str | Path): Root directory.
        aug_params  (dict|None):  Augmentation parameters.
        split       (str):        Dataset split identifier (e.g. 'MiddEval3').
        mode        (str):        'training' or 'test'.
        resolution  (str):        'Q' (quarter) or 'H' (half).
        load_mask   (bool):       If True, load per-pixel occlusion masks.
    """
    def __init__(self,root_dir: str | Path,aug_params: dict | None, split: str  = "MiddEval3", mode: str  = "training", resolution: str  = "H", load_mask: bool = True):
        assert split in ("MiddEval3",)
        assert mode in ("training", "test")
        assert resolution in ("Q", "H")

        self.split      = split
        self.resolution = resolution
        # Middlebury has dense gt → non-sparse augmentor
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=False)

    def _supports_masks(self) -> bool:
        return True

    def _load_data(self) -> None:
        data_dir = self.root_dir / self.split / (self.mode + self.resolution)
        self._assert_exists(data_dir, "Data directory")

        for lp in sorted(data_dir.rglob("im0.png")):
            rp = lp.parent / "im1.png"
            self._assert_exists(lp, "Left image")
            self._assert_exists(rp, "Right image")
            self.left_image_paths.append(lp)
            self.right_image_paths.append(rp)

            if self.mode == "training":
                dp  = lp.parent / "disp0GT.pfm"
                mp  = lp.parent / "mask0nocc.png"
                self._assert_exists(dp, "Disparity map")
                self._assert_exists(mp, "Occlusion mask")
                self.disparity_paths.append(dp)
                self.mask_occ_paths.append(mp if self.load_mask else None)
            else:
                self.disparity_paths.append(None)
                self.mask_occ_paths.append(None)


class ETH3D(StereoDataset):
    """
    ETH3D Two-View Stereo Dataset.

    Args:
        root_dir    (str | Path): Root directory.
        aug_params  (dict|None):  Augmentation parameters.
        mode        (str):        'training' or 'testing'.
        load_mask   (bool):       If True, load per-pixel occlusion masks.
    """
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "training", load_mask: bool = True):
        assert mode in ("training", "testing"), f"Unknown mode '{mode}'. Expected 'training' or 'testing'."
        # ETH3D sparse ground truth
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=True)

    def _load_data(self) -> None:
        data_dir = self.root_dir / self.mode
        self._assert_exists(data_dir, "Data directory")

        for lp in sorted(data_dir.rglob("im0.png")):
            rp = lp.parent / "im1.png"
            self._assert_exists(lp, "Left image")
            self._assert_exists(rp, "Right image")
            self.left_image_paths.append(lp)
            self.right_image_paths.append(rp)

            if self.mode == "training":
                dp  = lp.parent / "disp0GT.pfm"
                mp  = lp.parent / "mask0nocc.png"
                self._assert_exists(dp, "Disparity map")
                self._assert_exists(mp, "Occlusion mask")
                self.disparity_paths.append(dp)
                self.mask_occ_paths.append(mp if self.load_mask else None)
            else:
                self.disparity_paths.append(None)
                self.mask_occ_paths.append(None)


class BoosterQ(StereoDataset):
    """
    BoosterQ Stereo Dataset.

    Args:
        root_dir    (str | Path): Root directory.
        aug_params  (dict|None):  Augmentation parameters.
        mode        (str):        'train' or 'test'.
        setup       (str):        'balanced'.
        resolution  (str):        'Q', 'H', or 'F'.
        load_mask   (bool):       If True, load per-pixel occlusion masks.
    """
    _SCALE_MAP = {"F": 1.0, "H": 0.5, "Q": 0.25}
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "train", setup: str = "balanced", resolution: str = "H", load_mask: bool = True):
        assert setup in ("balanced",)
        assert mode in ("train", "test")
        assert resolution in ("Q", "H", "F")

        self.setup        = setup
        self.scale_factor = self._SCALE_MAP[resolution]
        # dense disparity (.npy)
        super().__init__(root_dir, aug_params, mode, load_mask, sparse=False)

    def _supports_masks(self) -> bool:
        return True

    def _load_data(self) -> None:
        data_dir = self.root_dir / self.mode / self.setup
        self._assert_exists(data_dir, "Data directory")

        for scene in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            left_dir       = scene / "camera_00"
            right_dir      = scene / "camera_02"
            disparity_path = scene / "disp_00.npy"
            mask_occ_path  = scene / "mask_00.png"

            for f in sorted(left_dir.iterdir()):
                lp = left_dir  / f.name
                rp = right_dir / f.name
                self._assert_exists(lp, "Left image")
                self._assert_exists(rp, "Right image")
                self.left_image_paths.append(lp)
                self.right_image_paths.append(rp)

                if self.mode == "train":
                    self._assert_exists(disparity_path, "Disparity map")
                    self._assert_exists(mask_occ_path, "Occlusion mask")
                    self.disparity_paths.append(disparity_path)
                    self.mask_occ_paths.append(mask_occ_path if self.load_mask else None)
                else:
                    self.disparity_paths.append(None)
                    self.mask_occ_paths.append(None)

    def _read_disparity(self, path: Path) -> np.ndarray:
        return np.ascontiguousarray(np.load(path), dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        img1 = self._read_image(self.left_image_paths[idx])
        img2 = self._read_image(self.right_image_paths[idx])

        disp_path = self.disparity_paths[idx] if self.disparity_paths else None
        disp = self._read_disparity(disp_path) if disp_path is not None else None

        mask_occ = None
        if self.load_mask:
            mask_path = self.mask_occ_paths[idx] if self.mask_occ_paths else None
            mask_occ  = self._read_mask(mask_path) if mask_path is not None else None

        if self.scale_factor != 1.0:
            h, w = img1.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)

            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if disp is not None:
                disp = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_NEAREST) * self.scale_factor

            if mask_occ is not None:
                mask_occ = cv2.resize(mask_occ, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if self.augmentor is not None:
            img1, img2, disp, mask_occ = self.augmentor(img1, img2, disp, mask_occ)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        sample: dict = {"left_image": img1, "right_image": img2}

        if disp is not None:
            sample["disparity"] = torch.from_numpy(disp).float()

        if mask_occ is not None:
            sample["mask_occ"] = torch.from_numpy(mask_occ).float()

        return sample

class FSD(StereoDataset):
    def __init__(self, root_dir: str | Path, aug_params: dict | None, mode: str = "training", load_mask: bool = False):
        if load_mask:
            raise ValueError("SceneFlow does not provide occlusion masks.")
        self._zip_cache: dict[Path, zipfile.ZipFile] = {}
        super().__init__(root_dir, aug_params, mode, load_mask=False, sparse=False)

    @staticmethod
    def _frame_count_from_seq(seq: str) -> int:
        match = re.search(r'_(\d+)_\d+$', seq)
        return int(match.group(1)) if match else 0

    def _load_data(self) -> None:
        PATH_JSON = Path(__file__).parent / 'filenames' / 'fsd_filenames.json'
        with open(PATH_JSON) as f:
            data_scenes = json.load(f)

        for zip_name, sequences in data_scenes.items():
            zip_path = self.root_dir / zip_name

            for seq in sequences:
                print(f"Processing: {seq}")
                seq = seq.lstrip("/")

                n_frames = self._frame_count_from_seq(seq)
                if seq == "manipulation_v5-b2_realistic_large_warehouse_stocked_2500_3":
                    n_frames = 1515
                if seq == "manipulation_v5-b2_realistic_large_warehouse_stocked_2500_5":
                    n_frames = 2373

                for i in range(n_frames):
                    stem = f"{i:04d}" if n_frames > 1000 else f"{i:03d}"
                    self.left_image_paths.append((zip_path, f"{seq}/dataset/data/left/rgb/{stem}.jpg"))
                    self.right_image_paths.append((zip_path, f"{seq}/dataset/data/right/rgb/{stem}.jpg"))
                    self.disparity_paths.append((zip_path, f"{seq}/dataset/data/left/disparity/{stem}.png"))
        """ for zip_name, sequences in data_scenes.items():
            zip_path = self.root_dir / zip_name
            with zipfile.ZipFile(zip_path) as zf:
                # Build a set of all member names for O(1) lookup
                members: set[str] = set(zf.namelist())

                for seq in sequences:
                    seq = seq.lstrip("/")  # strip leading slash → "amr_v5-b2_chaos_2500_1"
                    print(f"Processing: {seq}")

                    left_rgb_prefix  = f"{seq}/dataset/data/left/rgb/"
                    right_rgb_prefix = f"{seq}/dataset/data/right/rgb/"
                    disp_prefix      = f"{seq}/dataset/data/left/disparity/"

                    # Collect and sort left RGB frames by stem ("0000", "0001", …)
                    left_frames: dict[str, str] = {
                        Path(m).stem: m
                        for m in members
                        if m.startswith(left_rgb_prefix) and m.endswith(".jpg")
                    }
                    right_frames: dict[str, str] = {
                        Path(m).stem: m
                        for m in members
                        if m.startswith(right_rgb_prefix) and m.endswith(".jpg")
                    }
                    disp_frames: set[str] = {
                        Path(m).stem
                        for m in members
                        if m.startswith(disp_prefix) and m.endswith(".png")
                    }

                    # Only emit samples where both left and right exist
                    common_frames = sorted(left_frames.keys() & right_frames.keys())

                    for stem in common_frames:
                        self.left_image_paths.append(
                            zipfile.Path(zip_path, left_frames[stem])
                        )
                        self.right_image_paths.append(
                            zipfile.Path(zip_path, right_frames[stem])
                        )
                        if stem in disp_frames:
                            self.disparity_paths.append(
                                zipfile.Path(zip_path, f"{disp_prefix}{stem}.png")
                            )
                        else:
                            self.disparity_paths.append(None) """

    def _get_zip(self, zip_path: Path) -> zipfile.ZipFile:
        if zip_path not in self._zip_cache:
            self._zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")
        return self._zip_cache[zip_path]

    def _read_bytes(self, entry: tuple[Path, str]) -> bytes:
        zip_path, member = entry
        with self._get_zip(zip_path).open(member) as f:
            return f.read()

    def _read_image(self, entry: tuple[Path, str]) -> np.ndarray:
        return imageio.imread(io.BytesIO(self._read_bytes(entry)))

    def _read_disparity(self, entry: tuple[Path, str]) -> np.ndarray:
        raw = imageio.imread(io.BytesIO(self._read_bytes(entry)))
        return dataset.dataset_io.depth_uint8_decoding(raw)

    """ def _read_image(self, path: zipfile.Path) -> np.ndarray:
        data = path.read_bytes()
        return imageio.imread(io.BytesIO(data))

    def _read_disparity(self, path: zipfile.Path) -> np.ndarray:
        data = path.read_bytes()
        raw = imageio.imread(io.BytesIO(data))
        return dataset.dataset_io.depth_uint8_decoding(raw) """
    
    def __getitem__(self, idx: int) -> dict:
        img1 = self._read_image(self.left_image_paths[idx])
        img2 = self._read_image(self.right_image_paths[idx])
        disp = self._read_disparity(self.disparity_paths[idx])

        mask_occ = None

        if self.augmentor is not None:
            img1, img2, disp, mask_occ = self.augmentor(img1, img2, disp, mask_occ)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        sample: dict = {
            "left_image":  img1,
            "right_image": img2,
            "disparity":   torch.from_numpy(disp).float(),
        }

        return sample