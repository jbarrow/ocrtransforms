import io
import math
import torch
import random

import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageFilter

# ---------- Geometry helpers (apply 2x3 affine to bboxes) ----------
def _boxes_xyxy_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: (N,4) -> corners: (N,4,2)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    return torch.stack([
        torch.stack([x1, y1], dim=-1),
        torch.stack([x2, y1], dim=-1),
        torch.stack([x2, y2], dim=-1),
        torch.stack([x1, y2], dim=-1),
    ], dim=1)

def _apply_affine_to_points(pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    # pts: (..., 2), M: (2,3)
    # homogenous multiply
    ones = torch.ones_like(pts[..., :1])
    pts_h = torch.cat([pts, ones], dim=-1)            # (..., 3)
    M_full = M                                        # (2,3)
    out = torch.matmul(pts_h, M_full.T)               # (..., 2)
    return out

def _affine_boxes_xyxy(boxes: torch.Tensor, M: torch.Tensor, w: int, h: int) -> torch.Tensor:
    # Apply 2x3 forward affine about image center (already baked into M)
    corners = _boxes_xyxy_to_corners(boxes)           # (N,4,2)
    corners_t = _apply_affine_to_points(corners, M)   # (N,4,2)
    x_min, _ = corners_t[..., 0].min(dim=1)
    y_min, _ = corners_t[..., 1].min(dim=1)
    x_max, _ = corners_t[..., 0].max(dim=1)
    y_max, _ = corners_t[..., 1].max(dim=1)
    new_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    # clamp to image frame
    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(0, w)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(0, h)
    return new_boxes

def _update_area_size(target: dict, w: int, h: int):
    target["size"] = torch.tensor([h, w])
    if "boxes" in target:
        b = target["boxes"]
        area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        target["area"] = area
    return target


class RandomRotateSmall(object):
    """
    Small deskew for scans/phone captures. Keeps canvas size fixed.
    max_degrees: typical 1.0–3.0
    """
    def __init__(self, max_degrees=2.0, p=0.3):
        self.max_degrees = float(max_degrees)
        self.p = p

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target

        angle = random.uniform(-self.max_degrees, self.max_degrees)
        w, h = img.size
        cx, cy = w * 0.5, h * 0.5

        # Apply to image
        img = F.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=(0.0, 0.0), center=[cx, cy])

        if target is None or "boxes" not in target or len(target["boxes"]) == 0:
            return img, target

        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        # Forward affine matrix rotating about center
        tx = cx - c * cx + s * cy
        ty = cy - s * cx - c * cy
        M = torch.tensor([[c, -s, tx],
                          [s,  c, ty]], dtype=torch.float32)

        boxes = target["boxes"].clone()
        boxes = _affine_boxes_xyxy(boxes, M, w, h)
        target = target.copy()
        target["boxes"] = boxes
        target = _update_area_size(target, w, h)

        # Optional: masks
        if "masks" in target:
            mlist = []
            for m in target["masks"]:
                pm = F.to_pil_image(m.byte() * 255)
                pm = F.affine(pm, angle=angle, translate=(0,0), scale=1.0, shear=(0.0,0.0), center=[cx,cy])
                tm = F.pil_to_tensor(pm).squeeze(0) > 127
                mlist.append(tm)
            target["masks"] = torch.stack(mlist, dim=0)

        return img, target



class RandomShearSmall(object):
    """
    Mild shear to mimic camera keystone / curl.
    max_shear: degrees for X and Y (use 1–5 deg)
    """
    def __init__(self, max_shear=3.0, p=0.3):
        self.max_shear = float(max_shear)
        self.p = p

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target

        sx = random.uniform(-self.max_shear, self.max_shear)
        sy = random.uniform(-self.max_shear, self.max_shear)
        w, h = img.size
        cx, cy = w * 0.5, h * 0.5

        # Image first
        img = F.affine(img, angle=0.0, translate=(0, 0), scale=1.0, shear=(sx, sy), center=[cx, cy])

        if target is None or "boxes" not in target or len(target["boxes"]) == 0:
            return img, target

        shx = math.tan(math.radians(sx))
        shy = math.tan(math.radians(sy))
        # Shear about center: [1 shx; shy 1] with translation so center stays fixed
        tx = cx - (1.0) * cx - shx * cy
        ty = cy - shy * cx - (1.0) * cy
        M = torch.tensor([[1.0, shx, tx],
                          [shy, 1.0, ty]], dtype=torch.float32)

        boxes = target["boxes"].clone()
        boxes = _affine_boxes_xyxy(boxes, M, w, h)
        target = target.copy()
        target["boxes"] = boxes
        target = _update_area_size(target, w, h)

        # Masks
        if "masks" in target:
            mlist = []
            for m in target["masks"]:
                pm = F.to_pil_image(m.byte() * 255)
                pm = F.affine(pm, angle=0.0, translate=(0,0), scale=1.0, shear=(sx,sy), center=[cx,cy])
                tm = F.pil_to_tensor(pm).squeeze(0) > 127
                mlist.append(tm)
            target["masks"] = torch.stack(mlist, dim=0)

        return img, target


class RandomBrightnessContrastGammaSharpness(object):
    """
    Lightweight photometric jiggles common in scanners/cameras.
    """
    def __init__(self, p=0.7,
                 brightness=(0.9, 1.1),
                 contrast=(0.9, 1.1),
                 gamma=(0.9, 1.1),
                 sharpness=(0.9, 1.1)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.sharpness = sharpness

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        # Brightness
        if random.random() < 0.8:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(*self.brightness))
        # Contrast
        if random.random() < 0.8:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(*self.contrast))
        # Gamma (approx via point LUT)
        if random.random() < 0.6:
            g = random.uniform(*self.gamma)
            inv = 1.0 / max(g, 1e-6)
            img = img.point(lambda x: int(255 * ((x / 255.0) ** inv)))
        # Sharpness
        if random.random() < 0.7:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(*self.sharpness))
        return img, target


class RandomGaussianBlur(object):
    def __init__(self, p=0.15, sigma=(0.3, 1.1)):
        self.p = p
        self.sigma = sigma

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        sig = random.uniform(*self.sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sig)), target


class RandomGaussianNoise(object):
    def __init__(self, p=0.2, std=(3.0, 12.0)):
        self.p = p
        self.std = std

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0.0, random.uniform(*self.std), size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), target


class RandomJPEGCompression(object):
    def __init__(self, p=0.2, quality=(35, 85)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        q = random.randint(*self.quality)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=False, progressive=False)
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")
        return comp, target


class RandomMorphology(object):
    """
    Mimic ink bleed / light erosion with small Min/Max filters.
    """
    def __init__(self, p=0.12):
        self.p = p

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        # Choose op
        if random.random() < 0.5:
            return img.filter(ImageFilter.MinFilter(size=3)), target  # erode thin strokes
        else:
            return img.filter(ImageFilter.MaxFilter(size=3)), target  # dilate (ink bleed)


class RandomToGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img: Image.Image, target: dict):
        if random.random() >= self.p:
            return img, target
        return img.convert("L").convert("RGB"), target
