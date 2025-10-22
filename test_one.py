import formalpdf
import sys

import ocrtransforms.transforms as OT

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string



class FilterInvalidBoxes(object):
    def __init__(self, min_size=1):
        self.min_size = min_size
    def __call__(self, img, target):
        if target and "boxes" in target and len(target["boxes"]) > 0:
            b = target["boxes"]
            w = (b[:,2] - b[:,0])
            h = (b[:,3] - b[:,1])
            keep = (w >= self.min_size) & (h >= self.min_size)
            for k in ["boxes","labels","area","iscrowd","masks"]:
                if k in target and isinstance(target[k], torch.Tensor):
                    target[k] = target[k][keep]
        return img, target


def test_one(path):
    doc = formalpdf.open(path)

    for page in doc:
        image = page.render()
        pipeline = Compose([
            # Geometry (very mild, keeps boxes valid)
            OT.RandomRotateSmall(max_degrees=2.0, p=0.30),
            OT.RandomShearSmall(max_shear=3.0, p=0.30),

            # Photometric / codec / optics
            OT.RandomBrightnessContrastGammaSharpness(p=0.70,
                brightness=(0.92, 1.08),
                contrast=(0.90, 1.12),
                gamma=(0.90, 1.10),
                sharpness=(0.85, 1.20),
            ),
            OT.RandomGaussianBlur(p=0.15, sigma=(0.3, 1.1)),
            OT.RandomGaussianNoise(p=0.20, std=(3.0, 10.0)),
            OT.RandomJPEGCompression(p=0.20, quality=(40, 85)),
            OT.RandomMorphology(p=0.12),
            OT.RandomToGrayscale(p=0.08),

        ])
        image, _ = pipeline(image, None)

        image.show()

    doc.document.close()


if __name__ == "__main__":
    test_one(sys.argv[1])
