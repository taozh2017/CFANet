import os
from PIL import Image
import random
import numpy as np
from PIL import ImageEnhance

from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset
from mindspore import ops


# several data augumentation strategies
def cv_random_flip(img, label, edge):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge


def randomCrop(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def randomPeper_eg(img, edge):
    img = np.array(img)
    edge = np.array(edge)

    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0
            edge[randX, randY] = 0

        else:

            img[randX, randY] = 255
            edge[randX, randY] = 255

    return Image.fromarray(img), Image.fromarray(edge)


# dataset for training
class PolypObjDataset:
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        # get filenames

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.egs = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]

        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.egs = sorted(self.egs)

        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()

        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        eg = self.binary_loader(self.egs[index])

        # data augumentation
        image, gt, eg = cv_random_flip(image, gt, eg)
        image, gt, eg = randomCrop(image, gt, eg)
        image, gt, eg = randomRotation(image, gt, eg)

        image = colorEnhance(image)
        gt, eg = randomPeper_eg(gt, eg)
        return image, gt, eg

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, eg_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    # transforms
    img_transform = transforms.Compose([
        vision.Resize((trainsize, trainsize)),
        vision.ToTensor(),
        vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)])
    gt_transform = transforms.Compose([
        vision.Resize((trainsize, trainsize)),
        vision.ToTensor()])

    eg_transform = transforms.Compose([
        vision.Resize((trainsize, trainsize)),
        vision.ToTensor()])

    dataset = PolypObjDataset(image_root, gt_root, eg_root, trainsize)
    data_loader = GeneratorDataset(source=dataset, column_names=["images", "gts", "egs"])
    data_loader = data_loader.map(img_transform, ["images"])
    data_loader = data_loader.map(gt_transform, ["gts"])
    data_loader = data_loader.map(eg_transform, ["egs"])

    data_loader = data_loader.batch(batch_size=batchsize, num_parallel_workers=num_workers)
    return data_loader


def get_loader_test(image_root, gt_root, testsize):
    testdata = test_dataset(image_root, gt_root, testsize)
    test_loader = GeneratorDataset(source=testdata, column_names=["image", "gt", "name"])
    transform = transforms.Compose([
        vision.Resize((testsize, testsize)),
        vision.ToTensor(),
        vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)])
    test_loader = test_loader.map(transform, "image")
    return test_loader


class test_dataset:
    """load test dataset (batchsize=1)"""

    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        # image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[index])
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
