# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import os
import timm
import cv2


from torch.utils.data import dataloader, Dataset
from PIL import Image


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def create_thumb_images(full_folder, thumb_folder, suffix='thumb', height=100, del_former_thumb=False):
    if del_former_thumb:
        del_file(thumb_folder)
    cnt = 0
    print('Creating thumb images')
    for image_file in tqdm(os.listdir(full_folder)):
        # print(image_file)
        if 'DS_Store' in image_file:
            continue
        image = cv2.imread(full_folder + image_file)
        cnt += 1
        # print(str(cnt) + "." + full_folder + image_file)
        height_src, width_src, _ = image.shape
        #print('width: {}, height: {}'.format(width_src, height_src))

        width = (height*1.0 / height_src) * width_src
        # print(' Thumb width: {}, height: {}'.format(width, height))
        resized_image = cv2.resize(image, (int(width), int(height)))

        image_name, image_extension = os.path.splitext(image_file)
        cv2.imwrite(thumb_folder + image_name + suffix + image_extension, resized_image)
    print('Creating thumb images finished')


def get_file_list(file_path_list, sort=True):
    """
    Get list of file paths in one folder.
    :param file_path: A folder path or path list.
    :return: file list: File path list of
    """
    import random
    if isinstance(file_path_list, str):
        file_path_list = [file_path_list]
    file_lists = []
    for file_path in file_path_list:    
        assert os.path.isdir(file_path)
        file_list_ = os.listdir(file_path)
        file_list = []
        for file_name in file_list_:
            if 'DS_Store' in file_name:
                continue
            file_list.append(file_name)
        if sort:
            file_list.sort()
        else:
            random.shuffle(file_list)
        file_list = [file_path + file for file in file_list]
        file_lists.append(file_list)
    if len(file_lists) == 1:
        file_lists = file_lists[0]
    return file_lists


class Gallery(Dataset):
    """
    Images in database.
    """

    def __init__(self, image_paths, transform=None):
        super(Gallery, self).__init__()

        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path

    def __len__(self):
        return len(self.image_paths)


def load_data(data_path, batch_size=1, shuffle=False, transform='default'):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if transform == 'default' else transform

    image_path_list = get_file_list(data_path)

    gallery_data = Gallery(image_paths=image_path_list,
                           transform=data_transform,
                           )

    data_loader = dataloader.DataLoader(dataset=gallery_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=0,
                                        )
    return data_loader


def extract_feature(model, dataloaders, use_gpu=True):
    features = torch.FloatTensor()
    path_list = []

    use_gpu = use_gpu and torch.cuda.is_available()
    print("Extracting images features ...")
    for img, path in tqdm(dataloaders):
        # print(path)
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)
        outputs = model(input_img)
        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        path_list += list(path)
    print('Extracting images features finished')
    return features, path_list


def extract_feature_query(model, img, use_gpu=True):
    c, h, w = img.size()
    img = img.view(-1,c,h,w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff,p=2,dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


def load_query_image(query_path):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image


def load_model(model_name, use_gpu=True):
    """

    :param check_point: Pretrained model path.
    :return:
    """
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
    if model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    '''
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model))
    '''

    # remove the final fc layer
    if model_name == 'alexnet':
        model.classifier = nn.Sequential()
    if model_name == 'resnet':
        model.fc = nn.Sequential()
    if model_name == 'vit':
        model.head = nn.Sequential()
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model


# sort the images
def sort_img(qf, gf, method):
    if method == 'e':
        score = -torch.sqrt(((gf-qf)**2).sum(1))
    if method == 'cos':
        score = (gf*qf).sum(1)
    if method == 'kl':
        score = torch.zeros(gf.size(0))
        for i in range(gf.size(0)):
            score[i] = -F.kl_div(qf.softmax(dim=-1).log(), gf[i].softmax(dim=-1), reduction='sum')
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    import numpy as np
    s = np.around(s, 3)
    return s, index


if __name__ == '__main__':

    # Prepare data.
    data_loader = load_data(data_path='./test_pytorch/gallery/images/',
                            batch_size=2,
                            shuffle=False,
                            transform='default',
                            )

    # Prepare model.
    model = load_model(pretrained_model='./model/ft_ResNet50/net_best.pth', use_gpu=True)

    # Extract database features.
    gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

    # Query.
    query_image = load_query_image('./test_pytorch/query/query.jpg')

    # Extract query features.
    query_feature = extract_feature_query(model=model, img=query_image)

    # Sort.
    similarity, index = sort_img(query_feature, gallery_feature)

    sorted_paths = [image_paths[i] for i in index]
    print(sorted_paths)

