import tensorflow as tf
import torch
import torchvision
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image


class LFWFolder(data.Dataset):
    """LFW folder"""

    def __init__(self, root, split='train', transform=None, max_per_class=-1,
                 path_to_female_names="./female_names_lfw.txt"):
        """Initializes image paths and preprocessing module."""
        img_dir = os.path.join(root, split)
        self.image_names = os.listdir(img_dir)
        self.image_paths = list(map(lambda x: os.path.join(img_dir, x), self.image_names))

        with open(path_to_female_names, 'r') as f:
            self.female_names = set(map(lambda x: x.strip(), f.readlines()))

        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        image_label = 0 if image_name in self.female_names else 1
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image, image_label

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)


def get_lfw_data(path_to_namedir, is_train=True):
    """
    """
    import pickle, os
    if is_train:
        datafile = os.path.join(path_to_namedir, "LFW_train_info.pkl")
    else:
        datafile = os.path.join(path_to_namedir, "LFW_test_info.pkl")

    with open(datafile, 'rb') as f:
        data = pickle.load(f)

    names = sorted(list(data.keys()))
    at_arr = np.asarray([data[n]['at'] for n in names])
    true_label_arr = np.asarray([data[n]['true'] for n in names])
    pred_label_arr = np.asarray([data[n]['pred'] for n in names])
    return {'name': names, 'at': at_arr,
            'true': true_label_arr, 'pred': pred_label_arr}


def load_lfw(path_to_data='../data/lfw/lfw_data',
             path_to_name='../data/lfw/lfw_np',
             path_to_female_names='../data/lfw/lfw_np/female_names_lfw.txt'):
    names_in_train = sorted(get_lfw_data(
        path_to_name, is_train=True)['name'])
    names_in_test = sorted(get_lfw_data(
        path_to_name, is_train=False)['name'])

    trainloader = get_LFW_loader(image_path=path_to_data,
                                 split='train', batch_size=1,
                                 path_to_female_names=path_to_female_names)
    testloader = get_LFW_loader(image_path=path_to_data,
                                split='test', batch_size=1,
                                path_to_female_names=path_to_female_names)

    train_vs = {}
    for image_names, images, image_labels in trainloader:
        train_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
    #
    train_data = [[], []]
    sorted_train_vs = [train_vs[n] for n in names_in_train]
    train_data[0] = np.array([np.moveaxis(vs[0], [0], [-1]) for vs in sorted_train_vs])
    train_data[1] = np.array([vs[1] for vs in sorted_train_vs])

    test_data = [[], []]
    test_vs = {}
    for image_names, images, image_labels in testloader:
        test_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
    #
    sorted_test_vs = [test_vs[n] for n in names_in_test]
    test_data[0] = np.array([np.moveaxis(vs[0], [0], [-1]) for vs in sorted_test_vs])
    test_data[1] = np.array([vs[1] for vs in sorted_test_vs])
    return (train_data, test_data)


def get_LFW_loader(image_path, split, batch_size, num_workers=0, path_to_female_names="./female_names_lfw.txt"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    dataset = LFWFolder(image_path, split, transform=transform, path_to_female_names=path_to_female_names)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # True, # b/c data index is used later
        num_workers=num_workers
    )
    return data_loader


def load_adv(data_file='../data/adv/data/mnist_adv_csv_fgsm_epsilon0.1.txt',
             label_file='../data/adv/data/mnist_adv_val_label.txt'):
    with open(data_file, 'r') as f1, open(label_file, 'r') as f2:
        data_lines = f1.readlines()
        label_lines = f2.readlines()
    x = []
    y = []
    for data_line, label_line in zip(data_lines, label_lines):
        x.append(np.array(data_line.strip().split(','), dtype=float).reshape((28,28,1)))
        y.append(np.array(label_line.strip(), dtype=int))
    return np.array(x), np.array(y)


def load_data(data_name, path_to_data):
    """
    return:tuple(list(train_x,train_y),list(test_x,test_y))
    """
    if data_name=='GTSRB':
        data_name='gtsrb'
    if data_name in ['fashion_mnist', 'cifar10']:
        if data_name == 'fashion_mnist':
            trainset = torchvision.datasets.FashionMNIST(root=path_to_data, train=True,
                                                         download=True, transform=transforms.ToTensor())
            testset = torchvision.datasets.FashionMNIST(root=path_to_data, train=False,
                                                        download=True, transform=transforms.ToTensor())
        else:
            path_to_data = path_to_data + "/cifar10"
            trainset = torchvision.datasets.CIFAR10(root=path_to_data, train=True,
                                                    download=True, transform=transforms.ToTensor())
            testset = torchvision.datasets.CIFAR10(root=path_to_data, train=False,
                                                   download=True, transform=transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

        train_data = [[], []]
        test_data = [[], []]

        for data in trainloader:
            image, labels = data
            if data_name == "fashion_mnist":
                train_data[0].append(image.numpy()[0].reshape(1, -1))  # c*(h*w)
            else:
                train_data[0].append(image.numpy()[0])  # c×h×w
            train_data[1].append(labels.item())

        train_data[0] = np.asarray(train_data[0])
        train_data[1] = np.asarray(train_data[1])

        for data in testloader:
            image, labels = data
            if data_name == "fashion_mnist":
                test_data[0].append(image.numpy()[0].reshape(1, -1))
            else:
                test_data[0].append(image.numpy()[0])
            test_data[1].append(labels.item())

        test_data[0] = np.asarray(test_data[0])
        test_data[1] = np.asarray(test_data[1])

    elif data_name in ["gtsrb"]:
        import pickle
        # train
        with open(os.path.join(path_to_data, "gtsrb/train_data.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        if isinstance(train_data, dict):
            train_data = [train_data['data'], train_data['label']]

        # test
        with open(os.path.join(path_to_data, "gtsrb/test_data.pkl"), 'rb') as f:
            test_data = pickle.load(f)
        if isinstance(test_data, dict):
            test_data = [test_data['data'], test_data['label']]

    return (train_data, test_data)


def format_label(labels, num_label):
    """format label which has a integer label to flag typee.g., [3, 5] -> [[0,0,1,0,0],[0,0,0,0,1]]"""
    num_data = len(labels)
    from collections.abc import Iterable

    new_labels = np.zeros([num_data, num_label])
    for i, v in enumerate(labels):
        if isinstance(v, Iterable):
            new_labels[i, v[0]] = 1
        else:
            new_labels[i, v] = 1

    return new_labels


def split_into_wrong_and_correct(correct_predictions):
    """
    Spilt wrong and correct classification result
    Ret (dict):
        ret = {'wrong':indices_to_wrong(list),
            'correct':indices_to_correct(list)}
    """
    indices = {'wrong': [], 'correct': []}
    indices_to_wrong, = np.where(correct_predictions == False)
    indices_to_correct, = np.where(correct_predictions == True)
    indices['correct'] = list(indices_to_correct)
    indices['wrong'] = list(indices_to_wrong)
    return indices


def interpolate_inputs(base_line,
                       input,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    base_line_x = tf.expand_dims(base_line, axis=0)
    input_x = tf.expand_dims(input, axis=0)
    delta = input_x - base_line_x
    inputs = base_line_x + alphas_x * delta
    return inputs[1:]
