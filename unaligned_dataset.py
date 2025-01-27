import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import json
import torch

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        # Load fourcorner from JSON file
        with open('hanzi_four_corner_onehot.json', 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.label_dict = {label['hanzi']: label['one_hot'] for label in self.labels}

        # Randomize label order
        self.train_dataA = []
        self.train_dataB = []
        random.shuffle(self.A_paths)
        random.shuffle(self.B_paths)

        for path in self.A_paths:
            label = self.get_label_from_filename(path)
            self.train_dataA.append([path, label])

        for path in self.B_paths:
            label = self.get_label_from_filename(path)
            self.train_dataB.append([path, label])

    def get_label_from_filename(self, path):
        filename = os.path.basename(path)
        hanzi_code = filename.split('_')[1].split('.')[0]
        hanzi = chr(int(hanzi_code, 16))
        return self.label_dict.get(hanzi, [0] * 50)  # return zero vector if no label found

    def __getitem__(self, index):
        A_path, A_label = self.train_dataA[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path, B_label = self.train_dataB[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # Convert labels to tensors
        A_label = torch.tensor(A_label, dtype=torch.float32)  # [50]
        B_label = torch.tensor(B_label, dtype=torch.float32)  # [50]

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label, 'B_label': B_label}

    def __len__(self):
        return max(self.A_size, self.B_size)
