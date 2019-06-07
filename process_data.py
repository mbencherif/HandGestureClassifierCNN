import os
import glob
from multiprocessing import Pool
import numpy as np
import cv2
import sys

image_size = None


def resize_folder(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg"))
    image_files = sorted(image_files)
    images = list()
    for image_file in image_files:
        image = cv2.imread(image_file)
        images.append(image)
    old_size = images[0].shape[:2]
    if old_size == image_size:
        return
    resize_height = old_size[0] / old_size[1] > image_size[0] / image_size[1]
    if resize_height:
        new_size = (image_size[0], int(image_size[0] / old_size[0] * old_size[1]))
        d_pad = image_size[1] - new_size[1]
        padding_l = d_pad // 2
        padding_r = d_pad - padding_l
    else:
        new_size = (int(image_size[1] / old_size[1] * old_size[0]), image_size[1])
        d_pad = image_size[0] - new_size[0]
        padding_l = d_pad // 2
        padding_r = d_pad - padding_l
    for i in range(len(images)):
        image = images[i]
        images[i] = cv2.resize(image, dsize=(new_size[1], new_size[0]))

    if resize_height:
        for name, image in zip(image_files, images):
            save = np.pad(image, [(0, 0), (padding_l, padding_r), (0, 0)], 'constant')
            cv2.imwrite(name, save)
    else:
        for name, image in zip(image_files, images):
            save = np.pad(image, [(padding_l, padding_r), (0, 0), (0, 0)], 'constant')
            cv2.imwrite(name, save)


def resize_images(home_path="/home/yliu102199"):
    dataset_base = os.path.join(home_path, "DataSets/data")

    train_labels_file = os.path.join(home_path, "DataSets/train.csv")
    validation_labels_file = os.path.join(home_path, "DataSets/validation.csv")
    # test_labels_file = os.path.join(home_path, "DataSets/test.csv")

    feature_labels = list()
    with open(train_labels_file, 'r') as train_csv:
        for line in train_csv:
            folder_name, _ = line.split(";", maxsplit=2)
            feature_labels.append(os.path.join(dataset_base, folder_name))

    validation_feature_labels = list()
    with open(validation_labels_file, 'r') as val_csv:
        for line in val_csv:
            folder_name, _ = line.split(";", maxsplit=2)
            validation_feature_labels.append(os.path.join(dataset_base, folder_name))

    # test_feature_labels = list()
    # with open(test_labels_file, 'r') as test_csv:
    #     for line in test_csv:
    #         folder_name, _ = line.split(";", maxsplit=2)
    #         test_feature_labels.append(os.path.join(dataset_base, folder_name))

    with Pool(16) as pool:
        print("Resizing training images")
        # for i, _ in enumerate(pool.map(resize_folder, feature_labels, 16)):
        for i, _ in enumerate(map(resize_folder, feature_labels)):
            sys.stdout.write('\rDone... {0:%}'.format(i / len(feature_labels)))
        sys.stdout.write('\n')
        sys.stdout.flush()

        print("Resizing validation images")
        # for i, _ in enumerate(pool.map(resize_folder, validation_feature_labels, 16)):
        for i, _ in enumerate(map(resize_folder, validation_feature_labels)):
            sys.stdout.write('\rDone... {0:%}'.format(i / len(validation_feature_labels)))
        sys.stdout.write('\n')
        sys.stdout.flush()

        # print("Resizing test images")
        # for i, _ in enumerate(pool.imap_unordered(resize_folder, test_feature_labels)):
        #     sys.stdout.write('\rDone... {0:%}'.format(i / len(test_feature_labels)))
        # sys.stdout.write('\n')
        # sys.stdout.flush()

        print("Done")


if __name__ == '__main__':
    image_size = [144, 256]
    # resize_images(home_path='./SampleDataset')
    resize_images(home_path='/home/yliu102199')
