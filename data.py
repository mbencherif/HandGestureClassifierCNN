import os
import glob
import random

import tensorflow as tf
import numpy as np

import cv2


class DataPipeline:
    def __init__(self, session, image_size=(144, 256), home_path="/Users/Yuhan"):
        self.sess = session
        self.image_size = image_size if type(image_size) is list else list(image_size)

        self.home = home_path
        self.data_set_base = os.path.join(home_path, "DataSets/data")

        self.train_labels_file = os.path.join(home_path, "DataSets/train.csv")
        self.validation_labels_file = os.path.join(home_path, "DataSets/validation.csv")
        self.test_labels_file = os.path.join(home_path, "DataSets/test.csv")
        self.labels_file = os.path.join(home_path, "DataSets/labels.csv")

        self.feature_labels = list()
        with open(self.train_labels_file, 'r') as train_csv:
            for line in train_csv:
                folder_name, gesture = line.split(";", maxsplit=2)
                gesture = gesture.rstrip()
                self.feature_labels.append((os.path.join(self.data_set_base, folder_name), gesture))

        self.validation_feature_labels = list()
        with open(self.validation_labels_file, 'r') as val_csv:
            for line in val_csv:
                folder_name, gesture = line.split(";", maxsplit=2)
                gesture = gesture.rstrip()
                self.validation_feature_labels.append((os.path.join(self.data_set_base, folder_name), gesture))

        self.labels = list()
        with open(self.labels_file, 'r') as label_csv:
            for line in label_csv:
                self.labels.append(line.rstrip())

        def imgs_flows_from_folder(folder):
            image_files = glob.glob(os.path.join(folder, "*.jpg"))
            image_files = sorted(image_files)
            images = list()
            for image_file in image_files:
                image = cv2.imread(image_file)
                images.append(image)
            old_size = images[0].shape[:2]
            resize_height = old_size[0] / old_size[1] > self.image_size[0] / self.image_size[1]
            if resize_height:
                new_size = (self.image_size[0], int(self.image_size[0] / old_size[0] * old_size[1]))
                d_pad = self.image_size[1] - new_size[1]
                padding_l = d_pad // 2
                padding_r = d_pad - padding_l
            else:
                new_size = (int(self.image_size[1] / old_size[1] * old_size[0]), self.image_size[1])
                d_pad = self.image_size[0] - new_size[0]
                padding_l = d_pad // 2
                padding_r = d_pad - padding_l
            for i in range(len(images)):
                image = images[i]
                images[i] = cv2.resize(image, dsize=(new_size[1], new_size[0]))
            images = np.array(images, dtype=np.float32)

            images_gray = np.zeros(images.shape[:-1])
            for i in range(images.shape[0]):
                images_gray[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            motion_flow = list()
            for i in range(images.shape[0]):
                n = min(i + 1, images.shape[0] - 1)
                im1 = images_gray[i]
                im2 = images_gray[n]
                # prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
                new_flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5, levels=4,
                                                        winsize=25, iterations=3, poly_n=7, poly_sigma=1.2,
                                                        flags=0)
                new_flow[:, :, 0] -= new_flow[:, :, 0].mean()
                new_flow[:, :, 1] -= new_flow[:, :, 1].mean()
                mag = np.sum(new_flow * new_flow, axis=2) ** 0.5
                new_flow /= mag.std()
                motion_flow.append(new_flow)
            motion_flow = np.array(motion_flow)

            if resize_height:
                images = np.pad(images, [(0, 0), (0, 0), (padding_l, padding_r), (0, 0)], 'constant')
                motion_flow = np.pad(motion_flow, [(0, 0), (0, 0), (padding_l, padding_r), (0, 0)], 'constant')
            else:
                images = np.pad(images, [(0, 0), (padding_l, padding_r), (0, 0), (0, 0)], 'constant')
                motion_flow = np.pad(motion_flow, [(0, 0), (padding_l, padding_r), (0, 0), (0, 0)], 'constant')

            images /= 255.0

            # make sure we have only 35 frames
            num_frames = images.shape[0]
            frames_to_remove = num_frames - 35
            if frames_to_remove < 0:
                images_pad = np.zeros([-frames_to_remove] + list(images.shape[1:]), dtype=np.float32)
                motion_flow_pad = np.zeros([-frames_to_remove] + list(motion_flow.shape[1:]), dtype=np.float32)
                images = np.concatenate([images_pad, images], axis=0)
                motion_flow = np.concatenate([motion_flow_pad, motion_flow], axis=0)
            elif frames_to_remove > 0:
                front_remove = 4 * frames_to_remove // 5
                back_remove = frames_to_remove - front_remove
                images = images[front_remove:-back_remove, :, :, :]
                motion_flow = motion_flow[front_remove:-back_remove, :, :, :]

            return {'images': images, 'flows': motion_flow}

        def one_hot_label(label):
            i = self.labels.index(label)
            one_hot = np.zeros(len(self.labels), dtype=np.float32)
            one_hot[i] = 1.0
            return one_hot

        def process_feature_label(feature_label):
            feature = feature_label[0]
            label = feature_label[1]
            return imgs_flows_from_folder(feature), one_hot_label(label)

        def generator():
            for feature in self.feature_labels:
                yield process_feature_label(feature_label=feature)

        def val_generator():
            while True:
                feature = random.choice(self.validation_feature_labels)
                yield process_feature_label(feature_label=feature)

        self.generator = generator
        self.val_generator = val_generator

    def label_from_one_hot(self, one_hot_vector):
        index = np.argmax(one_hot_vector)
        return self.labels[index]

    def num_classes(self):
        return len(self.labels)
