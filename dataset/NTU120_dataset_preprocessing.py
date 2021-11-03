import os
import math
import torch
import pickle
import argparse
import numpy as np
import os.path as osp

from tqdm import tqdm


class NTU120_preprocess:
    def __init__(self):

        self.skeletons_path = '{}/raw/ntu_120/nturgb+d_skeletons'.format(args.dataset_path)
        self.save_path = '{}/processed/NTU_120'.format(args.dataset_path)
        self.skeletons_ignored_file = '{}/raw/ntu_120/samples_with_missing_skeletons.txt'.format(self.save_path)

        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        self.noise_length_threshold = 11
        self.noise_spread_threshold1 = 0.8
        self.noise_spread_threshold2 = 0.69754
        self.noise_motion_threshold_lo = 0.089925
        self.noise_motion_threshold_hi = 2

        self.downsample_frames = args.downsample_frames
        self.actors = args.actors

        # Step 1/7: Get raw skeleton data
        self.get_raw_skeletons_data()

        # Step 2/7: Denoise the skeleton data
        self.get_raw_denoised_data()

        # Step 3/7: Apply invariant transformation
        self.ntu_transform_skeleton()

        # Step 4/7: Normalize skeleton sequence
        self.normalize_video()

        # Step 5/7: Set skeleton sequence with fixed length
        self.normalize_time()

        # Step 6/7: Select number of actors
        self.select_actors()

        # Step 7/7: Train-Test Split by Cross-Subject and Cross-Setup
        self.data_split()

        print('Preprocessing done.')

    def get_raw_skeletons_data(self):
        """ Open each raw skeleton file and extract infos into a pickle file """

        self.skeletons_name_all = np.array([line.strip('.skeleton') for line in
                                            sorted(os.listdir(self.skeletons_path))], dtype=str)
        self.skeletons_name_ignored = np.searchsorted(self.skeletons_name_all,
                                                      np.loadtxt(self.skeletons_ignored_file, dtype=str))
        self.skeletons_name = np.delete(self.skeletons_name_all,
                                        self.skeletons_name_ignored)

        self.setup = []  # Setup ID 1~3
        self.performer = []  # Subject ID 1~40
        for filename in self.skeletons_name:
            self.setup.append(int(filename[filename.find('S') + 1:filename.find('S') + 4]))
            self.performer.append(int(filename[filename.find('P') + 1:filename.find('P') + 4]))
        self.setup = np.array(self.setup, dtype=np.int)
        self.performer = np.array(self.performer, dtype=np.int)

        self.raw_skeletons_data = []
        self.label = []
        for _, skeleton_name in enumerate(tqdm(self.skeletons_name,
                                               desc='Step 1/7: Get raw skeleton data')):
            bodies_data = self.get_raw_bodies_data(skeleton_name)
            self.raw_skeletons_data.append(bodies_data)
        for data in self.raw_skeletons_data:
            self.label.append(int(data['name'][-3:]))

        with open('{}/setup.pkl'.format(self.save_path), 'wb') as fc:
            pickle.dump(self.setup, fc)
        with open('{}/performer.pkl'.format(self.save_path), 'wb') as fp:
            pickle.dump(self.performer, fp)
        with open('{}/raw_skeletons_data.pkl'.format(self.save_path), 'wb') as fd:
            pickle.dump(self.raw_skeletons_data, fd)
        with open('{}/label.pkl'.format(self.save_path), 'wb') as fl:
            pickle.dump(self.label, fl)

    def get_raw_denoised_data(self):
        """ Denoise raw skeleton data """

        self.raw_denoised_joints = []
        for bodies_data in tqdm(self.raw_skeletons_data,
                                desc='Step 2/7: Denoise the skeleton data'):

            num_bodies = len(bodies_data['data'])
            if num_bodies == 1:  # For samples with only 1 actor
                num_frames = bodies_data['num_frames']
                body_data = list(bodies_data['data'].values())[0]
                joints = self.get_one_actor_points(body_data, num_frames)
            else:  # For more than 1 actor, select the two main ones
                joints = self.get_two_actors_points(bodies_data)

            # Remove missing frames
            joints = self.remove_missing_frames(joints)
            self.raw_denoised_joints.append(joints)

        with open('{}/raw_denoised_joints.pkl'.format(self.save_path), 'wb') as fd:
            pickle.dump(self.raw_denoised_joints, fd)

    def ntu_transform_skeleton(self):
        """ Apply view-invariant transformation """

        self.transform_data = []
        for data in tqdm(self.raw_denoised_joints,
                         desc='Step 3/7: Apply invariant transformation'):
            trans_data = self.invariant_transform(data)
            self.transform_data.append(trans_data)

        with open('{}/transform_data.pkl'.format(self.save_path), 'wb') as fd:
            pickle.dump(self.transform_data, fd)

    def normalize_video(self):
        """ Normalize skeleton coordinates """

        self.normalized_video_data = []
        for video in tqdm(self.transform_data,
                          desc='Step 4/7: Normalize skeleton sequence'):
            max_s = np.amax(video, axis=0)
            min_s = np.amin(video, axis=0)
            max_x = np.max([max_s[i] for i in range(0, video.shape[1], 3)])
            max_y = np.max([max_s[i] for i in range(1, video.shape[1], 3)])
            max_z = np.max([max_s[i] for i in range(2, video.shape[1], 3)])
            min_x = np.min([min_s[i] for i in range(0, video.shape[1], 3)])
            min_y = np.min([min_s[i] for i in range(1, video.shape[1], 3)])
            min_z = np.min([min_s[i] for i in range(2, video.shape[1], 3)])
            norm = np.zeros_like(video)
            for i in range(0, video.shape[1], 3):
                """ Range [-1, 1] """
                norm[:, i] = 2 * (video[:, i] - min_x) / (max_x - min_x) - 1
                norm[:, i + 1] = 2 * (video[:, i + 1] - min_y) / (max_y - min_y) - 1
                norm[:, i + 2] = 2 * (video[:, i + 2] - min_z) / (max_z - min_z) - 1
            self.normalized_video_data.append(norm)

        with open('{}/normalized_video_data.pkl'.format(self.save_path), 'wb') as fd:
            pickle.dump(self.normalized_video_data, fd)

    def normalize_time(self):
        """ Downsample input data into number of target frames """

        self.downsampled_data = []
        for value in tqdm(self.normalized_video_data,
                          desc='Step 5/7: Set skeleton sequence with fixed length'):

            if value.shape[0] > self.downsample_frames:
                new_value = np.zeros((self.downsample_frames, value.shape[1]))
                difference = math.floor(value.shape[0] / self.downsample_frames)
                idx = 0
                for j in range(0, value.shape[0], difference):
                    new_value[idx, :] = value[j, :]
                    idx += 1
                    if idx >= self.downsample_frames:
                        break
                self.downsampled_data.append(new_value)

            elif value.shape[0] == self.downsample_frames:
                self.downsampled_data.append(value)

            elif value.shape[0] < self.downsample_frames:
                new_value = np.zeros((self.downsample_frames, value.shape[1]))
                new_value[:value.shape[0], :] = value
                for i_f, frame in enumerate(new_value):
                    if frame.sum() == 0:
                        if new_value[i_f:].sum() == 0:
                            rest = len(new_value) - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate([new_value[0:i_f]
                                                  for _ in range(num)], 0)[:rest]
                            new_value[i_f:] = pad
                            break
                self.downsampled_data.append(new_value)

        with open('{}/downsampled_data.pkl'.format(self.save_path), 'wb') as fd:
            pickle.dump(self.downsampled_data, fd)

    def select_actors(self):
        """
            Select the number of actors for input data.
                - If 1 actor is selected, prune the second actor
                  for samples which contain two actors.
                - If 2 actors are selected, zero-pad the second actor
                  for samples which contain only one actor.
        """

        if args.actors == 1:
            self.final_data = []
            for sample in tqdm(self.downsampled_data,
                               desc='Step 6/7: Select one actor'):
                if self.actors == 1 and sample.shape[1] == 150:
                    self.final_data.append(sample[:, :75])
                else:
                    self.final_data.append(sample)

            with open('{}/final_data_1_actor.pkl'.format(self.save_path), 'wb') as fd:
                pickle.dump(self.final_data, fd)

        elif args.actors == 2:
            self.final_data = []
            for sample in tqdm(self.downsampled_data,
                               desc='Step 6/7: Select two actors'):
                if self.actors == 2 and sample.shape[1] == 75:
                    half_sample = np.zeros((sample.shape[0], 150))
                    half_sample[:, :75] = sample
                    self.final_data.append(half_sample)
                else:
                    self.final_data.append(sample)

            with open('{}/final_data_2_actor.pkl'.format(self.save_path), 'wb') as fd:
                pickle.dump(self.final_data, fd)

    def data_split(self):
        """ Arrange tensor data """

        # Cross Subject Split
        xsub_split_path = '{}/xsub'.format(self.save_path)
        if not osp.exists(xsub_split_path):
            os.mkdir(xsub_split_path)

        train_subject = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18,
                         19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
                         50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                         80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94,
                         95, 97, 98, 100, 103}

        self.xsub_train_data, self.xsub_test_data = [], []
        self.xsub_train_label, self.xsub_test_label = [], []
        for i in tqdm(range(len(self.final_data)),
                      desc='Step 7/7: Cross-Subject Train-Test Split'):

            # Train set
            if self.performer[i] in train_subject:
                self.xsub_train_data.append(torch.tensor(self.final_data[i]
                                                         .reshape(-1, 25 * self.actors, 3)
                                                         .transpose(2, 1, 0)).float())
                self.xsub_train_label.append(torch.tensor(self.label[i] - 1))

            # Test set
            elif self.performer[i] not in train_subject:
                self.xsub_test_data.append(torch.tensor(self.final_data[i]
                                                        .reshape(-1, 25 * self.actors, 3)
                                                        .transpose(2, 1, 0)).float())
                self.xsub_test_label.append(torch.tensor(self.label[i] - 1))

        torch.save(self.xsub_train_data,
                   '{}/train_data_{}.pt'.format(xsub_split_path, self.actors))
        torch.save(self.xsub_train_label,
                   '{}/train_label_{}.pt'.format(xsub_split_path, self.actors))
        torch.save(self.xsub_test_data,
                   '{}/test_data_{}.pt'.format(xsub_split_path, self.actors))
        torch.save(self.xsub_test_label,
                   '{}/test_label_{}.pt'.format(xsub_split_path, self.actors))

        # Cross Setup Split
        xset_split_path = '{}/xset'.format(self.save_path)
        if not osp.exists(xset_split_path):
            os.mkdir(xset_split_path)

        # Even numbered setups (2, 4, ..., 32) used for training
        train_setup = set(range(2, 33, 2))

        self.xset_train_data, self.xset_test_data = [], []
        self.xset_train_label, self.xset_test_label = [], []
        for i in tqdm(range(len(self.final_data)),
                      desc='Step 7/7: Cross-Setup Train-Test Split'):

            # Train set
            if self.setup[i] in train_setup:
                self.xset_train_data.append(torch.tensor(self.final_data[i]
                                                         .reshape(-1, 25 * self.actors, 3)
                                                         .transpose(2, 1, 0)).float())
                self.xset_train_label.append(torch.tensor(self.label[i] - 1))

            # Test set
            elif self.setup[i] not in train_setup:
                self.xset_test_data.append(torch.tensor(self.final_data[i]
                                                        .reshape(-1, 25 * self.actors, 3)
                                                        .transpose(2, 1, 0)).float())
                self.xset_test_label.append(torch.tensor(self.label[i] - 1))

        torch.save(self.xset_train_data,
                   '{}/train_data_{}.pt'.format(xset_split_path, self.actors))
        torch.save(self.xset_train_label,
                   '{}/train_label_{}.pt'.format(xset_split_path, self.actors))
        torch.save(self.xset_test_data,
                   '{}/test_data_{}.pt'.format(xset_split_path, self.actors))
        torch.save(self.xset_test_label,
                   '{}/test_label_{}.pt'.format(xset_split_path, self.actors))

    def denoising_bodies_data(self, bodies_data):
        bodies_data = bodies_data['data']

        # Step 1: Denoising based on frame length
        bodies_data = self.denoising_by_length(bodies_data)
        if len(bodies_data) == 1:
            return bodies_data.items()

        # Step 2: Denoising based on spread
        bodies_data = self.denoising_by_spread(bodies_data)
        if len(bodies_data) == 1:
            return bodies_data.items()

        # Step 3: Denoising based on motion
        bodies_motion = dict()
        for (bodyID, body_data) in bodies_data.items():
            bodies_motion[bodyID] = body_data['motion']
        bodies_data = self.denoising_by_motion(bodies_data, bodies_motion)

        return bodies_data

    def denoising_by_length(self, bodies_data):
        new_bodies_data = bodies_data.copy()

        for (bodyID, body_data) in new_bodies_data.items():
            length = len(body_data['interval'])
            if length <= self.noise_length_threshold:
                del bodies_data[bodyID]

        return bodies_data

    def denoising_by_motion(self, bodies_data, bodies_motion):
        # Sort bodies based on the motion
        bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)

        # Reserve the body data with the largest motion
        denoised_bodies_data = [(bodies_motion[0][0], bodies_data[bodies_motion[0][0]])]
        for (bodyID, motion) in bodies_motion[1:]:
            if not ((motion < self.noise_motion_threshold_lo)
                    or (motion > self.noise_motion_threshold_hi)):
                denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

        return denoised_bodies_data

    def denoising_by_spread(self, bodies_data):
        new_bodies_data = bodies_data.copy()

        for (bodyID, body_data) in new_bodies_data.items():
            if len(bodies_data) == 1:
                break
            valid_frames = self.get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
            num_frames = len(body_data['interval'])
            num_noise = num_frames - len(valid_frames)
            if num_noise == 0:
                continue
            ratio = num_noise / float(num_frames)
            if ratio >= self.noise_spread_threshold2:
                del bodies_data[bodyID]

        return bodies_data

    @staticmethod
    def get_one_actor_points(body_data, num_frames):
        joints = np.zeros((num_frames, 75), dtype=np.float32)
        start, end = body_data['interval'][0], body_data['interval'][-1]
        joints[start:end + 1] = body_data['joints'].reshape(-1, 75)

        return joints

    def get_raw_bodies_data(self, skeleton_name):
        skeleton_file = '{}/{}.skeleton'.format(self.skeletons_path, skeleton_name)
        assert osp.exists(skeleton_file), 'Error: Skeleton file %s not found' % skeleton_file

        # Read all data from .skeleton file into a list (in string format)
        with open(skeleton_file, 'r') as fr:
            str_data = fr.readlines()
        num_frames = int(str_data[0].strip('\r\n'))
        frames_drop = []
        bodies_data = dict()
        valid_frames = -1
        current_line = 1

        for f in range(num_frames):
            num_bodies = int(str_data[current_line].strip('\r\n'))
            current_line += 1
            if num_bodies == 0:  # No data in this frame, drop it
                frames_drop.append(f)
                continue
            valid_frames += 1
            joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
            for b in range(num_bodies):
                bodyID = str_data[current_line].strip('\r\n').split()[0]
                current_line += 1
                num_joints = int(str_data[current_line].strip('\r\n'))
                current_line += 1
                for j in range(num_joints):
                    temp_str = str_data[current_line].strip('\r\n').split()
                    joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                    current_line += 1
                if bodyID not in bodies_data:
                    body_data = dict()
                    body_data['joints'] = joints[b]
                    body_data['interval'] = [valid_frames]
                else:
                    body_data = bodies_data[bodyID]
                    body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                    pre_frame_idx = body_data['interval'][-1]
                    body_data['interval'].append(pre_frame_idx + 1)
                bodies_data[bodyID] = body_data
        num_frames_drop = len(frames_drop)
        assert num_frames_drop < num_frames,\
            'Error: All frames data (%d) of %s is missing or lost' % (num_frames,
                                                                      skeleton_name)
        if len(bodies_data) > 1:
            for body_data in bodies_data.values():
                body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

        return {'name': skeleton_name,
                'data': bodies_data,
                'num_frames': num_frames - num_frames_drop}

    def get_two_actors_points(self, bodies_data):
        num_frames = bodies_data['num_frames']
        bodies_data = self.denoising_bodies_data(bodies_data)
        bodies_data = list(bodies_data)
        if len(bodies_data) == 1:
            _, body_data = bodies_data[0]
            joints = self.get_one_actor_points(body_data, num_frames)
        else:
            joints = np.zeros((num_frames, 150), dtype=np.float32)
            _, actor1 = bodies_data[0]  # The first actor with largest motion
            _, actor2 = bodies_data[1]  # The second actor with largest motion
            start1, end1 = actor1['interval'][0], actor1['interval'][-1]
            start2, end2 = actor2['interval'][0], actor2['interval'][-1]
            # No overlap with actor 2
            joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
            # No overlap with actor 1
            joints[start2:end2 + 1, 75:] = actor2['joints'].reshape(-1, 75)

        return joints

    def get_valid_frames_by_spread(self, points):
        num_frames = points.shape[0]
        valid_frames = []
        for i in range(num_frames):
            x = points[i, :, 0]
            y = points[i, :, 1]
            if (x.max() - x.min()) <= self.noise_spread_threshold1 * (y.max() - y.min()):
                valid_frames.append(i)

        return valid_frames

    @staticmethod
    def invariant_transform(data):
        data = np.asarray(data)
        transform_raw_data = []
        d = data[0, 0:3]
        i = 0
        while (d == 0).all():
            i += 1
            d = data[i, 0:3]
        if i == 0:
            v1 = data[0, 1 * 3:1 * 3 + 3] - data[0, 0 * 3:0 * 3 + 3]
            v2_ = data[0, 12 * 3:12 * 3 + 3] - data[0, 16 * 3:16 * 3 + 3]
        else:
            v1 = data[i, 1 * 3:1 * 3 + 3] - data[i, 0 * 3:0 * 3 + 3]
            v2_ = data[i, 12 * 3:12 * 3 + 3] - data[i, 16 * 3:16 * 3 + 3]
        while (v2_ == 0).all():
            i += 1
            v2_ = data[i, 12 * 3:12 * 3 + 3] - data[i, 16 * 3:16 * 3 + 3]
        v1 = v1 / np.linalg.norm(v1)
        proj_v2_v1 = np.dot(v1.T, v2_) * v1 / np.linalg.norm(v1)
        v2 = v2_ - np.squeeze(proj_v2_v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v2, v1) / np.linalg.norm(np.cross(v2, v1))
        v1 = np.reshape(v1, (3, 1))
        v2 = np.reshape(v2, (3, 1))
        v3 = np.reshape(v3, (3, 1))
        R = np.hstack([v2, v3, v1])
        for i in range(data.shape[0]):
            xyzs = []
            for j in range(int(data.shape[1] / 3)):
                xyz = np.squeeze(
                    np.matmul(
                        np.linalg.inv(R),
                        np.reshape(data[i][j * 3:j * 3 + 3] - d, (3, 1))))
                xyzs.append(xyz)
            xyzs = np.reshape(np.asarray(xyzs), (-1, data.shape[1]))
            transform_raw_data.append(xyzs)
        transform_raw_data = np.squeeze(np.asarray(transform_raw_data))

        return transform_raw_data

    @staticmethod
    def remove_missing_frames(joints):
        # Find valid frame indices that the data is not missing or lost
        # For two-subjects action, this means both data of actor1 and actor2 is missing
        valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # zero-based index
        missing_indices = np.where(joints.sum(axis=1) == 0)[0]
        num_missing = len(missing_indices)
        if num_missing > 0:
            joints = joints[valid_indices]

        return joints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--downsample_frames', type=int, default=100,
                        help='select the number of timeframes to preprocess')
    parser.add_argument('--actors', type=int, choices=[1, 2], default=1,
                        help='select the number of actors to preprocess')
    args = parser.parse_args()
    NTU120_preprocess()
