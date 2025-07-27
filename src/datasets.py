# -*- coding: utf-8 -*-

import copy
import torch
import random

from torch.utils.data import Dataset

from utils import neg_sample, nCr
from data_augmentation_time import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate,Downsample, TimeWarping


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, time_seq, not_aug_users=None, data_type='train',
                 similarity_model_type='offline', total_train_users=0):
        self.args = args
        self.user_seq = user_seq
        self.time_seq = time_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.not_aug_users = not_aug_users

        self.total_train_users = total_train_users
        self.model_warm_up_train_users = args.model_warm_up_epochs * len(user_seq)

        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        if similarity_model_type == 'offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type == 'online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type == 'hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(args.crop_mode, args.crop_rate),
                              'mask': Mask(self.similarity_model,args.mask_mode, args.mask_rate),
                              'reorder': Reorder(args.reorder_mode, args.reorder_rate),
                              'substitute': Substitute(self.similarity_model, args.substitute_mode,
                                                       args.substitute_rate),
                              'insert': Insert(self.similarity_model, args.insert_rate, args.max_insert_num_per_pos),
                              'Downsample': Downsample(args.downsample_rate),
                              'random': Random(args, self.similarity_model),
                              'combinatorial_enumerate': CombinatorialEnumerate(args, self.similarity_model),
                              'TimeWarping':  TimeWarping(args.base_warping_factor, args.range_width)
                              }

        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views
    def _process_sequence_label_signal(self, seq_label_signal):
        seq_class_label = torch.tensor(seq_label_signal, dtype=torch.long)
        return seq_class_label

    def _one_pair_data_augmentation(self, input_ids, input_times, not_aug=False):
        """
            同时增强项目序列和时间序列
            """
        augmented_seqs = []
        augmented_times = []

        for i in range(2):
            if not_aug:
                if self.args.not_aug_data_mode == 'zero':
                    augmented_input_ids = [0] * self.max_len
                    augmented_input_times = [0] * self.max_len
                else:
                    pad_id_len = self.max_len - len(input_ids)
                    augmented_input_ids = [0] * pad_id_len + input_ids
                    pad_time_len = self.max_len - len(input_times)
                    augmented_input_times = [0] * pad_time_len + input_times
                    augmented_input_ids = augmented_input_ids[-self.max_len:]
                    augmented_input_times = augmented_input_times[-self.max_len:]
            else:
                # 同时增强项目序列和时间序列
                augmented_input_ids, augmented_input_times = self.base_transform(
                    input_ids, input_times
                )

                pad_id_len = self.max_len - len(augmented_input_ids)
                augmented_input_ids = [0] * pad_id_len + augmented_input_ids
                pad_time_len = self.max_len - len(augmented_input_times)
                augmented_input_times = [0] * pad_time_len + augmented_input_times
                augmented_input_ids = augmented_input_ids[-self.max_len:]
                augmented_input_times = augmented_input_times[-self.max_len:]

            augmented_seqs.append(torch.tensor(augmented_input_ids, dtype=torch.long))
            augmented_times.append(torch.tensor(augmented_input_times, dtype=torch.long))

        return augmented_seqs, augmented_times

    def _data_sample_rec_task(self, user_id, items, input_ids, input_times, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        copied_input_times = copy.deepcopy(input_times)  # 添加时间序列拷贝

        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        # 对项目序列和时间序列进行相同的填充
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_times = [0] * pad_len + copied_input_times  # 时间序列用0填充
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        copied_input_times = copied_input_times[-self.max_len:]  # 截断
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(copied_input_times) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(copied_input_times, dtype=torch.long),  # 添加时间序列
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        times = self.time_seq[index]  # 原始时间序列
        # 添加时间序列处理
        if self.data_type == "train":
            input_ids = items[:-3]
            input_times = times[:-3]  # 对应input_ids的时间序列
            target_pos = items[1:-2]
            seq_label_signal = items[-2]
            answer = [0]
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            input_times = times[:-2]  # 对应input_ids的时间序列
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            input_times = times[:-1]  # 对应input_ids的时间序列
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
            # 生成负样本 - 新增代码
            #sample_negs = []
            #seq_set = set(items_with_noise)
            #for _ in range(99):  # 假设需要99个负样本
            #    sample_negs.append(neg_sample(seq_set, self.args.item_size))

        if self.data_type == "train":
            # 传入时间序列
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, input_times, target_pos, answer)
            cf_tensors_list = []
            cf_times_list = []  # 新增时间序列列表
            not_aug = False
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentation_pairs = nCr(self.n_views, 2)

            for i in range(total_augmentation_pairs):
                augmented_seqs, augmented_times = self._one_pair_data_augmentation(input_ids, input_times, not_aug)
                cf_tensors_list.append(augmented_seqs)
                cf_times_list.append(augmented_times)  # 保存时间序列
            seq_class_label = self._process_sequence_label_signal(seq_label_signal)
            return cur_rec_tensors, cf_tensors_list, cf_times_list, seq_class_label
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, input_times, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, input_times, target_pos, answer)
            return cur_rec_tensors
            # 返回包含负样本的元组
            #return (*cur_rec_tensors, torch.tensor(sample_negs, dtype=torch.long))

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)
