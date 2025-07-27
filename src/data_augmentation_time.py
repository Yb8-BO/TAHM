# -*- coding: utf-8 -*-

import copy
import random
import itertools
import numpy as np

def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []

    if length == 1:
        return 0

    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    avg_diff = total / len(diffs)

    total = 0
    for diff in diffs:
        total = total + (diff - avg_diff) ** 2
    result = total / len(diffs)

    return result

class CombinatorialEnumerate(object):
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs.
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.

    For example, M = 3, the argumentation methods to be called are in following order:
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """

    def __init__(self, args, similarity_model):
        self.data_augmentation_methods = [Crop(args.crop_mode, args.crop_rate),
                                          Mask(similarity_model,args.mask_mode,args.insert_rate, args.mask_rate,args.max_insert_num_per_pos),
                                          Reorder(args.reorder_mode, args.reorder_rate),
                                          Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                 args.max_insert_num_per_pos),
                                          Substitute(similarity_model, args.substitute_mode,
                                                     args.substitute_rate),
                                          Downsample(args.downsample_rate),
                                          TimeWarping(args.base_warping_factor,args.range_width)]
        self.n_views = args.n_views
        self.augmentation_idx_list = self.__get_augmentation_idx_order()  # length of the list == C(M, 2)
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0

    def __get_augmentation_idx_order(self):
        augmentation_idx_list = []
        for (view_1, view_2) in itertools.combinations([i for i in range(self.n_views)], 2):
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list

    def __call__(self, item_sequence, time_sequence):
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx]
        augment_method = self.data_augmentation_methods[augmentation_idx]
        self.cur_augmentation_idx_of_idx += 1  # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx = self.cur_augmentation_idx_of_idx % self.total_augmentation_samples
        return augment_method(item_sequence, time_sequence)

class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, args, similarity_model):
        self.short_seq_data_aug_methods = None
        self.augment_threshold = args.augment_threshold
        self.augment_type_for_short = args.augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(similarity_model,args.mask_mode,args.insert_rate, args.mask_rate,args.max_insert_num_per_pos),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate),
                                              Downsample(args.downsample_rate),
                                              TimeWarping(args.base_warping_factor, args.range_width)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            print("short sequence augment type:", self.augment_type_for_short)
            self.short_seq_data_aug_methods = []
            if 'S' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Substitute(similarity_model, args.substitute_mode, args.substitute_rate))
            if 'I' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Insert(similarity_model, args.insert_mode, args.insert_rate, args.max_insert_num_per_pos))
            if 'M' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Mask(similarity_model,args.mask_mode,args.insert_rate, args.mask_rate,args.max_insert_num_per_pos), )
            if 'R' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Reorder(args.reorder_mode, args.reorder_rate))
            if 'C' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Crop(args.crop_mode, args.crop_rate))
            if 'D' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Downsample(args.downsample_rate))
            if 'T' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(TimeWarping(args.base_warping_factor, args.range_width))
            if len(self.augment_type_for_short) == 7:
                print("all aug set for short sequences")
            self.long_seq_data_aug_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(similarity_model,args.mask_mode,args.insert_rate, args.mask_rate,args.max_insert_num_per_pos),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate),
                                              Downsample(args.downsample_rate),
                                              TimeWarping(args.base_warping_factor, args.range_width)]
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else:
            raise ValueError("Invalid data type.")

    def __call__(self, item_sequence, time_sequence):
        if self.augment_threshold == -1:
            # randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            return augment_method(item_sequence, time_sequence)
        elif self.augment_threshold > 0:
            seq_len = len(item_sequence)
            if seq_len > self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods) - 1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)
            elif seq_len <= self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods) - 1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)

def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]

class Insert(object):
    """
    Insert similar items every time call.
    Priority is given to places with large time intervals.
    maximum: Insert at larger time intervals
    minimum: Insert at smaller time intervals
    """

    def __init__(self, item_similarity_model, mode, insert_rate=0.4, max_insert_num_per_pos=1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.mode = mode
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, item_sequence, time_sequence):
        # 创建副本以避免修改原始序列
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)

        # 计算时间差异
        time_diffs = []
        length = len(copied_time_sequence)
        for i in range(length - 1):
            diff = abs(copied_time_sequence[i + 1] - copied_time_sequence[i])
            time_diffs.append(diff)

        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            diff_sorted = np.argsort(time_diffs)[::-1]  # 从大到小排序
        elif self.mode == 'minimum':
            diff_sorted = np.argsort(time_diffs)  # 从小到大排序
        diff_sorted = diff_sorted.tolist()

        insert_idx = diff_sorted[:insert_nums]

        # 确保索引按顺序处理（从后往前插入，避免索引变化）
        insert_idx.sort()
        insert_idx=insert_idx[::-1]

        # 在指定位置插入新项目和时间
        for index in insert_idx:
            # 计算插入位置（在index之后）
            insert_position = index + 1

            # 生成要插入的项目
            top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], top_k=top_k, with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], top_k=top_k, with_score=True)
                insert_items = _ensmeble_sim_models(top_k_one, top_k_two)
            else:
                insert_items = self.item_similarity_model.most_similar(copied_sequence[index], top_k=top_k)

            # 计算新时间（取前一个时间）
            base_time = copied_time_sequence[index]
            insert_times = [base_time] * len(insert_items)

            # 插入项目和时间
            copied_sequence = copied_sequence[:insert_position] + insert_items + copied_sequence[insert_position:]
            copied_time_sequence = copied_time_sequence[:insert_position] + insert_times + copied_time_sequence[
                                                                                           insert_position:]

        return copied_sequence, copied_time_sequence

class Substitute(object):
    """
    Substitute with similar items
    maximum: Substitute items with larger time interval
    minimum: Substitute items with smaller time interval
    """

    def __init__(self, item_similarity_model, mode, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # 创建副本以避免修改原始序列
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        if len(copied_sequence) <= 1:
            return copied_sequence, copied_time_sequence

        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)

        # 计算时间差异
        time_diffs = []
        length = len(copied_time_sequence)
        for i in range(length - 1):
            diff = abs(copied_time_sequence[i + 1] - copied_time_sequence[i])
            time_diffs.append(diff)

        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            diff_sorted = np.argsort(time_diffs)[::-1]  # 从大到小排序
        elif self.mode == 'minimum':
            diff_sorted = np.argsort(time_diffs)  # 从小到大排序
        diff_sorted = diff_sorted.tolist()

        substitute_idx = diff_sorted[:substitute_nums]

        # 替换项目
        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
                copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])[0]

        return copied_sequence, copied_time_sequence

class Crop(object):
    """
    maximum: Crop subsequences with the maximum time interval variance
    minimum: Crop subsequences with the minimum time interval variance
    """

    # 通过计算时间序列中不同子序列的时间差异方差，找到具有最大或最小方差的子序列，并从原始序列中提取并返回这个子序列。
    def __init__(self, mode, tao=0.2):
        self.tao = tao
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        sub_seq_length = int(self.tao * len(copied_sequence))
        if sub_seq_length < 1:
            return copied_sequence, copied_time_sequence

        # 计算每个可能子序列的时间差异方差
        cropped_vars = []
        crop_index = []
        for i in range(len(copied_sequence) - sub_seq_length + 1):
            temp_time_sequence = copied_time_sequence[i:i + sub_seq_length]
            temp_var = get_var(temp_time_sequence)
            cropped_vars.append(temp_var)
            crop_index.append(i)

        # 根据模式选择子序列
        if self.mode == 'maximum':
            selected_index = crop_index[np.argmax(cropped_vars)]
        elif self.mode == 'minimum':
            selected_index = crop_index[np.argmin(cropped_vars)]
        else:
            selected_index = random.choice(crop_index)

        # 裁剪序列
        cropped_sequence = copied_sequence[selected_index:selected_index + sub_seq_length]
        cropped_time_sequence = copied_time_sequence[selected_index:selected_index + sub_seq_length]

        return cropped_sequence, cropped_time_sequence

class Reorder(object):
    """
    Randomly shuffle a continuous sub-sequence
    maximum: Reorder subsequences with the maximum time interval variance
    minimum: Reorder subsequences with the minimum variance of time interval
    """

    def __init__(self, mode, beta=0.2):
        self.beta = beta
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        sub_seq_length = int(self.beta * len(copied_sequence))
        if sub_seq_length < 2:
            return copied_sequence, copied_time_sequence

        # 计算每个可能子序列的时间差异方差
        cropped_vars = []
        crop_index = []
        for i in range(len(copied_sequence) - sub_seq_length + 1):
            temp_time_sequence = copied_time_sequence[i:i + sub_seq_length]
            temp_var = get_var(temp_time_sequence)
            cropped_vars.append(temp_var)
            crop_index.append(i)

        # 根据模式选择子序列
        if self.mode == 'maximum':
            start_index = crop_index[np.argmax(cropped_vars)]
        elif self.mode == 'minimum':
            start_index = crop_index[np.argmin(cropped_vars)]
        else:
            start_index = random.choice(crop_index)

        # 重排子序列
        end_index = start_index + sub_seq_length
        sub_seq = copied_sequence[start_index:end_index]
        sub_time = copied_time_sequence[start_index:end_index]

        # 创建索引并打乱
        indices = list(range(len(sub_seq)))
        random.shuffle(indices)

        reordered_seq = sub_seq.copy()
        reordered_time = sub_time.copy()

        for i, idx in enumerate(indices):
            reordered_seq[i] = sub_seq[idx]
            reordered_time[i] = sub_time[idx]

        # 组合结果
        result_sequence = copied_sequence[:start_index] + reordered_seq + copied_sequence[end_index:]
        result_time_sequence = copied_time_sequence[:start_index] + reordered_time + copied_time_sequence[end_index:]

        return result_sequence, result_time_sequence

class Mask(object):
    def __init__(self, item_similarity_model, mode, insert_rate=0.4, gamma=0.7, max_insert_num_per_pos=1):
        self.max_insert_num_per_pos = max_insert_num_per_pos
        self.gamma = gamma
        self.mode = mode
        self.insert_rate = insert_rate
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        # 计算时间差异
        time_diffs = []
        if len(copied_time_sequence) > 1:
            for i in range(len(copied_time_sequence) - 1):
                diff = abs(copied_time_sequence[i + 1] - copied_time_sequence[i])
                time_diffs.append(diff)

        # 确定需要mask的位置
        mask_nums = max(int(self.gamma * len(copied_sequence)), 1)
        if time_diffs:
            if self.mode == 'maximum':
                diff_sorted = np.argsort(time_diffs)[::-1]  # 时间差异大的位置优先
            elif self.mode == 'minimum':
                diff_sorted = np.argsort(time_diffs)  # 时间差异小的位置优先
            else:  # Random
                diff_sorted = list(range(len(time_diffs)))
                random.shuffle(diff_sorted)
            mask_idx = diff_sorted[:mask_nums]
        else:
            mask_idx = random.sample(range(len(copied_sequence)), mask_nums)

        # 对项目进行mask
        for idx in mask_idx:
            if idx < len(copied_sequence):
                copied_sequence[idx] = 0  # 假设0是mask值

        # 插入新项目
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)
        if time_diffs:
            insert_idx = diff_sorted[:insert_nums]
        else:
            insert_idx = random.sample(range(len(copied_sequence)), insert_nums)

        # 确保索引按顺序处理（从后往前插入，避免索引变化）
        insert_idx.sort()
        insert_idx=insert_idx[::-1]

        for idx in insert_idx:
            if idx < len(copied_sequence):
                # 生成要插入的项目
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(copied_sequence[idx], top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(copied_sequence[idx], top_k=top_k, with_score=True)
                    insert_items = _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    insert_items = self.item_similarity_model.most_similar(copied_sequence[idx], top_k=top_k)

                # 计算新时间（取前一个时间）
                base_time = copied_time_sequence[idx]
                insert_times = [base_time] * len(insert_items)

                # 插入位置（在idx之后）
                insert_position = idx + 1

                # 插入项目和时间
                copied_sequence = copied_sequence[:insert_position] + insert_items + copied_sequence[insert_position:]
                copied_time_sequence = copied_time_sequence[:insert_position] + insert_times + copied_time_sequence[
                                                                                               insert_position:]

        return copied_sequence, copied_time_sequence

class Downsample(object):
    """
    Randomly downsample a sequence to simulate data sparsity.
    The downsampling rate determines the proportion of items to be retained.
    """

    def __init__(self, downsample_rate=0.3):
        """
        Initialize the downsampler.

        :param downsample_rate: The proportion of items to retain in the sequence.
                                Value should be between 0 and 1.
        """
        if not 0 < downsample_rate <= 1:
            raise ValueError("downsample_rate must be between 0 and 1")
        self.downsample_rate = downsample_rate

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        copied_time_sequence = copy.deepcopy(time_sequence)

        # 计算需要保留的项目数量
        total_items = len(copied_sequence)
        items_to_retain = max(int(total_items * self.downsample_rate), 1)

        # 随机选择要保留的索引
        retained_indices = random.sample(range(total_items), items_to_retain)
        retained_indices.sort()

        # 创建下采样序列
        downsampled_items = [copied_sequence[i] for i in retained_indices]
        downsampled_times = [copied_time_sequence[i] for i in retained_indices]

        return downsampled_items, downsampled_times

class TimeWarping(object):
    """
    Apply time warping to a sequence using a randomly selected warping factor within a specified range.
    """

    def __init__(self, base_warping_factor=1.0, range_width=0.5):
        """
        Initialize the time warping object with a base warping factor and a range width.

        :param base_warping_factor: The central point of the warping factor range.
        :param range_width: The width of the range around the base warping factor.
        """
        self.warping_factor_range = (base_warping_factor - range_width / 2, base_warping_factor + range_width / 2)

    def __call__(self, item_sequence, time_sequence):
        """
        Apply time warping to the given time sequence using a randomly selected warping factor within the specified range.
w
        :param item_sequence: The sequence of items.
        :param time_sequence: The sequence of timestamps to be warped.
        :return: A new item sequence reordered according to the warped time sequence.
        """
        # Ensure the sequences are not altered in-place
        warped_time_sequence = copy.deepcopy(time_sequence)
        warped_item_sequence = copy.deepcopy(item_sequence)
        # Generate a random warping factor within the specified range
        warping_factor = random.uniform(*self.warping_factor_range)
        # Apply the random warping factor to each interval in the time sequence
        total_time = 0
        for i in range(len(warped_time_sequence) - 1):
            time_diff = warped_time_sequence[i + 1] - warped_time_sequence[i]
            warped_time_diff = time_diff * warping_factor
            total_time += warped_time_diff
            warped_time_sequence[i + 1] = total_time

        # Reorder items based on warped time intervals
        reordered_indices = sorted(range(len(warped_item_sequence)), key=lambda i: warped_time_sequence[i])
        warped_item_sequence = [warped_item_sequence[i] for i in reordered_indices]
        time_sequence = [time_sequence[i] for i in reordered_indices]
        return warped_item_sequence,time_sequence
