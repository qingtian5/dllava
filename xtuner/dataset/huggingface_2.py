from functools import partial
import json
import os
from datasets import Dataset as HFDataset
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
import torch
from torch import distributed as dist
from xtuner.registry import BUILDER, MAP_FUNC
from .utils import encode_fn
from itertools import chain
import concurrent.futures
from datasets import load_from_disk
import random

from xtuner.utils import IMAGE_TOKEN_INDEX

def get_lengths(example):
    assert 'input_ids' in example
    return {'length': len(example['input_ids'])}


def preprocess_json_file(file_path, data_path, workspace_dir, rank, tokenizer, max_length, dataset_map_fn, template_map_fn):
    json_file_path = os.path.join(data_path, file_path)
    cur_dataset = load_dataset_from_json(json_file_path)
    # file_size, valid_size, tagged_indices, length_list, tokens_list = post_process_sharding(dataset=cur_dataset, tokenizer=tokenizer, max_length=max_length, dataset_map_fn=dataset_map_fn, template_map_fn=template_map_fn, with_image_token=True)
    file_size, valid_size, tagged_indices, length_list, tokens_list, image_group_list = post_process_sharding(dataset=cur_dataset, tokenizer=tokenizer, max_length=max_length, dataset_map_fn=dataset_map_fn, template_map_fn=template_map_fn, with_image_token=True)
    with open(os.path.join(workspace_dir + '/tagged_indices', file_path), 'w') as f:
        f.write(json.dumps(tagged_indices))
    with open(os.path.join(workspace_dir + '/count', file_path), 'w') as f:
        f.write(json.dumps(str(file_size)))
    with open(os.path.join(workspace_dir + '/valid_count', file_path), 'w') as f:
        f.write(json.dumps(str(valid_size)))
    with open(os.path.join(workspace_dir + '/length_list', file_path), 'w') as f:
        f.write(json.dumps(length_list))
    with open(os.path.join(workspace_dir + '/grouped_indices', file_path), 'w') as f:
        f.write(json.dumps(image_group_list))

    image_group_list_none, image_group_list_single, image_group_list_multi = [], [], []
    for i in range(len(image_group_list)):
        if image_group_list[i] == -1:
            image_group_list_none += [i]
        elif image_group_list[i] == 100:
            image_group_list_single += [i]
        elif image_group_list[i] == 200:
            image_group_list_multi += [i]
    with open(os.path.join(workspace_dir + '/image_group', file_path + ".none"), 'w') as f:
        f.write(json.dumps(image_group_list_none))
    with open(os.path.join(workspace_dir + '/image_group', file_path + ".single"), 'w') as f:
        f.write(json.dumps(image_group_list_single))
    with open(os.path.join(workspace_dir + '/image_group', file_path + ".multi"), 'w') as f:
        f.write(json.dumps(image_group_list_multi))

    return f'[rank-{rank}]完成单个文件预处理: {file_path}'


def get_merged_list(data_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    final_list = []
    for i in range(len(file_list)):
        file_path = os.path.join(data_path, file_list[i])
        with open(file_path, 'r') as f:
            cur_list = json.load(f)
            final_list.extend(cur_list)
    return final_list

def get_merged_list_image_group(data_path, group):
    file_list = os.listdir(data_path)
    file_list.sort()
    file_list = [os.path.join(data_path, f) for f in file_list if f.endswith(group)]
    final_list = []
    final_list_size = []
    for i in range(len(file_list)):
        file_path = file_list[i]
        with open(file_path, 'r') as f:
            cur_list = json.load(f)
            final_list.extend(cur_list)
            final_list_size += [len(cur_list)]
    return final_list, final_list_size

def build_sharding_indices_2(workspace_dir, data_path, sampler_name, group_batch_size, tokenizer, max_length, dataset_map_fn, template_map_fn, skip_filer_and_map, seed, sharding_indices_file_path, rank, world_size, file_list):

    file_list = os.listdir(data_path)
    file_list.sort()

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(preprocess_json_file, file_list[i], data_path, workspace_dir, rank, tokenizer, max_length, dataset_map_fn, template_map_fn) for i in range(len(file_list)) if i % world_size == rank]

        for future in concurrent.futures.as_completed(futures, 600):
            print_log(future.result(), logger='current')
    dist.barrier()
    # print(f'\n[rank-{rank}]完成json数据预处理！{sampler_name}, 开始构建shuffle索引...')
    print_log(f'[rank-{rank}]预处理文件完成!{sampler_name}, 开始构建索引...', logger='current')

    if rank == 0:
        if sampler_name == "ImageGroupedSampler":
            assert skip_filer_and_map == False
            assert group_batch_size > 0
            tagged_indices_flattened = get_merged_list(workspace_dir + '/tagged_indices')
            length_list_flattened = get_merged_list(workspace_dir + '/length_list')

            image_group_list_multi, image_group_list_multi_size_list = get_merged_list_image_group(workspace_dir + '/image_group', '.multi')
            image_group_list_single, image_group_list_single_size_list = get_merged_list_image_group(workspace_dir + '/image_group', '.single')
            image_group_list_none, image_group_list_none_size_list = get_merged_list_image_group(workspace_dir + '/image_group', '.none')
            image_multi_samples = len(image_group_list_multi)
            image_single_samples = len(image_group_list_single)
            image_none_samples = len(image_group_list_none)

            valid_size = 0
            valid_size_list = []
            for i in range(len(file_list)):
                file_path = os.path.join(workspace_dir + '/valid_count', file_list[i])
                with open(file_path, 'r') as f:
                    cur_size = json.load(f)
                    valid_size_list += [int(cur_size)]
                    valid_size += int(cur_size)

            assert len(tagged_indices_flattened) == len(length_list_flattened)

            total_size = len(tagged_indices_flattened)

            num_samples = valid_size // world_size * world_size // group_batch_size * group_batch_size # will ignore the last world_size - 1 items at most

            micro_batch_size = group_batch_size
            group_batch_size = group_batch_size * 50

            num_groups = num_samples // world_size // group_batch_size
            print(f'rank-{rank}, world_size-{world_size}, using ImageGroupedSampler, num_samples={num_samples}, group_batch_size={group_batch_size}, num_groups={num_groups}\n')

            print(f'rank-{rank}, world_size-{world_size}, using ImageGroupedSampler, image_multi_samples={image_multi_samples}, image_single_samples={image_single_samples}, image_none_samples={image_none_samples}\n')
            image_multi_samples_per_group = image_multi_samples // world_size // num_groups // micro_batch_size * micro_batch_size
            image_single_samples_per_group = image_single_samples // world_size // num_groups // micro_batch_size * micro_batch_size
            image_none_samples_per_group = group_batch_size - image_multi_samples_per_group - image_single_samples_per_group
            print(f'rank-{rank}, world_size-{world_size}, using ImageGroupedSampler, image_multi_samples_per_group={image_multi_samples_per_group}, image_single_samples_per_group={image_single_samples_per_group}, image_none_samples_per_group={image_none_samples_per_group}\n')

            g = torch.Generator()
            g.manual_seed(seed)

            image_multi_samples = image_multi_samples // world_size * world_size
            image_multi_random_indices = torch.randperm(image_multi_samples, generator=g).tolist()
            rank_size = image_multi_samples // world_size
            for i in range(image_multi_samples):
                image_multi_random_indices[i] = (200, image_multi_random_indices[i])
            image_multi_random_indices_list_by_rank = [image_multi_random_indices[i:i + rank_size] for i in range(0, image_multi_samples, rank_size)]

            image_single_samples = image_single_samples // world_size * world_size
            image_single_random_indices = torch.randperm(image_single_samples, generator=g).tolist()
            for i in range(image_single_samples):
                image_single_random_indices[i] = (100, image_single_random_indices[i])
            rank_size = image_single_samples // world_size
            image_single_random_indices_list_by_rank = [image_single_random_indices[i:i + rank_size] for i in range(0, image_single_samples, rank_size)]
            image_none_samples = image_none_samples // world_size * world_size
            image_none_random_indices = torch.randperm(image_none_samples, generator=g).tolist()
            for i in range(image_none_samples):
                image_none_random_indices[i] = (-1, image_none_random_indices[i])
            rank_size = image_none_samples // world_size
            image_none_random_indices_list_by_rank = [image_none_random_indices[i:i + rank_size] for i in range(0, image_none_samples, rank_size)]

            multi_indices_mapping = [(0, -1, -1)] * image_multi_samples
            single_indices_mapping = [(0, -1, -1)] * image_single_samples
            none_indices_mapping = [(0, -1, -1)] * image_none_samples
            for rank_id in range(world_size):
                rank_random_indice = []
                offset = 0
                for i in range(num_groups):
                    rank_random_indice.extend(image_multi_random_indices_list_by_rank[rank_id][offset*image_multi_samples_per_group:(offset+1)*image_multi_samples_per_group])
                    rank_random_indice.extend(image_single_random_indices_list_by_rank[rank_id][offset*image_single_samples_per_group:(offset+1)*image_single_samples_per_group])
                    rank_random_indice.extend(image_none_random_indices_list_by_rank[rank_id][offset*image_none_samples_per_group:(offset+1)*image_none_samples_per_group])
                    offset += 1
                for i in range(len(rank_random_indice)):
                    group, from_index = rank_random_indice[i]
                    if group == 200:
                        multi_indices_mapping[from_index] = (1, rank_id, i)
                    elif group == 100:
                        single_indices_mapping[from_index] = (1, rank_id, i)
                    else:
                        none_indices_mapping[from_index] = (1, rank_id, i)

            multi_indices_mapping_by_file = []
            offset, j = 0, 0
            while offset < len(multi_indices_mapping):
                chunk_size = image_group_list_multi_size_list[j]
                cur_indices = multi_indices_mapping[offset:offset + chunk_size]
                multi_indices_mapping_by_file += [cur_indices]
                j += 1
                offset += chunk_size

            single_indices_mapping_by_file = []
            offset, j = 0, 0
            while offset < len(single_indices_mapping):
                chunk_size = image_group_list_single_size_list[j]
                cur_indices = single_indices_mapping[offset:offset + chunk_size]
                single_indices_mapping_by_file += [cur_indices]
                j += 1
                offset += chunk_size

            none_indices_mapping_by_file = []
            offset, j = 0, 0
            while offset < len(none_indices_mapping):
                chunk_size = image_group_list_none_size_list[j]
                cur_indices = none_indices_mapping[offset:offset + chunk_size]
                none_indices_mapping_by_file += [cur_indices]
                j += 1
                offset += chunk_size
            for i in range(len(multi_indices_mapping_by_file)):
                with open(os.path.join(workspace_dir + '/mapping', file_list[i] + '.multi'), 'w') as f:
                    f.write(json.dumps(multi_indices_mapping_by_file[i]))
            for i in range(len(single_indices_mapping_by_file)):
                with open(os.path.join(workspace_dir + '/mapping', file_list[i] + '.single'), 'w') as f:
                    f.write(json.dumps(single_indices_mapping_by_file[i]))
            for i in range(len(none_indices_mapping_by_file)):
                with open(os.path.join(workspace_dir + '/mapping', file_list[i] + '.none'), 'w') as f:
                    f.write(json.dumps(none_indices_mapping_by_file[i]))

        if sampler_name == "LengthGroupedSampler":
            assert skip_filer_and_map == False
            assert group_batch_size > 0
            tagged_indices_flattened = get_merged_list(workspace_dir + '/tagged_indices')
            length_list_flattened = get_merged_list(workspace_dir + '/length_list')

            valid_size = 0
            valid_size_list = []
            for i in range(len(file_list)):
                file_path = os.path.join(workspace_dir + '/valid_count', file_list[i])
                with open(file_path, 'r') as f:
                    cur_size = json.load(f)
                    valid_size_list += [int(cur_size)]
                    valid_size += int(cur_size)

            assert len(tagged_indices_flattened) == len(length_list_flattened)

            total_size = len(tagged_indices_flattened)

            num_samples = valid_size // world_size * world_size # will ignore the last world_size - 1 items at most

            group_batch_size = group_batch_size * 50

            print(f'rank-{rank}, world_size-{world_size}, using LengthGroupedSampler, num_samples={num_samples}, group_batch_size={group_batch_size}\n')

            valid_length_list_all_gathered = [0] * valid_size
            valid_cnt = 0
            for i in range(total_size):
                if tagged_indices_flattened[i] == 1:
                    valid_length_list_all_gathered[valid_cnt] = length_list_flattened[i]
                    valid_cnt += 1

            g = torch.Generator()
            g.manual_seed(seed)
            random_indices = torch.randperm(num_samples, generator=g).tolist() # j=random_indice[i], j is the index
            rank_size = num_samples // world_size # total size for each rank
            random_indices_list_by_rank = [random_indices[i:i + rank_size] for i in range(0, num_samples, rank_size)]
            valid_indices_mapping = [(0, -1, -1)] * num_samples
            for rank_id, rank_random_indice in enumerate(random_indices_list_by_rank):
                sorted_indice_list = []
                for k in range(0, len(rank_random_indice), group_batch_size): # for each group
                    current_group_indice = rank_random_indice[k:k+group_batch_size]
                    sorted_indice_list += [sorted(current_group_indice, key=lambda x: valid_length_list_all_gathered[x], reverse=True)]
                sorted_indice_flattened = list(chain.from_iterable(sorted_indice_list))
                for i in range(len(sorted_indice_flattened)):
                    from_index = sorted_indice_flattened[i]
                    valid_indices_mapping[from_index] = (1, rank_id, i)

            chunk_size_list = valid_size_list

            valid_indices_mapping_by_file = []

            offset, j = 0, 0
            while offset < len(valid_indices_mapping):
                chunk_size = chunk_size_list[j]
                cur_indices = valid_indices_mapping[offset:offset + chunk_size]
                valid_indices_mapping_by_file += [cur_indices]
                j += 1
                offset += chunk_size

            for i in range(len(file_list)):
                path=os.path.join(workspace_dir + '/mapping', file_list[i])
                print(f'file={path}')
                with open(os.path.join(workspace_dir + '/mapping', file_list[i]), 'w') as f:
                    f.write(json.dumps(valid_indices_mapping_by_file[i]))

        else:
            tagged_indices_flattened = get_merged_list(workspace_dir + '/tagged_indices')

            valid_size = tagged_indices_flattened.count(1)
            num_samples = valid_size // world_size * world_size # will ignore the last world_size - 1 items at most
            g = torch.Generator()
            g.manual_seed(seed)
            random_indices = torch.randperm(num_samples, generator=g).tolist() # j=random_indice[i], j is the index
            rank_size = num_samples // world_size # total size for each rank
            random_indices_list_by_rank = [random_indices[i:i + rank_size] for i in range(0, num_samples, rank_size)]


            valid_indices_mapping = [(1, -1, -1)] * num_samples
            for rank_id, rank_random_indice in enumerate(random_indices_list_by_rank):
                for i in range(len(rank_random_indice)):
                    from_index = rank_random_indice[i]
                    valid_indices_mapping[from_index] = (1, rank_id, i)

            chunk_size_list = []
            for i in range(len(file_list)):
                file_path = os.path.join(workspace_dir + '/tagged_indices', file_list[i])
                with open(file_path, 'r') as f:
                    cur_size_list = json.load(f)
                    chunk_size_list += [int(cur_size_list.count(1))]

            valid_indices_mapping_by_file = []

            offset, j = 0, 0
            while offset < len(valid_indices_mapping):
                if j % 100 == 0:
                    print(f'\n[rank-{rank}]索引映射第{j}/{len(chunk_size_list)}个文件ing')
                    print_log(f'[rank-{rank}]索引映射第{j}/{len(chunk_size_list)}个文件ing', logger='current')
                chunk_size = chunk_size_list[j]
                cur_indices = valid_indices_mapping[offset:offset + chunk_size]
                valid_indices_mapping_by_file += [cur_indices]
                j += 1
                offset += chunk_size

            for i in range(len(file_list)):
                with open(os.path.join(workspace_dir + '/mapping', file_list[i]), 'w') as f:
                    f.write(json.dumps(valid_indices_mapping_by_file[i]))

    dist.barrier()
    # print(f'\n[rank-{rank}]构建shuffle索引完成!')
    print_log(f'[rank-{rank}]构建shuffle索引完成!', logger='current')


def move_to_rank_elements(file_path, data_path, workspace_dir, rank, world_size, sampler_name):
    # with open(os.path.join(workspace_dir + '/tagged_indices', file_path), 'r') as f:
    #     tagged_indices_mapping = json.load(f)
    #
    # if sampler_name == "ImageGroupedSampler":
    #     with open(os.path.join(workspace_dir + '/grouped_indices', file_path), 'r') as f:
    #         grouped_indices_mapping = json.load(f)
    #     elements = [[] for _ in range(world_size)]
    #
    #     multi_path = os.path.join(workspace_dir + '/mapping', file_path + ".multi")
    #     if os.path.exists(multi_path):
    #         with open(multi_path, 'r') as f:
    #             indices_mapping = json.load(f)
    #         with open(os.path.join(data_path, file_path), 'r') as f:
    #             json_data = json.load(f)
    #             idx_icr = 0
    #             for idx, example in enumerate(json_data):
    #                 is_valid = tagged_indices_mapping[idx]
    #                 if is_valid == 1:
    #                     group = grouped_indices_mapping[idx]
    #                     if group == 200:
    #                         if idx_icr >= len(indices_mapping):
    #                             break
    #                         _, to_rank, to_index = indices_mapping[idx_icr]
    #                         if to_index == -1:
    #                             idx_icr += 1
    #                             continue
    #                         if isinstance(example['id'], int):
    #                             example['id'] = str(example['id'])
    #                         elements[to_rank] += [(to_index, example)]
    #                         idx_icr += 1
    #
    #     single_path = os.path.join(workspace_dir + '/mapping', file_path + ".single")
    #     if os.path.exists(single_path):
    #         with open(single_path, 'r') as f:
    #             indices_mapping = json.load(f)
    #         with open(os.path.join(data_path, file_path), 'r') as f:
    #             json_data = json.load(f)
    #             idx_icr = 0
    #             for idx, example in enumerate(json_data):
    #                 is_valid = tagged_indices_mapping[idx]
    #                 if is_valid == 1:
    #                     group = grouped_indices_mapping[idx]
    #                     if group == 100:
    #                         if idx_icr >= len(indices_mapping):
    #                             break
    #                         _, to_rank, to_index = indices_mapping[idx_icr]
    #                         if to_index == -1:
    #                             idx_icr += 1
    #                             continue
    #                         if isinstance(example['id'], int):
    #                             example['id'] = str(example['id'])
    #                         elements[to_rank] += [(to_index, example)]
    #                         idx_icr += 1
    #
    #     none_path = os.path.join(workspace_dir + '/mapping', file_path + ".none")
    #     if os.path.exists(none_path):
    #         with open(none_path, 'r') as f:
    #             indices_mapping = json.load(f)
    #         with open(os.path.join(data_path, file_path), 'r') as f:
    #             json_data = json.load(f)
    #             idx_icr = 0
    #             for idx, example in enumerate(json_data):
    #                 is_valid = tagged_indices_mapping[idx]
    #                 if is_valid == 1:
    #                     group = grouped_indices_mapping[idx]
    #                     if group == -1:
    #                         if idx_icr >= len(indices_mapping):
    #                             break
    #                         _, to_rank, to_index = indices_mapping[idx_icr]
    #                         if to_index == -1:
    #                             idx_icr += 1
    #                             continue
    #                         if isinstance(example['id'], int):
    #                             example['id'] = str(example['id'])
    #                         elements[to_rank] += [(to_index, example)]
    #                         idx_icr += 1
    # world_size_list = list(range(world_size))
    # random.shuffle(world_size_list)
    # for i in world_size_list:
    #     print(f'\n[rank-{rank}]写入rank-{i} elements!')
    #     with open(os.path.join(workspace_dir + '/rank_elements/' + str(i), file_path), 'w', encoding='utf-8') as f:
    #         f.write(json.dumps(elements[i]))
    #
    # return f'[rank-{rank}]完成json文件移动: {file_path}'

    elements = [[] for _ in range(world_size)]
    with open(os.path.join(workspace_dir + '/mapping', file_path), 'r') as f:
        indices_mapping = json.load(f)

    with open(os.path.join(workspace_dir + '/tagged_indices', file_path), 'r') as f:
        tagged_indices_mapping = json.load(f)


    with open(os.path.join(data_path, file_path), 'r') as f:
        json_data = json.load(f)
        idx_icr = 0
        for idx, example in enumerate(json_data):
            is_valid = tagged_indices_mapping[idx]
            if is_valid == 1:
                if idx_icr >= len(indices_mapping):
                    break
                _, to_rank, to_index = indices_mapping[idx_icr]
                if isinstance(example['id'], int):
                    example['id'] = str(example['id'])
                if sampler_name == "LengthGroupedSampler":
                    elements[to_rank] += [(to_index, example)]
                else:
                    elements[to_rank] += [example]
                idx_icr += 1
    world_size_list = list(range(world_size))
    random.shuffle(world_size_list)
    for i in world_size_list:
        print(f'\n[rank-{rank}]写入rank-{i} elements!')
        with open(os.path.join(workspace_dir + '/rank_elements/' + str(i), file_path), 'w') as f:
            f.write(json.dumps(elements[i]))

    return f'[rank-{rank}]完成json文件移动: {file_path}'


def build_sharding_datasets_2(workspace_dir, data_path, sampler_name, group_batch_size, tokenizer, max_length, dataset_map_fn, template_map_fn, skip_filer_and_map, seed, sharding_indices_file_path, rank, world_size, file_list):
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(move_to_rank_elements, file_list[i], data_path, workspace_dir, rank, world_size, sampler_name) for i in range(len(file_list)) if i % world_size == rank]

        for future in concurrent.futures.as_completed(futures, 1200):
            print_log(future.result(), logger='current')
            print(future.result())
    dist.barrier()
    print_log(f'[rank-{rank}]完成所有json文件移动！开始合并...', logger='current')

    file_list = os.listdir(workspace_dir + '/rank_elements/' + str(rank))
    file_list.sort()

    json_data_list = []
    if sampler_name == "LengthGroupedSampler":
        tmp_data_list = []
        for i in range(len(file_list)):
            file_path = os.path.join(workspace_dir + '/rank_elements/' + str(rank), file_list[i])
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                tmp_data_list.extend(json_data)

        indexed_data_on_rank = [None] * len(tmp_data_list)
        for i, example in tmp_data_list:
            indexed_data_on_rank[i] = example
        del tmp_data_list
        for idx in range(len(indexed_data_on_rank)):
            if isinstance(indexed_data_on_rank[idx]['id'], int):
                indexed_data_on_rank[idx]['id'] = str(indexed_data_on_rank[idx]['id'])
        json_data_list = indexed_data_on_rank
        dataset = HFDataset.from_list(json_data_list)
    elif sampler_name == "ImageGroupedSampler":
        tmp_data_list = []
        for i in range(len(file_list)):
            file_path = os.path.join(workspace_dir + '/rank_elements/' + str(rank), file_list[i])
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                tmp_data_list.extend(json_data)

        indexed_data_on_rank = [None] * len(tmp_data_list)
        for i, example in tmp_data_list:
            indexed_data_on_rank[i] = example
        del tmp_data_list
        for idx in range(len(indexed_data_on_rank)):
            if isinstance(indexed_data_on_rank[idx]['id'], int):
                indexed_data_on_rank[idx]['id'] = str(indexed_data_on_rank[idx]['id'])
            if 'image' in indexed_data_on_rank[idx] and isinstance(indexed_data_on_rank[idx]['image'], str):
                indexed_data_on_rank[idx]['image'] = [indexed_data_on_rank[idx]['image']]

        json_data_list = indexed_data_on_rank
        if len(json_data_list) > 0:
            if 'image' not in json_data_list[0]:
                json_data_list[0]['image'] = None
        dataset = HFDataset.from_list(json_data_list)
    else:
        for i in range(len(file_list)):
            file_path = os.path.join(workspace_dir + '/rank_elements/' + str(rank), file_list[i])
            print(f'\n[rank-{rank}]读入文件-{file_path}')
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            if len(json_data) > 0:
                if 'image' not in json_data[0]:
                    json_data[0]['image'] = None
            json_data_list.extend(json_data)

        dataset = HFDataset.from_list(json_data_list)
        dataset = dataset.shuffle(seed=seed)

    dist.barrier()
    print_log(f'[rank-{rank}]完成合并！开始encode-1...', logger='current')
    if dataset_map_fn is not None:
        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError('dataset_map_fn must be a function or a '
                                "registered function's string in MAP_FUNC, "
                                f"but got a string of '{dataset_map_fn}'")

        dataset = dataset.map(dataset_map_fn, num_proc=8)
    dist.barrier()
    print_log(f'[rank-{rank}]完成合并！开始encode-2...', logger='current')
    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(
                template_map_fn, Config) or isinstance(template_map_fn,
                                                    ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=32)
    dist.barrier()
    print_log(f'[rank-{rank}]完成合并！开始encode-3...', logger='current')
    # tokenize
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_image_token=True,
            input_ids_with_output=True),
        remove_columns=None,
        num_proc=16)
    dist.barrier()
    print_log(f'[rank-{rank}]完成合并！开始encode-4...', logger='current')

    dataset = dataset.map(get_lengths, num_proc=32)
    setattr(dataset, 'length', dataset['length'])

    dist.barrier()
    print_log(f'[rank-{rank}]完成构造数据集！长度为{len(dataset)}， 开始dump...', logger='current')

    dataset.save_to_disk(workspace_dir + '/rank_datasets/' + str(rank) +'/data.bin')

    if rank == 0:
        print(f"workspace_dir = {workspace_dir}, path = {os.path.join(workspace_dir, 'rank_datasets/_SUCC')}")
        with open(os.path.join(workspace_dir, 'rank_datasets/_SUCC'), 'w') as f:
            f.write('Done')
    dist.barrier()
    print_log(f'[rank-{rank}]完成dump！', logger='current')
    return dataset



def process_sharding_files(data_path, sampler_name, group_batch_size, tokenizer, max_length, dataset_map_fn, template_map_fn, skip_filer_and_map, seed, work_dir):

    rank, world_size = dist.get_rank(), dist.get_world_size()

    file_list = os.listdir(data_path)
    file_list.sort()

    workspace_dir = os.path.join(work_dir, data_path.replace('/', '_') + '_' + sampler_name.lower() + '_g' + str(group_batch_size) + '_s' + str(int(skip_filer_and_map)) + '_w' + str(world_size))


    if os.path.exists(workspace_dir + '/rank_datasets/_SUCC'):
        print(f"[rank-{rank}]检查到数据集已构建完成，接下来直接加载: {workspace_dir + '/rank_datasets/' + str(rank) +'/data.bin'}")
        return load_from_disk(workspace_dir + '/rank_datasets/' + str(rank) +'/data.bin')

    print(f'\n[rank-{rank}]没有检测到子数据集，开始构建...')
    print_log(f'[rank-{rank}]没有检测到子数据集，开始构建...', logger='current')

    dist.barrier()
    if rank == 0:
        print(f'\n[rank-{rank}]创建子目录...')
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(workspace_dir + '/count', exist_ok=True)
        os.makedirs(workspace_dir + '/valid_count', exist_ok=True)
        os.makedirs(workspace_dir + '/length_list', exist_ok=True)
        os.makedirs(workspace_dir + '/image_group', exist_ok=True)
        os.makedirs(workspace_dir + '/grouped_indices', exist_ok=True)
        os.makedirs(workspace_dir + '/tagged_indices', exist_ok=True)
        os.makedirs(workspace_dir + '/mapping', exist_ok=True)
        os.makedirs(workspace_dir + '/rank_elements', exist_ok=True)
        for i in range(world_size):
            os.makedirs(workspace_dir + '/rank_elements/' + str(i), exist_ok=True)
        os.makedirs(workspace_dir + '/rank_datasets', exist_ok=True)
        for i in range(world_size):
            os.makedirs(workspace_dir + '/rank_datasets/' + str(i), exist_ok=True)
    print(f'\n[rank-{rank}]准备构建索引...')
    dist.barrier()

    build_sharding_indices_2(workspace_dir=workspace_dir, data_path=data_path, sampler_name=sampler_name, group_batch_size=group_batch_size, tokenizer=tokenizer, max_length=max_length,
                               dataset_map_fn=dataset_map_fn, template_map_fn=template_map_fn, skip_filer_and_map=skip_filer_and_map,
                               seed=seed, sharding_indices_file_path=None, rank=rank, world_size=world_size, file_list=file_list)

    dataset = build_sharding_datasets_2(workspace_dir=workspace_dir, data_path=data_path, sampler_name=sampler_name, group_batch_size=group_batch_size, tokenizer=tokenizer, max_length=max_length,
                               dataset_map_fn=dataset_map_fn, template_map_fn=template_map_fn, skip_filer_and_map=skip_filer_and_map,
                               seed=seed, sharding_indices_file_path=None, rank=rank, world_size=world_size, file_list=file_list)
    return dataset


def load_dataset_from_json(file_path):
    json_data = json.load(open(file_path))
    for idx in range(len(json_data)):
        if isinstance(json_data[idx]['id'], int):
            json_data[idx]['id'] = str(json_data[idx]['id'])
    if len(json_data) > 0:
        if 'image' not in json_data[0]:
            json_data[0]['image'] = None
    dataset = HFDataset.from_list(json_data)
    return dataset


def post_process_sharding(dataset,
                        tokenizer,
                        max_length,
                        dataset_map_fn=None,
                        template_map_fn=None,
                        input_ids_with_output=True,
                        with_image_token=False,
                        map_num_proc=32,
                        skip_filer_and_map=False):

    file_size, valid_size = len(dataset), 0
    # tagged_indices, length_list, tokens_list = [1] * file_size, [0] * file_size, [0] * file_size
    tagged_indices, length_list, tokens_list, image_group_list = [1] * file_size, [0] * file_size, [0] * file_size, [0] * file_size

    if skip_filer_and_map:
        return file_size, file_size, tagged_indices, None

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(template_map_fn, Config) or isinstance(template_map_fn, ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=map_num_proc)

    # tokenize
    if isinstance(tokenizer, dict) or isinstance(tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)

    # filtering
    for i, example in enumerate(dataset):
        if len(example['conversation']) > 0:
            encoded_ex = encode_fn(example=example, tokenizer=tokenizer, max_length=max_length, with_image_token=with_image_token, input_ids_with_output=input_ids_with_output)
            # image_valid = image_map_fn(example)
            if any(label >= 0 for label in encoded_ex['labels']):
                if 'input_ids' in encoded_ex:
                    num_image_tokens = (torch.tensor(encoded_ex['input_ids']) == IMAGE_TOKEN_INDEX).sum()

                    if example.get('image', None) is not None and isinstance(example['image'], list) and len(example['image']) > 1:
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 200
                        image_group_list[i] = 200
                        num_images = len(example['image'])
                    elif example.get('image', None) is not None and isinstance(example['image'], list) and len(example['image']) == 1:
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 100
                        image_group_list[i] = 100
                        num_images = len(example['image'])
                    elif example.get('image', None) is not None and isinstance(example['image'], str):
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 100
                        image_group_list[i] = 100
                        num_images = 1
                    elif example.get('image', None) is None:
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = -1
                        image_group_list[i] = -1
                        num_images = 0

                    if num_image_tokens == num_images:
                        valid_size += 1
                    else:
                        tagged_indices[i] = 0
            else:
                tagged_indices[i] = 0
        else:
            tagged_indices[i] = 0

    return file_size, valid_size, tagged_indices, length_list, tokens_list, image_group_list
