import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import (CenterCrop, RandAugment,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize)
from tqdm import tqdm

from datasets import register
from collections import defaultdict
import random
from PIL import Image

decord.bridge.set_bridge('torch')

DEFAULT_MEAN_STD = {'mse': (0.008022358641028404, 0.006658855825662613),
    'perceptual': (0.19749267399311066, 0.07615979015827179),
    'psnr': (22.416458129882812, 4.018338680267334),
    'ssim': (0.9994804859161377, 0.0005285052466206253),
    'fake_logits': (0.06471613049507141, 0.221718892455101)}

def func_none():
    return None


def read_video_with_retry(uri, retries=5, delay=1):
    for i in range(retries):
        try:
            vr = decord.VideoReader(uri)
            return vr
        except Exception as e:
            print(f"Error reading {uri}, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to read {uri} after {retries} retries")


def VideoTransform(crop_size=128, scale=1.00, ratio=1.00, eval_tfm=False, 
    rand_flip='no'):
    if eval_tfm:
        transform = transforms.Compose([Resize(size=crop_size, antialias=True), CenterCrop(crop_size)])
    else:
        if scale == 1.0 and ratio == 1.0:
            tfm_list = [Resize(size=crop_size, antialias=True), CenterCrop(crop_size)]
        else:
            tfm_list = [Resize(size=int(crop_size/scale), antialias=True), 
                RandomResizedCrop(crop_size, (1./scale**2, 1), (1./ratio, ratio), antialias=True)]
        if rand_flip != 'no':
            tfm_list.append(RandomHorizontalFlip())
        transform = transforms.Compose(tfm_list)

    return transform    

@register('video_dataset')
class VideoDataset(Dataset):

    def __init__(
        self,
        root_path,
        frame_num,
        cls_vid_num,
        crop_size,
        rand_flip='no',
        split='train',
        csv_file='',
        scale=1.0,
        aspect_ratio=1.0,
        rand_augment='no',
        frame_rate='native', # 'uniform' or 'native'
        test_group=0,
        use_all_frames=False,
        pre_load=False,
        score_path=None,
        score_type='mse',
        with_scores=True,
        use_all_frames_step=None,
        ar_ann_path=None
    ):
        self.csv_file = csv_file
        self.is_image = "imagenet" in csv_file
        self.frame_num = frame_num
        self.crop_size = crop_size
        assert frame_rate in ['uniform', 'native', 'fixed']
        self.frame_rate = frame_rate
        self.test_group = test_group
        self.use_all_frames = use_all_frames
        self.use_all_frames_step = use_all_frames_step if use_all_frames_step is not None else frame_num
        self.num_classes = None
        self.label2action = None
        self.action2label = None
        self.vid2label = defaultdict(func_none)

        self.scores = None
        self.score_type = score_type

        if score_path is not None and os.path.exists(score_path):
            print(f"Load scores from {score_path}")
            self.scores = torch.load(score_path)
            if score_type in ("psnr", "ssim"):
                self.scores[score_type] *= -1
            self.score_mean = self.scores[score_type].mean()
            self.score_std = self.scores[score_type].std()
            self.scores['name2idx'] = {name: idx for idx, name in enumerate(self.scores['id'])}
        else:
            self.score_mean, self.score_std = DEFAULT_MEAN_STD[score_type]

        self.ar_ann_dir, self.ar_ann_path = None, None
        if ar_ann_path is not None and os.path.isfile(ar_ann_path):
            self.ar_annotations = torch.load(ar_ann_path)
            self.ar_ann_path = ar_ann_path
            fname2id = defaultdict(list)
            for idx, fname in enumerate(self.ar_annotations['id']):
                name = fname.split('frames__')[-1]
                fname2id[name].append(idx)
            self.fname2id = fname2id
        elif ar_ann_path is not None and os.path.isdir(ar_ann_path):
            self.ar_ann_dir = ar_ann_path
            self.file_list = os.listdir(self.ar_ann_dir)
            

        if csv_file.lower().startswith('null'): # fake dataset to test 
            if csv_file.lower().startswith('null128'):
                num = 128
            else:
                num = 32*7000
            self.fake = True
            self.vid_list = ['' for _ in range(num)]
            self.augment = None
            self.pre_load = False
            self.split = split
            self.rand_flip = rand_flip
            self.crop_size = crop_size
            self.scale = scale
            self.aspect_ratio = aspect_ratio
            self.frame_num = frame_num
            self.index_map_cache_dir = os.path.join(root_path, 'index_map_cache')
            self.idx2label = {i: i % 101 for i in range(num)}
            all_labels = list(self.idx2label.values())
            self.num_classes = 101
            self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
            if self.split == 'train':
                self.cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                    ratio=self.aspect_ratio, eval_tfm=False)
            elif self.split == 'test':
                self.cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
            else:
                raise NotImplementedError(f'Unknown split: {self.split}')
            
            return

        
        self.fake = False
        self.with_scores = with_scores
        if '+' in csv_file:
            self.multiple_datasets = True
            csv_files = csv_file.split('+')
            if cls_vid_num == '-1_-1':
                cls_vid_num = '+'.join(['-1_-1'] * len(csv_files))
            assert '+' in cls_vid_num, 'cls_vid_num should be separated by +'
            cls_vid_nums = cls_vid_num.split('+')
            assert len(csv_files) == len(cls_vid_nums), 'Number of csv_files should be the same as cls_vid_nums'
        else:
            self.multiple_datasets = False
            csv_files = [csv_file]
            cls_vid_nums = [cls_vid_num]
        self.vid_list = []

        self.index_map_cache_dir = os.path.join(root_path, 'index_map_cache')
        os.makedirs(self.index_map_cache_dir, exist_ok=True)

        for csv_file, cls_vid_num in zip(csv_files, cls_vid_nums):
            if csv_file != '':
                if not os.path.isabs(csv_file):
                    csv_file = os.path.join(root_path, csv_file)
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                if csv_file.endswith('.csv'):
                    self.process_csv_data(csv_file, cls_num, vid_num)
                elif csv_file.endswith('.js'):
                    with open(csv_file, 'r') as f:
                        vid_dict = json.load(f)
                    sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                    vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                    self.vid_list += sum(vid_list, [])
            else:
                vid_list = []
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                root_path = os.path.join(root_path, split)
                for cur_cls in sorted(os.listdir(root_path)[:cls_num]):
                    cur_dir = os.path.join(root_path, cur_cls)
                    for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                        vid_list.append(os.path.join(cur_dir, cur_vid))
                self.vid_list += vid_list

        self.vid_list = sorted(self.vid_list)
        self.split, self.frame_num, self.rand_flip = split, frame_num, rand_flip
        self.crop_size, self.scale, self.aspect_ratio = crop_size, scale, aspect_ratio
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        if self.split == 'train':
            self.cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                ratio=self.aspect_ratio, eval_tfm=False)
        elif self.split == 'test':
           self.cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        else:
            raise NotImplementedError(f'Unknown split: {self.split}')


        self.pre_load = pre_load
        if self.pre_load:
            raise NotImplementedError('Pre-loading is not implemented yet')

        if self.scores is not None and with_scores:
            pre_num = len(self.vid_list)
            self.vid_list = [i for i in self.vid_list if os.path.basename(i) in self.scores['name2idx']]
            
            print(f"filter out {pre_num - len(self.vid_list)} videos without scores gt.")

        self.index_videos()

    def process_csv_data(self, csv_file, cls_num, vid_num):
        for i in range(10):
            try:
                csv_data = pd.read_csv(csv_file)
                if 'label' in csv_data:
                    if vid_num == -1:
                        vid_list = csv_data.sort_values(['label', 'path']).groupby('label', group_keys=False).apply(lambda x: x)
                    else:
                        vid_list = csv_data.sort_values(['label', 'path']).groupby('label', group_keys=False).head(vid_num)   

                    if cls_num != -1:
                        vid_list = pd.concat([group for _,group in vid_list.groupby('label')][:cls_num])

                    vid_list, _, _ = [vid_list[k].tolist() for k in ['path', 'label', 'action']]
                    self.vid_list += vid_list
                else:
                    self.vid_list += csv_data['path'].tolist()
                    
                return

            except Exception as e:
                print(e)
                if not os.path.exists(csv_file):
                    print(f'{csv_file} does not exist')
                    raise FileNotFoundError(f'{csv_file} does not exist')
                print(f'Error reading {csv_file}, retrying ({i+1}/5)...')
                print(f'{vid_num=}, {cls_num=}')
                print(f'{csv_data.size=}') # should be 38148
                print(f'{csv_data.info=}')
                print(f'{csv_data.columns.tolist()=}')
                continue

        raise RuntimeError(f'Failed to read {csv_file} after 10 retries (100s)')


    @property
    def is_master(self):
        return not dist.is_initialized() or dist.get_rank() == 0


    def index_videos(self):
        vid_list = self.vid_list
        if not self.multiple_datasets and Path(self.csv_file).stem.startswith('ucf'):
            actions = set()
            vid2action = {}
            for vid in vid_list:
                video_name = Path(vid).stem
                assert video_name.startswith('v_')
                action = video_name.split('_')[1]
                actions.add(action)
                vid2action[vid] = action

            actions = sorted(list(actions))
            assert len(actions) == 101, f'UCF101 has 101 classes, but got {len(actions)} classes'
            self.num_classes = len(actions)
            self.label2action = {i: actions[i] for i in range(len(actions))}
            self.action2label = {actions[i]: i for i in range(len(actions))}
            self.vid2label = {vid: self.action2label[vid2action[vid]] for vid in vid_list}

        if self.use_all_frames:
            preloaded = '_preloaded' if self.pre_load else ''
            cache_name = f'{self.csv_file}_{self.frame_num}_all_frames_step{self.use_all_frames_step}{preloaded}.pkl'
            cache_path = os.path.join(self.index_map_cache_dir, cache_name)

            if dist.is_initialized():
                dist.barrier()

            if self.is_master:
                if os.path.exists(cache_path):
                    print(f'Loading index map from {cache_path}')
                    with open(cache_path, 'rb') as f:
                        cached = pickle.load(f)
                        self.idx2label = cached['idx2label']
                        self.index_map = cached['index_map']
                else:
                    self.idx2label = dict()
                    index_map = {}
                    video_len = {}
                    index = 0
                    for vid in tqdm(vid_list, desc='Indexing videos'):
                        vr = decord.VideoReader(vid)
                        video_len[vid] = len(vr)
                        for start_idx in range(0, len(vr), self.use_all_frames_step):
                            if start_idx + self.frame_num <= len(vr):
                                index_map[index] = (vid, start_idx, start_idx + self.frame_num)
                                self.idx2label[index] = self.vid2label[vid]
                                index += 1
                    self.index_map = index_map

                    cached = {'idx2label': self.idx2label, 'index_map': self.index_map, 'tot_frame': video_len}
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cached, f)

            if dist.is_initialized():
                dist.barrier()

            if not self.is_master:
                assert os.path.exists(cache_path), f'Failed to find {cache_path}'
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.idx2label = cached['idx2label']
                    self.index_map = cached['index_map']

        else:
            self.idx2label = {i: self.vid2label[vid] for i, vid in enumerate(vid_list)}

        if self.num_classes is not None:
            all_labels = list(self.idx2label.values())
            try:
                self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
            except:
                self.label_count = None

            assert set(all_labels) == set(
                range(self.num_classes)
            ), f'Labels should be 0-{self.num_classes-1}, but got {set(all_labels)}'

            # count the number each unique label, store in a dictionary
            self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
        else:
            self.label_count = None

        self.fname2csvid = dict(zip([os.path.basename(i) for i in self.vid_list], range(len(self.vid_list))))

    def __len__(self):
        if self.ar_ann_dir is not None:
            length = len(self.file_list)
        elif self.use_all_frames:
            length = len(self.index_map)
        else:
            length = len(self.vid_list)
        
        return length

    def get_video_batch_from_disk(self, idx):
        if self.fake:
            return torch.randint(0, 256, (self.frame_num, self.crop_size, self.crop_size, 3), dtype=torch.uint8), 'fake_path'

        if self.use_all_frames:
            vid, start, end = self.index_map[idx]
            vr = decord.VideoReader(vid)
            frame_idx = list(range(start, end))
            path = vid
        else:
            # Loading video
            vr = read_video_with_retry(self.vid_list[idx])
            frame_num = min(self.frame_num, len(vr))
            if self.frame_rate == 'uniform':
                frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
            elif self.frame_rate == 'native':
                starting_idx = np.random.randint(0, len(vr) - frame_num + 1)
                frame_idx = list(range(starting_idx, starting_idx + frame_num))
            elif self.frame_rate == 'fixed':
                frame_idx = list(range(frame_num))
            else:
                raise ValueError(f'Unknown frame_rate setting: {self.frame_rate}')
            path = self.vid_list[idx]

        video = vr.get_batch(frame_idx)
        return video, path, frame_idx

    def get_ann_batch_from_annotations(self, idx):
        if self.ar_ann_path is not None:
            fname = os.path.basename(self.vid_list[idx])
            if self.frame_rate == 'native':
                this_idx = random.choice(self.fname2id[fname])
            elif self.frame_rate == 'fixed':
                this_idx = 0
            else:
                raise NotImplementedError
            latent_nums = self.ar_annotations['latent_nums'][this_idx]
            bottleneck_rep = self.ar_annotations['bottleneck_rep'][this_idx]
            label = self.idx2label[idx] if isinstance(self.idx2label[idx], int) else -1
            
            data = {'latent_nums': latent_nums, 'bottleneck_rep': bottleneck_rep, 'label': label}

            if 'conditions' in self.ar_annotations.keys():
                data['conditions'] = self.ar_annotations['conditions'][this_idx]
        elif self.ar_ann_dir is not None:
            csvid = self.fname2csvid[os.path.basename(self.file_list[idx]).replace('.pt', '.mp4')]
            ar_annotations = torch.load(os.path.join(self.ar_ann_dir, self.file_list[idx]))
            if self.frame_rate == 'native':
                this_idx = random.randint(0, len(ar_annotations['id']) - 1)
            elif self.frame_rate == 'fixed':
                this_idx = [idx for idx, fname in enumerate(ar_annotations['id']) if fname.startswith('0_')]
                this_idx = this_idx[0] if this_idx else 0
            else:
                raise NotImplementedError
            pass

            data = {k: v[this_idx] for k, v in ar_annotations.items() if k != 'id'}
            data['label'] = self.idx2label[csvid] if isinstance(self.idx2label[csvid], int) else -1

        return data


    def get_images(self, idx):
        path = self.vid_list[idx]
        img = Image.open(path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = self.cur_tfm(img_tensor)
        img_tensor = img_tensor.unsqueeze(1).repeat(1, self.frame_num, 1, 1)
        
        return img_tensor, path


    def __getitem__(self, idx):
        if self.ar_ann_path is not None or self.ar_ann_dir is not None:
            data = self.get_ann_batch_from_annotations(idx)
        elif self.is_image:
            img, path = self.get_images(idx)
            data = {'gt': img, 'path': path}
        else:
            video, path, frame_idx = self.get_video_batch_from_disk(idx)

            if self.augment is not None:
                video = self.augment(video.permute(0,-1,1,2)).permute(0,2,3,1)
            video = video.permute(-1,0,1,2).float() / 255. # # T,H,W,C -> C,T,H,W

            video_data = self.cur_tfm(video)
            if video.shape[1] < self.frame_num:
                video_data = F.pad(video_data, (0,0,0,0,0,self.frame_num-video.shape[1]), mode='replicate')

            label = self.idx2label[idx] if isinstance(self.idx2label[idx], int) else -1
            
            data = {'gt': video_data, 'path': path, 'label': label, 'frame_start': frame_idx[0], 'frame_end': frame_idx[-1]}

            if self.scores is not None and self.with_scores:
                sid = self.scores['name2idx'][os.path.basename(path)]
                scores = (self.scores[self.score_type][sid] - self.score_mean) / self.score_std
                data.update(scores=scores)

        return data


