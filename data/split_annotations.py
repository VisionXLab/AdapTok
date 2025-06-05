import os
import copy
import argparse
import torch
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def save_sample(id_str, sample):
    save_path = os.path.join(output_dir, id_str.replace("/", "_").replace(".mp4", ".pt"))
    if os.path.exists(save_path):
        cache = torch.load(save_path)
        new_sample = {}
        for k in cache.keys():
            if k == 'id':
                new_sample[k] = cache[k] + sample[k]
            else:
                new_sample[k] = torch.cat([cache[k], sample[k]], dim=0)
        torch.save(new_sample, save_path)
    else:
        torch.save(sample, save_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_pattern', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    
    ann_paths = glob(args.ann_pattern)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    ann_paths.sort()

    for f_idx, ann_path in tqdm(enumerate(ann_paths)):
        annotations = torch.load(ann_path)
        id_list = annotations["id"]
        
        fname2id = defaultdict(list)
        for idx, fname in enumerate(id_list):
            name = fname.split('frames__')[-1]
            fname2id[name].append(idx)

        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = []
            for fname, ids in tqdm(fname2id.items(), total=len(fname2id), desc="Submitting tasks"):
                sample = {k: copy.deepcopy(annotations[k][ids]) for k in annotations if k != 'id'}
                sample['id'] = [annotations["id"][i] for i in ids]
                futures.append(executor.submit(save_sample, fname, sample))

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Writing files"):
                pass
