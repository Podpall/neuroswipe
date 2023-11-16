from argparse import ArgumentParser
import os
import pickle
from typing import List, Optional

import pandas as pd
import jsonlines
from tslearn.metrics import dtw as ts_dtw
import numpy as np
import typing as tp
from dataclasses import dataclass, field
from tqdm.auto import tqdm


############################################
# Code from baseline provided by organizers
############################################
@dataclass
class Hitbox:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    cx: int = field(init=False)
    cy: int = field(init=False)

    def __post_init__(self):
        self.cx, self.cy = (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def is_in(self, x, y):
        return self.left <= x < self.right and self.top <= y < self.bottom

    @property
    def central_coords(self):
        return [self.cx, self.cy]

class Grid:
    def __init__(self, grid_info: tp.Dict):
        self.hitboxes = self._build_hitboxes(grid_info)

    def get_the_nearest_hitbox(self, x, y) -> str:
        for label, hitbox in self.hitboxes.items():
            if hitbox.is_in(x, y):
                return label
        return 'a'

    def get_centered_curve(self, word: str) -> np.ndarray:
        curve = []
        for idx, l in enumerate(word):
            if l not in self.hitboxes:
                continue
            curve.append(self.hitboxes[l].central_coords)
        return np.array(curve, dtype=np.float32)

    @staticmethod
    def _build_hitboxes(grid_info: tp.Dict) -> tp.Dict[str, Hitbox]:
        def hitbox_from_key(key) -> tp.Optional[tp.Tuple[str, Hitbox]]:
            h_name = key.get('label') or key.get('action')
            if h_name is None or len(h_name) > 1:
                return None, None

            h = key['hitbox']
            x, y, w, h = h['x'], h['y'], h['w'], h['h']

            return h_name, Hitbox(top=y, bottom=y + h, left=x, right=x + w)

        hitboxes = {h_name: hitbox for (h_name, hitbox) in map(hitbox_from_key, grid_info['keys']) if h_name}
        return hitboxes

############################################


def compute_metrics(preds: List[List[str]], labels: List[str], return_per_line=False):
    weights = (1., 0.1, 0.09, 0.08)
    swipe_mrr = [0.] * len(labels)
    for i, (pred, label) in enumerate(zip(preds, labels)):
        for weight, cand in zip(weights, pred):
            if cand == label:
                swipe_mrr[i] = weight
                break
    if return_per_line:
        return swipe_mrr
    return sum(swipe_mrr) / len(labels)


class Combiner:
    def __init__(
        self,
        json_data_path: str,
        voc_path: str = './data/voc.txt'
    ):
        self.data = []
        with jsonlines.open(json_data_path) as reader:
            for obj in reader:
                self.data.append(obj)
        with open(voc_path, encoding='utf-8') as f:
            self.vocab = set([line.strip() for line in f.readlines()])
        self.list_vocab = list(self.vocab)

    def combine_cands(self, cands_reg: List[List[str]], cands_rect: List[List[str]]) -> List[List[str]]:
        preds = []
        for i, (reg, rect) in tqdm(enumerate(zip(cands_reg, cands_rect))):
            grid = Grid(self.data[i]['curve']['grid'])
            pred = []
            for cand_reg, cand_rect in zip(reg, rect):
                if cand_reg in self.vocab and cand_reg not in pred and cand_reg in rect:
                    pred.append(cand_reg)
                if len(pred) >= 4:
                    pred = pred[:4]
                    break
            if len(pred) < 4:
                for cand_reg, cand_rect in zip(reg, rect):
                    if cand_reg in self.vocab and cand_reg not in pred:
                        pred.append(cand_reg)
                    if cand_rect in self.vocab and cand_rect not in pred:
                        pred.append(cand_rect)
            pred = pred[:4]
            
            if len(pred) < 4:
                main_cand = 'а'
                for cand in reg:
                    if cand not in pred:
                        main_cand = cand
                        break
                cand_curve = grid.get_centered_curve(main_cand)
                dists = []
                for word in self.list_vocab:
                    if len(main_cand) > 0 and word[0] != main_cand[0]:
                        dists.append(999.)
                        continue
                    cur_dtw = ts_dtw(cand_curve, grid.get_centered_curve(word))
                    dists.append(cur_dtw)
                argsorted = np.argsort(dists)
                n = 4 - len(pred)
                pred += [self.list_vocab[i] for i in argsorted[:n]]
            if len(pred) < 4:
                print(pred)
            preds.append(pred)
        return preds


def combine(
    regular_preds_dir: str,
    rectangle_preds_dir: str,
    json_data_path: str,
    voc_path: Optional[str] = './data/voc.txt',
    valid_df_path: Optional[str] = None,
    out_file: Optional[str] = 'comb_preds.csv',
):

    raw_preds_path = os.path.join(regular_preds_dir, 'raw_cands.pkl')
    with open(raw_preds_path, 'rb') as f:
        cands = pickle.load(f)

    raw_preds_rectangle_path = os.path.join(rectangle_preds_dir, 'raw_cands.pkl')
    with open(raw_preds_rectangle_path, 'rb') as f:
        cands_rectangle = pickle.load(f)

    combiner = Combiner(json_data_path=json_data_path, voc_path=voc_path)
    preds = combiner.combine_cands(cands, cands_rectangle)

    if valid_df_path is not None:
        df = pd.read_csv(valid_df_path)
        labels = [seq.replace(' ', '') for seq in df.output_sequence.values]

        reg_preds_path = os.path.join(regular_preds_dir, 'preds.pkl')
        with open(reg_preds_path, 'rb') as f:
            reg_preds = pickle.load(f)
        reg_swipe_mrr = compute_metrics(reg_preds, labels)
        print('Regular model swipe mrr:', reg_swipe_mrr)

        rect_preds_path = os.path.join(rectangle_preds_dir, 'preds.pkl')
        with open(rect_preds_path, 'rb') as f:
            rect_preds = pickle.load(f)
        rect_swipe_mrr = compute_metrics(rect_preds, labels)
        print('Split-rectangles model swipe mrr:', rect_swipe_mrr)
        
        comb_swipe_mrr = compute_metrics(preds, labels)
        print('Combination swipe mrr:', comb_swipe_mrr)

    preds = pd.DataFrame(preds)
    preds.to_csv(out_file, index=False, header=None)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--reg_dir', type=str, default='./preds')
    argparser.add_argument('--rect_dir', type=str, default='./preds_rectangle')
    argparser.add_argument('--json_path', type=str, default='./data/test.jsonl')
    argparser.add_argument('--voc_path', type=str, default='./data/voc.txt')
    argparser.add_argument('--valid_df_path', type=str, default=None)
    argparser.add_argument('--out_file', type=str, default='comb_preds.csv')
    args = argparser.parse_args()    
    combine(
        regular_preds_dir=args.reg_dir,
        rectangle_preds_dir=args.rect_dir,
        json_data_path=args.json_path,
        voc_path=args.voc_path,
        valid_df_path=args.valid_df_path,
        out_file=args.out_file,
    )
