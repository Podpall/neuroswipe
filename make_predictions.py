from argparse import ArgumentParser
from typing import List, Optional, Tuple
import os
import pickle

from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class Predictor:
    def __init__(
        self,
        model_path: str,
        max_input_seq_length: int = 512,
        max_out_seq_length: int = 30,
        n_cands: int = 20,
        voc_path: str = './data/voc.txt',
        batch_size: int = 32,
        checkpoints_paths_for_averaging: Optional[List[str]] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to('cuda')
        self.max_input_seq_length = max_input_seq_length
        self.max_out_seq_length = max_out_seq_length
        self.n_cands = n_cands
        with open(voc_path, encoding='utf-8') as f:
            self.vocab = set([line.strip() for line in f.readlines()])
        self.batch_size = batch_size
        
        if checkpoints_paths_for_averaging is not None:
            state_dict = dict()
            for idx, dir_path in enumerate(checkpoints_paths_for_averaging):
                ckpt_path = os.path.join(dir_path, 'pytorch_model.bin')
                print(ckpt_path)
                current_state = torch.load(ckpt_path, map_location='cpu')
                for key, value in current_state.items():
                    if idx == 0:
                        state_dict[key] = value
                        continue
                    state_dict[key] = state_dict[key] + value
            state_dict = {key: value/len(checkpoints_paths_for_averaging) for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def batch_pred(self, rows: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        preds, raw_cands = [], []
        dataloader = DataLoader(rows, batch_size=self.batch_size)
        for batch in tqdm(dataloader):
            inputs = self.tokenizer(
                batch, return_tensors='pt',
                max_length=self.max_input_seq_length,
                truncation=True, padding=True, return_token_type_ids=False
            ).to('cuda')
            with torch.autocast(device_type='cuda'):
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_out_seq_length,
                    num_beams=self.n_cands, num_return_sequences=self.n_cands
                )
            cands = [cand.replace(' ', '') for cand in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            r_preds, r_raw_cands = [], []
            skipped_cands = []
            for i, cand in enumerate(cands):
                r_raw_cands.append(cand)
                if len(r_preds) < 4:
                    if cand in self.vocab:
                        r_preds.append(cand)
                    else:
                        skipped_cands.append(cand)
                if (i + 1) % self.n_cands == 0:
                    while len(r_preds) < 4:
                        r_preds.append('None')
                    preds.append(r_preds)
                    raw_cands.append(r_raw_cands)
                    r_preds, r_raw_cands, skipped_cands = [], [], []
        return preds, raw_cands

    def predict(self, file_path: str, output_dir: str):
        df = pd.read_csv(file_path)
        rows = df.input_sequence.values
        os.makedirs(output_dir, exist_ok=True)
        preds, raw_cands = self.batch_pred(rows)
        with open(os.path.join(output_dir, 'preds.pkl'), 'wb') as f:
            pickle.dump(preds, f)
        with open(os.path.join(output_dir, 'raw_cands.pkl'), 'wb') as f:
            pickle.dump(raw_cands, f)


if __name__ == '__main__':
    argparser = ArgumentParser()    
    argparser.add_argument('--model_path', type=str, required=True)
    argparser.add_argument('--inp_len', type=int, default=512)
    argparser.add_argument('--out_len', type=int, default=30)
    argparser.add_argument('--n_cands', type=int, default=20)
    argparser.add_argument('--voc_path', type=str, default='./data/voc.txt')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--ckpts_dir', type=str, default=None)
    argparser.add_argument('--data_path', type=str, required=True)
    argparser.add_argument('--out_dir', type=str, default='./preds')
    args = argparser.parse_args()
    ckpt_dirs = None
    if args.ckpts_dir is not None:
        ckpt_dirs = os.listdir(args.ckpts_dir)
        ckpt_dirs = [os.path.join(args.ckpts_dir, dir_path) for dir_path in ckpt_dirs]
    predictor = Predictor(
        model_path=args.model_path,
        max_input_seq_length=args.inp_len,
        max_out_seq_length=args.out_len,
        n_cands=args.n_cands,
        voc_path=args.voc_path,
        batch_size=args.batch_size,
        checkpoints_paths_for_averaging=ckpt_dirs,
    )
    predictor.predict(
        file_path=args.data_path,
        output_dir=args.out_dir,
    )
