from typing import Dict

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Trainer, T5Config, T5TokenizerFast,
    BertTokenizerFast,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PretrainedConfig
)


class BaseSwipeDataset:
    def __init__(
        self, path: str,
        tokenizer: PreTrainedTokenizer,
        shuffle: bool=False,
        max_length: int = 512,
        from_scratch: bool = False
    ):
        super().__init__()
        self._load_data(path, tokenizer, shuffle)
        self.max_length = max_length
        self.from_scratch = from_scratch

    def _load_data(self, path: str, tokenizer: PreTrainedTokenizer, shuffle: bool=False):
        print('Reading data')
        self.data = pd.read_csv(path).dropna()
        print('Dataframe successfully loaded')
        if shuffle:
            print('Shuffling data')
            self.data = self.data.sample(frac=1, random_state=0)
        self.tokenizer = tokenizer

    def preprocess_function(
        self,
        item: pd.Series,
    ) -> Dict[str, torch.Tensor]:
        out = self.tokenizer(
            item.input_sequence, truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False
        )
        if self.from_scratch:
            out['input_ids'] = out.input_ids[1:]  # skip cls token
            out['attention_mask'] = out.attention_mask[1:]  # skip cls token
            out['labels'] = self.tokenizer(item.output_sequence, truncation=True, max_length=self.max_length).input_ids[1:]
        else:
            out['labels'] = self.tokenizer(item.output_sequence, truncation=True, max_length=self.max_length).input_ids
        return out


class FixedSwipeDataset(Dataset, BaseSwipeDataset):
    def __getitem__(self, index):
        return self.preprocess_function(self.data.iloc[index])

    def __len__(self):
        return len(self.data)


class IterableSwipeDataset(IterableDataset, BaseSwipeDataset):
    def __iter__(self):
        for _, item in self.data.iterrows():
            yield self.preprocess_function(item)


class Fitter:
    def __init__(
        self,
        train_data_path: str,
        valid_data_path: str,
        model_name_or_path: str = None,
        vocab_path: str = None,
        from_scratch_config: PretrainedConfig = None,
        max_seq_length: int = 512,
        training_arguments: TrainingArguments = None,
    ):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.model_name_or_path = model_name_or_path
        self.vocab_path = vocab_path
        self.from_scratch_config = from_scratch_config
        self.max_seq_length = max_seq_length
        self.training_arguments = training_arguments or TrainingArguments(
            output_dir="./swipe_t5_base",
            learning_rate=1e-4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            max_steps=1_875_000,
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            warmup_steps=1000,
            save_total_limit=2,
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            dataloader_num_workers=0,
            gradient_checkpointing=True,
        )

    def train(self):
        from_scratch = self.model_name_or_path is None
        if from_scratch:
            tokenizer = BertTokenizerFast(vocab_file=self.vocab_path)
        else:
            tokenizer = T5TokenizerFast.from_pretrained(self.model_name_or_path)

        train_dataset = IterableSwipeDataset(
            path=self.train_data_path,
            tokenizer=tokenizer,
            shuffle=True,
            max_length=self.max_seq_length,
            from_scratch=from_scratch
        )
        valid_dataset = FixedSwipeDataset(
            path=self.valid_data_path,
            tokenizer=tokenizer,
            shuffle=False,
            max_length=self.max_seq_length,
            from_scratch=from_scratch
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=self.max_seq_length)

        print('Making model')
        if from_scratch:
            from_scratch_config = self.from_scratch_config(vocab_size=tokenizer.vocab_size)
            model = AutoModelForSeq2SeqLM.from_config(from_scratch_config)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        model.cuda()
        training_args = self.training_arguments

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        random_res = trainer.evaluate()
        print(random_res)
        print('Training')
        trainer.train()


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    fitter = hydra.utils.instantiate(cfg.experiment.fitter)
    print(fitter)
    fitter.train()


if __name__ == '__main__':
    main()
