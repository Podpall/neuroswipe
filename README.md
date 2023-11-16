# Neuroswipe
Solution for neuroswipe competition from Yandex ML Cup 2023

## Description

This solution implements sequence2sequence approach to translate input sequence of key hits into the letters of output word.
It is implemented in two variations, which are later combined:

### Simple case
Input data is represented as a sequence of letters. For example:

`е е е к а а а с с с м м м м м м м м и и т т ь б б б`,

which should be translated into

`е с т ь`

### Split into rectangles case
In this case input is modified to include more refined information about keyboard hits. Each key is split into given number of rectangles and the number of each rectangle follows letter in the input sequence. For example, if letter `a` was hit in the top left corner then we add `a 1` to the sequence. If number of rectangles to split each key is 5 and botton right corner of letter `a` is hit then we add `a 25` to the sequnce. Full example: 

`е 13 е 12 е 15 к 24 а 8 а 13 а 23 с 3 с 9 с 14 м 10 м 15 м 15 м 16 м 16 м 16 м 17 м 14 и 11 и 8 т 5 т 13 ь 11 б 10 б 12 б 12`

should be translated into

`е с т ь`

## Steps to run

1. Download competition data into some folder (further assuming it is `data` in the root of the repository).

2. Preprocess train, validation, and test data to make them suitable for seq2seq task.
To make data for [simple training case](#simple-case):
```
python preprocess_data.py --input_file=./data/train.jsonl --output_file=./data/train_processed.csv
python preprocess_data.py --input_file=./data/valid.jsonl --output_file=./data/valid_processed.csv --ref_file=./data/valid.ref
python preprocess_data.py --input_file=./data/test.jsonl --output_file=./data/test_processed.csv
```
Same for [rectangles case](#split-into-rectangles-case) can be done with additional option `--n_split=5`.

3. Train seq2seq `rut5-base` models:
```
python train_seq2seq.py experiment=train_base_t5
python train_seq2seq.py experiment=train_base_t5_rectangles
```
Alternatively, t5 model can be trained from scratch:
```
python train_seq2seq.py experiment=train_t5_from_scratch
```

4. Make predictions:
```
python make_predictions.py --model_path=./swipe_t5_base/your_checkpoint --data_path=./data/test_processed.csv --out_dir=./preds
python make_predictions.py --model_path=./swipe_t5_base_rectangles/your_checkpoint --data_path=./data/test_rectangle_processed.csv --out_dir=./preds_rectangle
```

5. Combine predictions:
```
python combine_predictions.py --reg_dir=./preds --rect_dir=./preds_rectangle --json_path=./data/test.jsonl
```
