import os
from argparse import ArgumentParser
from typing import Dict, List, Optional

import jsonlines
from tqdm.auto import tqdm


class Preprocessor:
    def __init__(self, n_split_hitbox: Optional[int] = None):
        self.n = n_split_hitbox
        if self.n is not None:
            assert self.n > 1
        self.do_split = self.n is not None

    def split_into_rectangles(self, hitbox: Dict[str, float]) -> List[Dict[str, float]]:
        rectangles = []
        rect_width = hitbox['w'] / self.n
        rect_height = hitbox['h'] / self.n
        for row_idx in range(self.n):
            for col_idx in range(self.n):
                rect_hitbox = {'h': rect_height, 'w': rect_width,
                            'x': hitbox['x'] + col_idx * rect_width,
                            'y': hitbox['y'] + row_idx * rect_height}
                rectangles.append(rect_hitbox)
        return rectangles

    @staticmethod
    def is_hit(hitbox: Dict[str, float], x: float, y: float) -> bool:
        if y < hitbox['y'] or y >= hitbox['y'] + hitbox['h']:
            return False
        if x < hitbox['x'] or x >= hitbox['x'] + hitbox['w']:
            return False
        return True

    def build_row(self, obj) -> str:
        row = []
        for x, y in zip(obj['curve']['x'], obj['curve']['y']):
            for key in obj['curve']['grid']['keys']:
                if not self.is_hit(key['hitbox'], x, y):
                    continue
                label = '@'
                if 'label' in key:
                    label = key['label']
                else:
                    label = key['action']
                if label == ',':
                    label = '@'
                if self.do_split:
                    rectangles = self.split_into_rectangles(key['hitbox'])
                    for rect_idx, rect_hitbox in enumerate(rectangles):
                        found_rect = False
                        if self.is_hit(rect_hitbox, x, y):
                            label += f' {rect_idx}'
                            found_rect = True
                            break
                    assert found_rect == True
                row.append(label)
                break
        return ' '.join(row)

    def preprocess(
        self,
        input_file_path: str,
        output_file_path: str,
        ref_file_path: Optional[str] = None,
    ):
        if os.path.exists(output_file_path):
            raise ValueError("Output file already exists!")

        valid_labels = None
        if ref_file_path is not None:
            with open(ref_file_path, encoding='utf-8') as f:
                valid_labels = f.readlines()
        with jsonlines.open(input_file_path) as reader, open(output_file_path, 'a+', encoding='utf-8') as out_f:
            out_f.write(f"input_sequence,output_sequence\n")
            for idx, obj in tqdm(enumerate(reader)):
                row = self.build_row(obj)
                label = 'dummy\n'
                if 'word' in obj:
                    label = ' '.join(obj['word']) + '\n'
                elif valid_labels is not None:
                    label = ' '.join(valid_labels[idx])
                out_f.write(f"{row},{label}")


if __name__ == '__main__':
    argparser = ArgumentParser()    
    argparser.add_argument('--input_file', type=str, required=True)
    argparser.add_argument('--output_file', type=str, required=True)
    argparser.add_argument('--ref_file', type=str, default=None)
    argparser.add_argument('--n_split', type=int, default=None)
    args = argparser.parse_args()
    preprocessor = Preprocessor(args.n_split)
    preprocessor.preprocess(args.input_file, args.output_file, args.ref_file)
