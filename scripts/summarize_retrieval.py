import argparse
import json
import pandas as pd
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('dir_name')
  parser.add_argument('--metric', default='ndcg_at_10')
  return parser.parse_args()

def main():
  args = setup_args()
  logging.info(f'args: {args}')

  results_dir = Path(args.dir_name)

  scores_dict = {}

  for json_path in results_dir.glob('*.json'):
    key = json_path.stem
    with open(json_path) as f:
      data = json.load(f)
      if 'test' in data:
        json_data = data['test']
      else:
        json_data = data['dev']
      scores_dict[key] = json_data[args.metric]
  
  cols = []
  cols.append(f'Avg {args.metric}')
  for key in scores_dict:
    cols.append(key)
  
  row = [
    scores_dict[key]
    for key in scores_dict
  ]
  row.insert(0, f'{np.mean(row):.4f}')
  df = pd.DataFrame([row], columns=cols)
  df.to_csv(results_dir / 'summary.csv', index=False)

if __name__ == '__main__':
  main()