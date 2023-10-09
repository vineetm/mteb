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

MODEL_KEYS = [
  'TRECCOVID',	'NFCorpus',	'NQ',	'HotpotQA',	'FiQA2018',	'ArguAna',	
  'Touche2020',	'DBPedia',	'SCIDOCS', 'ClimateFEVER', 'FEVER', 'SciFact'
]

COLUMNS = ['Model', 'Avg NDCG@10']
COLUMNS.extend(MODEL_KEYS)

def main():
  args = setup_args()
  logging.info(f'args: {args}')

  results_dir = Path(args.dir_name)

  rows = []
  for sub_dir in results_dir.glob('*'):
    if not sub_dir.is_dir():
      continue

    scores_dict = {}
    model_key = sub_dir.stem
    logging.info(f'Processing {sub_dir} {model_key}')

    for json_path in sub_dir.glob('*.json'):
      test_key = json_path.stem
      with open(json_path) as f:
        data = json.load(f)
        if 'test' in data:
          json_data = data['test']
        else:
          json_data = data['dev']
        scores_dict[test_key] = json_data[args.metric]

    row = []
    for key in MODEL_KEYS:
      if key in scores_dict:
        row.append(scores_dict[key])
      else:
        row.append(0.)
    
    avg_score = np.mean(row)
    row.insert(0, model_key)
    row.insert(1, f'{avg_score:.4f}')
    rows.append(row)
    
  df = pd.DataFrame(rows, columns=COLUMNS)
  df.to_csv(results_dir / 'summary.csv', index=False)

if __name__ == '__main__':
  main()