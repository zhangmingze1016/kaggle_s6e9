"""Run a matched small-data baseline/window comparison; never promotes automatically."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from training.experiment_suite import Tee
from training.review_experiments import compare
from utils.cv_runner import parse_args, run_cv
from utils.ensemble import load_bundles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size',type=int,default=60000)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--n-jobs',type=int,default=4)
    args=parser.parse_args()
    if not 100 <= args.sample_size <= 100000 or args.iterations < 1:
        parser.error('Screen sample must be 100–100000 rows; iterations must be positive')
    root=Path('artifacts/window_screen'); root.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,extra in [('baseline',[]),('windows',['--local-windows'])]:
        options=['--preset','strong','--sample-size',str(args.sample_size),
                 '--n-estimators',str(args.iterations),'--n-jobs',str(args.n_jobs),
                 '--output-dir',str(root/'submissions'),'--artifact-dir',str(root/'bundles'),
                 '--experiment-file',str(root/'experiments.csv'),'--run-root',str(root/name)]+extra
        with (root/f'{name}.log').open('a') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
                results.append(run_cv('LightGBM',parse_args('LightGBM',options)))
    baseline,candidate=load_bundles([r['bundle'] for r in results])
    report=compare(baseline,candidate)
    report.update(configuration=vars(args),baseline_auc=results[0]['oof_auc'],
                  fixed_half_blend_auc=float(roc_auc_score(baseline['y_true'],
                      .5*(baseline['oof_pred']+candidate['oof_pred']))),
                  sources=[r['bundle'] for r in results],
                  note='Small-sample screening only. Same rows/folds in both arms; not comparable to full-data OOF. No weight search or automatic full training. Window density changes with sample size.')
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
