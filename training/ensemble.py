"""Compare single models, probability blends and rank blends using aligned OOF."""
import argparse
import json
from pathlib import Path

from utils.ensemble import load_bundles, choose_blend
from utils.prediction_saver import PredictionSaver


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundles',nargs='+',help='OOF NPZ bundles from artifacts/predictions')
    p.add_argument('--output-dir',default='predictions')
    p.add_argument('--artifact-dir',default='artifacts/predictions')
    p.add_argument('--experiment-file',default='experiments.csv')
    p.add_argument('--seed',type=int,default=20260917)
    a=p.parse_args(argv)
    bundles=load_bundles(a.bundles)
    oof,test,report=choose_blend(bundles,a.seed)
    report['sources']=a.bundles
    ref=bundles[0]
    params={'model':'Ensemble','blend_mode':report['mode'],
        'weights':json.dumps(report['weights']),'sources':json.dumps(a.bundles),
        'audit_auc':report['audit_auc'],'selection_auc':report['selection_auc'],
        'score_kind':'OOF after blend selection; not nested CV'}
    saver=PredictionSaver(a.output_dir,a.experiment_file,a.artifact_dir)
    path=saver.save(ref['test_ids'],test,
        report['full_oof_auc_after_selection'],params,train_ids=ref['train_ids'],
        y_true=ref['y_true'],oof_predictions=oof,fold_ids=ref['fold_ids'])
    saver.artifact_path(path,'.blend.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return path


if __name__=='__main__':
    main()
