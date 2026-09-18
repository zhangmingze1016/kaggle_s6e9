"""Sequential full-CV candidates followed by OOF-guided blend selection."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

from utils.cv_runner import parse_args, run_cv
from training.ensemble import main as ensemble_main


CANDIDATES = {
    'multiscale_lgb': ('LightGBM', ['--preset','strong']),
    'notebook_lgb': ('LightGBM', ['--learning-rate','.02','--n-estimators','20000',
                                  '--early-stopping-rounds','300']),
    'multiscale_xgb': ('XGBoost', ['--preset','strong']),
    'multiscale_cb': ('CatBoost', ['--preset','strong']),
}


class Tee:
    def __init__(self,*streams):
        self.streams=streams
    def write(self,text):
        for stream in self.streams:
            stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',nargs='+',choices=list(CANDIDATES),default=list(CANDIDATES))
    parser.add_argument('--n-splits',type=int,default=5)
    parser.add_argument('--n-jobs',type=int,default=4)
    parser.add_argument('--random-seed',type=int,default=42)
    parser.add_argument('--sample-size',type=int)
    parser.add_argument('--max-iterations',type=int,help='Override for development smoke runs')
    parser.add_argument('--train-path',default='data/train.csv')
    parser.add_argument('--test-path',default='data/test.csv')
    parser.add_argument('--output-dir',default='predictions')
    parser.add_argument('--artifact-dir',default='artifacts/predictions')
    parser.add_argument('--experiment-file',default='experiments.csv')
    parser.add_argument('--run-root',default='artifacts/suite')
    a=parser.parse_args(argv)
    root=Path(a.run_root);root.mkdir(parents=True,exist_ok=True)
    records=[];started=time.time()
    status={'started_at':started,'configuration':vars(a),'completed':records}
    def write_status():
        temp=root/'suite_status.tmp.json';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'suite_status.json')
    for candidate in a.candidates:
        name,flags=CANDIDATES[candidate]
        options=flags+['--n-splits',str(a.n_splits),'--n-jobs',str(a.n_jobs),
            '--random-seed',str(a.random_seed),'--train-path',a.train_path,
            '--test-path',a.test_path,'--output-dir',a.output_dir,
            '--artifact-dir',a.artifact_dir,
            '--experiment-file',a.experiment_file,'--run-root',str(root/candidate)]
        if a.sample_size:
            options+=['--sample-size',str(a.sample_size)]
        if a.max_iterations:
            options+=['--iterations' if name=='CatBoost' else '--n-estimators',str(a.max_iterations)]
        status.update(state='training',current_candidate=candidate);write_status()
        print(f'\n===== CANDIDATE {candidate} =====',flush=True)
        try:
            with (root/f'{candidate}.log').open('a') as log:
                with contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
                    result=run_cv(name,parse_args(name,options))
            records.append({'candidate':candidate,**result})
            write_status()
        except Exception as exc:
            status.update(state='failed',error=repr(exc));write_status();raise
    if len(records)>=2:
        status.update(state='blending');write_status()
        result=ensemble_main([r['bundle'] for r in records]+[
            '--output-dir',a.output_dir,'--artifact-dir',a.artifact_dir,
            '--experiment-file',a.experiment_file])
        status['selected_submission']=str(result)
    else:
        status['selected_submission']=records[0]['submission']
    status.update(state='complete',elapsed_seconds=time.time()-started)
    write_status()
    print(f'Suite complete: {status["selected_submission"]}',flush=True)


if __name__=='__main__':
    main()
