"""Benchmark isolated processes, then merge pending-only shards with one writer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from extract_rtmpose_coco17 import load_or_create_parts, merge_dataset


def launch(args, root, names, labels, processes, split):
    jobs = []
    for rank, rows in enumerate(np.array_split(np.arange(len(names)), processes)):
        if not len(rows): continue
        folder = root / str(rank)
        folder.mkdir(parents=True)
        np.save(folder/f'{split}_sample_names.npy', names[rows])
        np.save(folder/f'{split}_labels.npy', labels[rows])
        np.save(folder/'source_rows.npy', rows)
        log = (folder/'worker.log').open('w')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        cmd = [sys.executable, '-u', str(Path(__file__).with_name('extract_rtmpose_coco17.py')),
               '--rgb-root', str(args.rgb_root), '--split-dir', str(folder),
               '--output-dir', str(folder), '--splits', split, '--skip-merge',
               '--decode-workers', str(max(2, 12//processes)), '--inference-workers', '2']
        jobs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), folder, rows, log))
    return jobs


def collect(jobs):
    try:
        for proc, folder, rows, log in jobs:
            code = proc.wait()
            log.close()
            if code:
                raise RuntimeError(f'Worker failed ({code}): {folder}/worker.log')
            if re.search(r'unreadable=[1-9]', (folder/'worker.log').read_text()):
                raise RuntimeError(f'Unreadable video in {folder}; refusing silent merge')
    finally:
        for proc, _, _, log in jobs:
            if proc.poll() is None: proc.terminate()
        for proc, _, _, log in jobs:
            proc.wait()
            log.close()


def checked(folder, split, names, labels):
    part = folder/'.rtmpose_parts'
    assert np.array_equal(np.load(part/f'{split}_sample_names.npy'), names)
    assert np.array_equal(np.load(part/f'{split}_labels.npy'), labels)
    assert np.load(part/f'{split}_done.npy').all()
    data = np.load(part/f'{split}_x.npy', mmap_mode='r')
    assert data.shape == (len(names), 3, 13, 17, 2)
    assert np.isfinite(data).all()
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--benchmark-only', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir/'parallel.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tuning = args.output_dir/'parallel_tuning.json'
    if not tuning.exists():
        names = np.load(args.split_dir/'train_sample_names.npy')[:96]
        labels = np.load(args.split_dir/'train_labels.npy')[:96]
        root = Path(tempfile.mkdtemp(prefix='benchmark_', dir=args.output_dir))
        results = []; baseline = None
        for processes in (1, 2, 3):
            start = time.monotonic()
            jobs = launch(args, root/str(processes), names, labels, processes, 'train')
            collect(jobs)
            seconds = time.monotonic()-start
            merged = np.empty((len(names),3,13,17,2),np.float32)
            for _, folder, rows, _ in jobs:
                merged[rows] = checked(folder,'train',names[rows],labels[rows])
            if baseline is None: baseline=merged.copy()
            else: np.testing.assert_allclose(merged, baseline, atol=1e-5, rtol=1e-5)
            row = dict(processes=processes, seconds=seconds, clips_per_second=len(names)/seconds)
            results.append(row)
            print('BENCH', json.dumps(row), flush=True)
        chosen = max(results, key=lambda r:r['clips_per_second'])['processes']
        tuning.write_text(json.dumps(dict(selected_processes=chosen, results=results,
                         benchmark_root=str(root), note='96 identical clips, startup included, outputs verified'),indent=2))
    chosen = json.loads(tuning.read_text())['selected_processes']
    print('SELECTED processes=', chosen, flush=True)
    if args.benchmark_only: return
    part = args.output_dir/'.rtmpose_parts'; part.mkdir(exist_ok=True)
    for split in ('train','val','test','dqn'):
        names = np.load(args.split_dir/f'{split}_sample_names.npy')
        labels = np.load(args.split_dir/f'{split}_labels.npy')
        clips, done = load_or_create_parts(part,split,len(names),13,names,labels)
        # Bounded waves retain completed work frequently; shards never write canonical data.
        pending = np.flatnonzero(~done)
        for start in range(0,len(pending),chosen*250):
            rows = pending[start:start+chosen*250]
            root = Path(tempfile.mkdtemp(prefix=f'shards_{split}_',dir=args.output_dir))
            np.save(root/'canonical_rows.npy',rows)
            begun=time.monotonic()
            jobs=launch(args,root,names[rows],labels[rows],chosen,split)
            collect(jobs)
            for _,folder,local,_ in jobs:
                data=checked(folder,split,names[rows[local]],labels[rows[local]])
                clips[rows[local]]=data
                clips.flush()
                done[rows[local]]=True
                done.flush()
            print(f'[{split}] {int(done.sum())}/{len(names)} complete; wave={len(rows)/(time.monotonic()-begun):.2f} clip/s; processes={chosen}',flush=True)
        assert done.all()
    merge_dataset(args.output_dir,part,('train','val','test','dqn'))


if __name__=='__main__': main()
