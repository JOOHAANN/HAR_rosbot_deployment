#!/usr/bin/env python3
"""Evaluate preserved checkpoints without training or overwriting the reference."""
import argparse, json, os, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=a.root.resolve(); vpo=root/'ws/VPOCLIP_plus_full'
os.chdir(vpo); sys.path.insert(0,str(vpo))
import torch
from rl.dual_robot_depth_ddqn_v1 import experiment as e
run=vpo/'work_dir/dual_robot_depth_ddqn_strict45_5'
cfg=json.loads((run/'config_resolved.json').read_text())
device=torch.device('cuda:0')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
depth=e.load_depth_lookup(Path(cfg['depth_root']))
raw,audit=e.load_raw(Path(cfg['cache_root']),Path(cfg['track_root']),'test',device,depth)
gate=e.fusion_gate.load_gate(Path(cfg['gate_checkpoint']),device)
policies={v:e.load_policy(Path(c),v,device) for v,c in cfg['selected_checkpoints'].items()}
with torch.inference_mode():
    fused=e.pair_fused_logits(raw,gate)
    rows=[]; singles=[]
    bank=cfg['final_unseen_test_classes']
    ids=e.eligible_episodes(raw,bank).nonzero().flatten()
    bank_t=torch.tensor(bank,device=device)
    for seed in cfg['eval_seeds']:
        rows.append(e.test_seed_evaluation(raw,fused,policies,cfg['variants'],seed,bank))
        start,_=e.sample_starts(raw,ids,seed)
        pred=bank_t[raw.logits[ids,start][:,bank_t].argmax(-1)]
        singles.append(float((pred==raw.labels[ids]).float().mean()))
        print('evaluated',seed,flush=True)
summary=e.summarize_seed_results(rows)
reference=json.loads((run/'test/summary.json').read_text())
delta={v:{k:summary[v][k]['mean']-reference[v][k]['mean'] for k in ['accuracy','mean_movement_cost']} for v in summary}
a.output.mkdir(parents=True,exist_ok=True)
for name,data in [('summary.json',summary),('per_seed.json',rows),('reference_delta.json',delta),('single_view.json',e.ci_summary(singles))]:
    (a.output/name).write_text(json.dumps(data,indent=2))
print(json.dumps({'single_view':e.ci_summary(singles),'reference_delta':delta},indent=2))
