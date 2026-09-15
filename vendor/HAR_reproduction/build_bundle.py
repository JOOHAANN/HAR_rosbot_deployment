#!/usr/bin/env python3
"""Read-only source export; excludes dataset arrays/media and unrelated weights."""
import hashlib, json, os, subprocess, tarfile, io
from pathlib import Path

HOME = Path('/home/youhan')
OUT = HOME / 'har_export'
PREFIX = 'HAR_reproduction'
VPO = HOME / 'ws/VPOCLIP_plus_full'
RUN = VPO / 'work_dir/dual_robot_depth_ddqn_strict45_5'
TEXT = {'.py','.pyi','.sh','.bash','.ps1','.bat','.yaml','.yml','.json','.jsonl','.toml','.ini','.cfg','.conf','.xml','.md','.rst','.txt','.csv','.tsv','.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.cmake','.ipynb','.tex','.cls','.bib','.js','.ts','.html','.css','.sql','.gitignore','.gitattributes'}
SKIP = {'.git','__pycache__','.pytest_cache','node_modules','.venv','venv','.mypy_cache'}
files = {}
def add(p, dest=None):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    files[dest or str(p.relative_to(HOME))] = p
def scan(root, weights=False, destroot=None):
    root = Path(root)
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in names:
            p = Path(directory)/name
            if p.is_file() and (p.suffix.lower() in TEXT or name in {'LICENSE','LICENSE.txt','Makefile','Dockerfile','requirements','setup.cfg'} or (weights and p.suffix.lower() in {'.pt','.pth','.onnx','.bin','.safetensors'})):
                add(p, str(Path(destroot)/p.relative_to(root)) if destroot else None)

for root in ['ws/VPOCLIP_plus_full','ws/CTR-GCN_17_full','ws/X3D_full','ws/yolov5_full','HAR']:
    scan(HOME/root)
for root in [RUN,VPO/'work_dir/fusion_lightweight_gating_strict45_5']:
    scan(root, weights=True)
mandatory = [
 VPO/'work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth',
 HOME/'ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt',
 HOME/'ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth',
 HOME/'ws/yolov5_full/best.pt',
 HOME/'.cache/clip/ViT-B-32.pt',
 HOME/'.cache/torch/hub/checkpoints/midas_v21_small_256.pt',
 HOME/'.cache/torch/hub/checkpoints/tf_efficientnet_lite3-b733e338.pth',
]
for p in mandatory: add(p)
for p in VPO.glob('*.xlsx'): add(p)
scan(HOME/'.cache/rtmlib', weights=True)
scan(HOME/'.cache/torch/hub', weights=False)
# Frozen language prototypes are model assets, not visual dataset caches.
for p in VPO.rglob('*'):
    if p.is_file() and p.suffix in {'.pt','.npy'} and ('text' in p.name or 'prototype' in p.name):
        add(p)
for p in [HOME/'ws/ETRI_subject_split_45_25_10_20.json',HOME/'ws/ETRI-Activity3D-RGB/low_view_camera_angles_split55.csv']:
    add(p)
for p in OUT.iterdir():
    if p.suffix in {'.py','.md'}: add(p,p.name)
scan(OUT/'verification', destroot='verification')
env = subprocess.check_output([str(HOME/'.conda/envs/clipgcn/bin/python'),'-m','pip','freeze'],text=True)
packages = json.loads(subprocess.check_output([str(HOME/'.conda/envs/clipgcn/bin/python'),'-m','pip','list','--format=json'],text=True))
portable = '\n'.join((next((s for s in env.splitlines() if s.startswith('clip @ git+')), 'clip=='+p['version']) if p['name'].lower()=='clip' else p['name']+'=='+p['version']) for p in packages)+'\n'
inventory=[]
archive=OUT/'HAR_reproduction_strict45_5.tar.gz'
def blob(tar,name,value):
    value=value.encode(); info=tarfile.TarInfo(PREFIX+'/'+name);info.size=len(value);tar.addfile(info,io.BytesIO(value))
with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
    for i,(dest,p) in enumerate(sorted(files.items())):
        sha=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
        inventory.append({'path':dest,'source':str(p),'bytes':p.stat().st_size,'sha256':sha.hexdigest()})
        tar.add(p,arcname=PREFIX+'/'+dest,recursive=False)
        if i%2000==0: print('PACKED',i,len(files),flush=True)
    for name,target in [('CTR-GCN_17','../ws/CTR-GCN_17_full'),('X3D','../ws/X3D_full')]:
        info=tarfile.TarInfo(PREFIX+'/HAR/'+name);info.type=tarfile.SYMTYPE;info.linkname=target;tar.addfile(info)
    blob(tar,'environment/pip-freeze.txt',env)
    blob(tar,'environment/requirements-portable.txt',portable)
    blob(tar,'MANIFEST.json',json.dumps(inventory,ensure_ascii=False,indent=2))
    blob(tar,'SHA256SUMS',''.join(f"{x['sha256']}  {x['path']}\n" for x in inventory))
print('ARCHIVE',archive,'FILES',len(inventory),'SOURCE_BYTES',sum(x['bytes'] for x in inventory),flush=True)
# Verify every archived file against its recorded hash, including weights.
expected={PREFIX+'/'+x['path']:x['sha256'] for x in inventory}
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        if member.name in expected:
            h=hashlib.sha256(); f=tar.extractfile(member)
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==expected.pop(member.name),member.name
assert not expected
print('VERIFIED_ALL_FILES',archive.stat().st_size,flush=True)
