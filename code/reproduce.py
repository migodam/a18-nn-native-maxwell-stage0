"""Reproduce offline outputs in a fresh child folder, preserving this delivery."""
import argparse,shutil,subprocess,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
arg=argparse.ArgumentParser();arg.add_argument('--output',required=True);arg.add_argument('--pdf',action='store_true');args=arg.parse_args()
out=Path(args.output).resolve();allowed=(ROOT/'reproductions').resolve()
if not out.is_relative_to(allowed) or out.exists():raise SystemExit('Output must be a new folder under A18_NN_NATIVE/reproductions')
out.mkdir(parents=True)
for folder in ['code','vendor','inputs']:
    shutil.copytree(ROOT/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__'))
for folder in ['results','figures','verification','literature','qa','manuscript']:(out/folder).mkdir()
for p in (ROOT/'verification').glob('*.py'):shutil.copyfile(p,out/'verification'/p.name)
shutil.copyfile(ROOT/'verification/REPAIR_LOG.jsonl',out/'verification/REPAIR_LOG.jsonl')
for p in (ROOT/'literature').glob('*'):
    if p.is_file():shutil.copyfile(p,out/'literature'/p.name)
for n in ['stage0_config.json','CONFIG_FREEZE_RECEIPT.json','PROTOCOL.md']:shutil.copyfile(ROOT/n,out/n)
cfg=json.loads((out/'stage0_config.json').read_text())
used=sum(json.loads(s)['cpu_s'] for s in (ROOT/'results/PROCESS_RESOURCE_RECEIPTS.jsonl').read_text().splitlines() if json.loads(s)['budget_category']=='numerical')
env=dict(os.environ,A18_GLOBAL_CPU_REMAINING=str(cfg['cpu_budget_s']-used))
rev=subprocess.check_output(['git','-C',cfg['source_root'],'rev-parse','HEAD'],text=True).strip()
if rev!=cfg['source_commit']:raise SystemExit('SOURCE_COMMIT_MISMATCH')
(out/'REPLAY_SCOPE.json').write_text(json.dumps({'source_commit':rev,'scope':'OFFLINE DERIVED OUTPUTS; no training or full-wave actions','archived_repair_log':'copied prior repair provenance; current replay attempts have fresh resource receipts'},indent=2)+'\n')
scripts=['verification/verify_stage0_algebra.py','code/run_stage0.py','code/analyze_stage0.py','code/bill_features.py','verification/check_interfaces.py','verification/check_source_integrity.py','code/build_report_assets.py','code/build_manuscripts.py']
if args.pdf:scripts.append('code/export_pdfs.py')
for name in scripts:
    command=[sys.executable,str(out/'code/run_bounded.py'),str(out/name)]
    if name in ['code/build_report_assets.py','code/build_manuscripts.py','code/export_pdfs.py']:command+=['--category','authoring']
    subprocess.run(command,check=True,env=env)
subprocess.run([sys.executable,str(out/'code/consolidate_costs.py')],check=True,env=env)
print(str(out))
