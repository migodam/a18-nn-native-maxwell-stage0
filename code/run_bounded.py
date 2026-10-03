"""One computation child, cumulative process CPU receipts, no retries."""
import os,sys,json,time,subprocess,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ledger=ROOT/'results/PROCESS_RESOURCE_RECEIPTS.jsonl'
old=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
used=sum(r['cpu_s'] for r in old if r.get('budget_category')=='numerical')
if len(sys.argv)<2:raise SystemExit('usage: run_bounded.py script [arguments]')
script=Path(sys.argv[1]).resolve()
category='numerical'
if '--category' in sys.argv:
    j=sys.argv.index('--category');category=sys.argv[j+1];del sys.argv[j:j+2]
cfg=json.loads((ROOT/'stage0_config.json').read_text())
cap=min(cfg['cpu_budget_s'],float(os.environ.get('A18_GLOBAL_CPU_REMAINING',cfg['cpu_budget_s'])))
if category=='numerical' and used>=cap:raise SystemExit('BUDGET_STOP_PARTIAL')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',A18_CPU_REMAINING=str(cap-used))
attempt=len(old)+1;log=ROOT/'results'/f'process_{attempt:02d}_{script.stem}.log'
start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
with log.open('w') as f:
    p=subprocess.Popen([sys.executable,str(script),*sys.argv[2:]],cwd=ROOT.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(.25)
        # Child guards its numerical work; external monitor checks actual resident bytes.
        mem=subprocess.run(['/bin/ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True)
        if mem.stdout.strip() and int(mem.stdout.strip())*1024>cfg['max_rss_bytes']:
            p.terminate();break
    rc=p.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN)
# ps CPU is included conservatively. ru_maxrss is bytes on this macOS host.
cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
row=dict(attempt=attempt,script=str(script),budget_category=category,exit_code=rc,
         wall_s=time.perf_counter()-start,cpu_s=cpu,peak_rss_bytes=after.ru_maxrss,
         log=str(log.relative_to(ROOT)),cumulative_numerical_cpu_s=used+cpu if category=='numerical' else used,
         new_full_wave_rhs=0,nn_training_epochs=0,new_nonlinear_trajectories=0)
with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
if rc:print(log.read_text()[-7000:])
sys.exit(rc)
