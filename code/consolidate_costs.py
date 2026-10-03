"""Preserve nested timings and create a non-double-counted process cost ledger."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def jl(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))
target=ROOT/'A18_COST_LEDGER.jsonl';component=ROOT/'results/MAIN_ENDPOINT_COMPONENT_COSTS.jsonl'
if not component.exists():component.write_bytes(target.read_bytes())
receipts=jl(ROOT/'results/PROCESS_RESOURCE_RECEIPTS.jsonl')
assert len({r['attempt'] for r in receipts})==len(receipts)
assert all(r['new_full_wave_rhs']==r['nn_training_epochs']==r['new_nonlinear_trajectories']==0 for r in receipts)
rows=[dict(stage='ACCOUNTING_RULE',scope='PROCESS rows are exclusive envelopes; component and nested replay rows are views, never added again. Reproduction parent receipt includes all replay children.',additive=False)]
for r in receipts:rows.append(dict(stage='PROCESS_RECEIPT',additive=True,**r))
numerical=sum(r['cpu_s'] for r in receipts if r['budget_category']=='numerical')
total=sum(r['cpu_s'] for r in receipts)
summary=dict(stage='MEASURED_PROCESS_TOTAL',additive=False,through_attempt=max(r['attempt'] for r in receipts),process_count=len(receipts),
             numerical_cpu_s=numerical,all_metered_process_cpu_s=total,
             authoring_cpu_s=total-numerical,all_metered_process_wall_s=sum(r['wall_s'] for r in receipts),
             failed_process_cpu_s=sum(r['cpu_s'] for r in receipts if r['exit_code']),
             peak_rss_bytes=max(r['peak_rss_bytes'] for r in receipts),
             cpu_budget_s=3600,max_rss_bytes=4294967296,new_full_wave_rhs=0,nn_training_epochs=0,new_nonlinear_trajectories=0,
             scope='CHILD_PROCESS_RUSAGE INCLUDING FAILED ATTEMPTS; native UI compiler CPU not instrumented; authoring included separately',
             no_measured_physics_speedup=True)
assert numerical<3600 and total<3600 and summary['peak_rss_bytes']<=4294967296
rows.append(summary)
for path in [component,*sorted((ROOT/'results').glob('attempt*_partial/A18_COST_LEDGER.jsonl'))]:
    for r in jl(path):rows.append(dict(stage='COMPONENT_TIMING_VIEW',source_ledger=str(path.relative_to(ROOT)),additive=False,detail=r))
for path in [ROOT/'A18_FEATURE_COSTS.csv',ROOT/'results/A18_COMPENSATION_SOURCE_COSTS.csv']:
    for r in csvrows(path):rows.append(dict(stage='ATOMIC_FEATURE_OR_SOURCE_TIMING_VIEW',source_ledger=str(path.relative_to(ROOT)),additive=False,detail=r))
replay=ROOT/'reproductions/verification_20261003/results/PROCESS_RESOURCE_RECEIPTS.jsonl'
for r in jl(replay):rows.append(dict(stage='NESTED_REPRODUCTION_RECEIPT_VIEW',additive=False,covered_by_parent_attempt=20,detail=r))
cfg=json.loads((ROOT/'stage0_config.json').read_text());up=Path(cfg['source_root'])
for state in sorted((up/'A17_R1/results').glob('object_*/setup_cost.json')):
    if 'object_2007_early/' in str(state):continue # canonical state uses early_portable
    rows.append(dict(stage='HISTORICAL_COLD_SETUP_REFERENCE',additive=False,source_path=str(state),
                     detail=json.loads(state.read_text()),scope='ORIGINAL A17 COST, NOT NEW A18 EXECUTION; nested anchor/pool/projection intervals do not double sum'))
for name,I,R,rho in [('historical_A17_multiround',216,1164,None),('primary_B4_single_decision',54,108,10/18)]:
    rows.append(dict(stage='COUNTERFACTUAL_RHS_COST',additive=False,scope=name,initialization_rhs=I,finalist_rhs=R,
                     six_rhs_per_shared_direction=True,ideal_halving_skip_floor=(1+I/R)/2,
                     oracle_skip_fraction=rho,ideal_initialized_rhs_ratio=(I+(1-rho)*R)/(I+R) if rho is not None else None,
                     formula='C_new=I+(1-rho)R+C_refresh+C_audit+C_failed+C_model/source/cold',
                     refresh_and_audit='NOT_RUN; required nonnegative future costs',measured=False,
                     cache_rule='Exact new Jd does not repair accumulated cached Jx error',new_full_wave_rhs=0))
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(ROOT/'results/RESOURCE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
