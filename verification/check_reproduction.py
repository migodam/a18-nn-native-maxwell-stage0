"""Compare a fresh full-scope run, excluding timings and local path metadata."""
import csv, json, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fresh = ROOT/'reproductions/verification_20261003'
checks=[]
tables=['A18_ANALYTIC_SHORTLIST_BASELINES.csv', 'A18_DEFECT_ORACLE.csv',
        'results/A18_BASELINE_OBJECT_SUMMARY.csv', 'results/A18_B_OBJECT_SUMMARY.csv',
        'results/A18_SOURCE_ORACLE.csv', 'A18_DEFECT_LABEL_AVAILABILITY.csv',
        'results/A18_SNAPSHOT_SPAN_AUDIT.csv']
for rel in tables:
    with (ROOT/rel).open() as f: a=list(csv.DictReader(f))
    with (fresh/rel).open() as f: b=list(csv.DictReader(f))
    assert len(a)==len(b),rel
    ignored={key for key in a[0] if key.endswith(('_wall_s','_cpu_s')) or key in ['wall_s','cpu_s']}
    # Timing changes are independent evidence; scientific values must be exact.
    for x,y in zip(a,b):
        for key in x:
            if key not in ignored: assert x[key]==y[key],(rel,key,x[key],y[key])
    checks.append(dict(path=rel, rows=len(a), ignored_columns=sorted(ignored), scientific_fields_exact=True))
array_checks=[]
for pattern in ['direction_*.npz','features_*.npz','public_query_*.npz','fit_*.npz']:
    paths=sorted((ROOT/'results').glob(pattern))
    max_difference=0.0
    for p in paths:
        with np.load(p,allow_pickle=False) as a,np.load(fresh/'results'/p.name,allow_pickle=False) as b:
            assert set(a.files)==set(b.files),p.name
            for key in a.files:
                x,y=a[key],b[key]
                if x.dtype.kind in 'fc':
                    assert np.array_equal(x,y,equal_nan=True),(p.name,key)
                    if x.size:max_difference=max(max_difference,float(np.nanmax(np.abs(x-y))))
                else: assert np.array_equal(x,y),(p.name,key)
    array_checks.append(dict(pattern=pattern, files=len(paths), array_values_exact=True, max_absolute_difference=max_difference))
for rel in ['A18_SPLITS.json','stage0_config.json','results/COUNTS.json']:
    assert json.loads((ROOT/rel).read_text())==json.loads((fresh/rel).read_text()),rel
result=dict(status='PASS', reproduction_directory=str(fresh), fresh_run_exit_code=0,
            configuration_sha256=hashlib.sha256((ROOT/'stage0_config.json').read_bytes()).hexdigest(),
            csv_checks=checks,array_checks=array_checks,
            equal_split_config_and_scope_counts=True,
            ignored_timing_is_not_runtime_reproducibility=True,
            no_new_physical_scenes_or_rhs=True,
            authoring_scope='NUMERICAL_RUN_BEFORE_FINAL_PROSE_LAYOUT_REFINEMENT; NO_PDF_BYTE_IDENTITY_CLAIM')
(ROOT/'verification/REPRODUCTION_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
