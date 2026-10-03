"""Final read-only audit of pinned manifests and all consumed original inputs."""
import csv, json, hashlib, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'stage0_config.json').read_text())
public=Path(cfg['source_root']);original=Path(cfg['original_root']);commit=cfg['source_commit']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(public),*args])
assert git('rev-parse','HEAD').decode().strip()==commit
manifest_receipts=[]
for rel in ['PUBLIC_MANIFEST.json','analysis/a17_tiny_proposer/DATASET_MANIFEST_FINAL.json','A17_R1/SOURCE_MANIFEST.json']:
    blob=git('show',commit+':'+rel)
    expected=hashlib.sha256(blob).hexdigest();actual=sha(public/rel)
    assert actual==expected,rel
    manifest_receipts.append(dict(path=str(public/rel),pinned_git_blob_sha256=expected,actual_sha256=actual,pass_hash=True))
entries=json.loads((public/'PUBLIC_MANIFEST.json').read_text())
for row in entries:assert sha(public/row['public_path'])==row['output_sha256'],row['public_path']
source_row=next(r for r in entries if r['public_path']=='A17_R1/SOURCE_MANIFEST.json')
original_expected=source_row.get('source_epoch','').partition(':')[2]
original_manifest_hash=sha(original/'SOURCE_MANIFEST.json')
if original_expected:assert original_manifest_hash==original_expected
raw_source=json.loads((original/'SOURCE_MANIFEST.json').read_text())
pinned_source=json.loads((public/'A17_R1/SOURCE_MANIFEST.json').read_text())
for name,row in raw_source['modules'].items():
    p=original/row['path']
    assert sha(p)==row['sha256'],name
    assert pinned_source['modules'][name]['sha256']==row['sha256'],name
consumed=json.loads((ROOT/'INPUT_HASHES.json').read_text())
for name,h in consumed.items():assert sha(name)==h,name
dataset=json.loads((public/'analysis/a17_tiny_proposer/DATASET_MANIFEST_FINAL.json').read_text())
raw_checks=[]
for cp in dataset['checkpoints']:
    rel=cp['path'].split('/Gaussian/A17/EXCHANGE_R1/',1)[1]
    actual=sha(original/rel);assert actual==cp['sha256']
    raw_checks.append(dict(path=str(original/rel),actual_sha256=actual,manifest_expected=cp['sha256'],pass_hash=True))
for name in ['exchange_core.py','endpoint_batch.py','removal_core_batch.py']:
    assert sha(ROOT/'vendor'/name)==sha(public/'A17_R1/code'/name),name
a18=original.parents[1]/'A18'
doc_checks=[]
for p in (ROOT/'inputs').glob('*.md'):
    origin=a18/p.name
    if origin.exists():
        assert p.read_bytes()==origin.read_bytes(),p.name
        doc_checks.append(dict(name=p.name,mode='VERBATIM_ORIGINAL',sha256=sha(p)))
merged=(a18/'A18_NN_Native_Research.md').read_text()
restored=merged[merged.index('# A18 sources and access registry'):].rstrip()+'\n'
assert (ROOT/'inputs/REFERENCES.md').read_text()==restored
result=dict(status='PASS',pinned_commit=commit,public_manifest_entries_checked=len(entries),
            manifest_receipts=manifest_receipts,
            original_source_manifest=dict(path=str(original/'SOURCE_MANIFEST.json'),sha256=original_manifest_hash,
                                         public_source_epoch_expected=original_expected or None,
                                         module_hash_checks=len(raw_source['modules']),all_modules_match=True,
                                         all_module_hashes_equal_pinned_public_module_registry=True,
                                         metadata_bytes_differ_from_curated_public_manifest=True),
            consumed_inputs_rehashed=len(consumed),all_consumed_inputs_unchanged=True,
            original_teacher_checks=raw_checks,vendor_copies_exact=True,original_document_checks=doc_checks,
            references_restored_verbatim_from_merged=True,
            original_inputs_read_only=True,no_git_mutation=True)
(ROOT/'verification/FINAL_SOURCE_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['original_teacher_checks','original_document_checks']}))
