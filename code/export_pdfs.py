"""Export the three native-compiled standalone sources with existing Tectonic."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=[]
for source in sorted((ROOT/'manuscript').glob('*.tex')):
    begin=time.perf_counter();r=subprocess.run(['/opt/homebrew/bin/tectonic','--keep-logs','--keep-intermediates','--outdir',str(source.parent),str(source)],capture_output=True,text=True)
    (ROOT/'qa'/f'{source.stem}_export.log').write_text(r.stdout+r.stderr)
    pdf=source.with_suffix('.pdf');out.append(dict(source=str(source.relative_to(ROOT)),pdf=str(pdf.relative_to(ROOT)),exit_code=r.returncode,wall_s=time.perf_counter()-begin,pdf_present=pdf.exists(),log='qa/'+source.stem+'_export.log'))
    print(json.dumps(out[-1]),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
(ROOT/'qa/TECTONIC_EXPORT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
