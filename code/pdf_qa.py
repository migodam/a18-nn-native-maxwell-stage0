"""Render every exported PDF page and prepare recorded visual review sheets."""
import json, subprocess, hashlib
from pathlib import Path
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
qa = ROOT / 'qa'
reports = []
for pdf in sorted((ROOT / 'manuscript').glob('*.pdf')):
    folder = qa / 'rendered' / pdf.stem
    folder.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf))
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-r', '100', '-png', str(pdf), str(folder/'page')], check=True, capture_output=True)
    text_path = qa / (pdf.stem + '_extracted.txt')
    subprocess.run(['/opt/homebrew/bin/pdftotext', '-layout', str(pdf), str(text_path)], check=True)
    text = text_path.read_text()
    images = sorted(folder.glob('page-*.png'))
    assert len(images) == len(reader.pages)
    page_records = []
    for i, (page, img_path) in enumerate(zip(reader.pages, images), 1):
        with Image.open(img_path) as im:
            assert im.width > 800 and im.height > 1100
            dims = list(im.size)
        page_records.append(dict(page=i, image=str(img_path.relative_to(ROOT)), pixels=dims,
                                 text_chars=len(page.extract_text()), visual_review='PENDING'))
    contacts = []
    for start in range(0, len(images), 4):
        canvas = Image.new('RGB', (1800, 2600), '#d8dde5')
        draw = ImageDraw.Draw(canvas)
        for j, path in enumerate(images[start:start+4]):
            im = Image.open(path).convert('RGB')
            im.thumbnail((875, 1245))
            x, y = (j % 2)*900 + 12, (j // 2)*1300 + 40
            draw.text((x, y-27), f'{pdf.stem}  page {start+j+1}', fill='#101820')
            canvas.paste(im, (x, y))
        target = folder / f'contact_{start+1:02d}_{min(start+4,len(images)):02d}.png'
        canvas.save(target)
        contacts.append(str(target.relative_to(ROOT)))
    log = (ROOT/'manuscript'/f'{pdf.stem}.log').read_text(errors='replace')
    reports.append(dict(pdf=str(pdf.relative_to(ROOT)), source=str(pdf.with_suffix('.tex').relative_to(ROOT)),
                        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(pdf.with_suffix('.tex').read_bytes()).hexdigest(),
                        pages=len(reader.pages), page_records=page_records, contacts=contacts,
                        unresolved_citation_marker='[?]' in text,
                        replacement_characters=text.count('\ufffd'),
                        overfull_warnings=[s for s in log.splitlines() if 'Overfull' in s],
                        missing_glyph_warnings=[s for s in log.splitlines() if 'Missing character' in s],
                        visual_review='PENDING'))
(qa/'PDF_QA_RESULTS.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False)+'\n')
print(json.dumps([dict(pdf=r['pdf'], pages=r['pages'], contacts=r['contacts'], overfull=r['overfull_warnings']) for r in reports]))
