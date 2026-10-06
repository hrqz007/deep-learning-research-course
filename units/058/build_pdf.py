"""Build readable CJK PDFs from original course Markdown; local inputs only."""
from pathlib import Path
import sys,os,re,hashlib,html,json,io
ROOT=Path(__file__).resolve().parent
if (ROOT/'.deps').exists(): sys.path.insert(0,str(ROOT/'.deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'tmp/cache'))
for key in ["MPLCONFIGDIR","XDG_CACHE_HOME"]:
 Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
import markdown
from weasyprint import HTML,CSS
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
CSS_TEXT='''
@page { size:A4; margin:20mm 19mm 19mm; @top-left {content: string(shorttitle); font-family:"Noto Sans CJK SC"; font-size:8pt;color:#64748b;} @bottom-right {content:counter(page);font-size:9pt;color:#64748b;} }
body { font-family:"Noto Serif CJK SC",serif; font-size:10.5pt; line-height:1.73;color:#18202a; }
h1,h2,h3,h4 {font-family:"Noto Sans CJK SC",sans-serif;color:#000;break-after:avoid; font-weight:700;}
h1 {font-size:24pt;line-height:1.4;margin:0 0 7mm;string-set:shorttitle content();}
h2 {font-size:16pt;line-height:1.4;margin:8mm 0 3mm;}
h3 {font-size:12.3pt;line-height:1.45;margin:5mm 0 2mm;}
p {margin:0 0 3mm;orphans:3;widows:3;text-align:justify;overflow-wrap:anywhere;}
a {color:#1e5a8a;text-decoration:none;overflow-wrap:anywhere;}
strong {font-family:"Noto Sans CJK SC";font-weight:700;}
figure {margin:4mm 0 4mm;break-inside:avoid;}
figure img {display:block;margin:auto;max-width:100%;max-height:83mm;object-fit:contain;}
figcaption {font-family:"Noto Sans CJK SC";font-size:9pt;line-height:1.55;color:#435064;margin-top:2mm;text-align:left;}
p>img {display:block;margin:4mm auto;max-width:100%;max-height:87mm;object-fit:contain;}
pre {white-space:pre-wrap;overflow-wrap:anywhere;font:8.5pt/1.5 "DejaVu Sans Mono",monospace;background:#f4f6f8;padding:3mm;break-inside:avoid;}
code {font-family:"DejaVu Sans Mono","Noto Sans CJK SC",monospace;font-size:9pt;}
table {border-collapse:collapse;width:100%;font-size:9.2pt;margin:4mm 0;line-height:1.5;}
thead {display:table-header-group;} th,td {border:0.5pt solid #d9dfe6;padding:2mm;vertical-align:middle;} th{background:#e7edf4;font-family:"Noto Sans CJK SC";} tr{break-inside:avoid;} tbody tr:nth-child(even){background:#f8fafc;}
li {margin-bottom:1.5mm;} ul,ol{padding-left:7mm;margin:2mm 0 4mm;}
.math-display {text-align:center;margin:4mm 0;break-inside:avoid;} .math-display img {max-width:100%;max-height:25mm;}
.math-inline {display:inline!important;vertical-align:-0.25em;max-height:1.5em!important;margin:0!important;}
hr {border:0;border-top:.5pt solid #ddd;margin:5mm 0;}
'''
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out');a=p.parse_args()
 src=Path(a.source).resolve();out=Path(a.out).resolve() if a.out else src.with_suffix('.pdf')
 raw=src.read_text(); cache=ROOT/'tmp/math';cache.mkdir(parents=True,exist_ok=True)
 formulas={}
 def formula(m,display=False):
  expr=m.group(1).strip()
  for short,long in [('le','leq'),('ge','geq'),('ne','neq')]:
   expr=re.sub(r'\\'+short+r'\b',lambda match:'\\'+long,expr)
  if '\\begin{' in expr:raise ValueError('Use separate display equations, not environments: '+expr)
  k=hashlib.sha256(expr.encode()).hexdigest()[:20]; dest=cache/(k+'.svg')
  if not dest.exists():
   math_to_image('$'+expr+'$',str(dest),prop=FontProperties(size=12 if display else 10.5),format='svg',dpi=160,color='#18202a')
  tag=f'<img src="{dest.as_uri()}" alt="{html.escape(expr,quote=True)}" class="'+('math-display-img' if display else 'math-inline')+'"/>'
  if display:tag='<div class="math-display">'+tag+'</div>'
  token='MATHPLACEHOLDER'+str(len(formulas))+'TOKEN';formulas[token]=tag
  return '\n\n'+token+'\n\n' if display else token
 # Do not convert code fences; code may contain currency or dollar variables.
 chunks=re.split(r'(```.*?```)',raw,flags=re.S)
 for i in range(0,len(chunks),2):
  chunks[i]=re.sub(r'\$\$(.+?)\$\$',lambda m:formula(m,True),chunks[i],flags=re.S)
  chunks[i]=re.sub(r'(?<!\\)\$([^\n$]+?)(?<!\\)\$',lambda m:formula(m,False),chunks[i])
 body=markdown.markdown(''.join(chunks),extensions=['tables','fenced_code','attr_list','sane_lists'])
 for token,tag in formulas.items():body=body.replace('<p>'+token+'</p>',tag).replace(token,tag)
 # Figure markup with captions from Markdown image alt text.
 body=re.sub(r'<p><img alt="([^"]*)" src="([^"]*)"\s*/></p>',lambda m:f'<figure><img src="{m[2]}" alt="{m[1]}"><figcaption>{m[1]}</figcaption></figure>',body)
 doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body>'+body+'</body></html>'
 out.parent.mkdir(parents=True,exist_ok=True)
 HTML(string=doc,base_url=str(src.parent)).write_pdf(str(out),stylesheets=[CSS(string=CSS_TEXT)],pdf_variant='pdf/a-3u')
 print(out, out.stat().st_size)
if __name__=='__main__':main()
