"""Optional full-TeX equation backend for later mathematical course units.

Uses MathJax SVG with no runtime network or browser. Does not alter existing PDFs.
A unit must still pass full visual QA. Local build dependencies are not learner deps.
"""
from pathlib import Path
import sys,os,re,json,hashlib,html,subprocess
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'build-tools'))
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'tmp/cache'))
import markdown
from weasyprint import HTML,CSS
from base_pdf import CSS_TEXT

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('--out');args=parser.parse_args()
 source=Path(args.source).resolve();output=Path(args.out).resolve() if args.out else source.with_suffix('.pdf')
 text=source.read_text();cache=ROOT/'tmp/mathjax-cache';cache.mkdir(parents=True,exist_ok=True)
 items=[]
 def collect(match,display=False):
  expression=match.group(1).strip();key=hashlib.sha256((str(display)+expression).encode()).hexdigest()[:24]
  item={'expression':expression,'display':display,'path':cache/(key+'.svg'),'token':'MJXPLACEHOLDER'+str(len(items))+'TOKEN'};items.append(item)
  return '\n\n'+item['token']+'\n\n' if display else item['token']
 chunks=re.split(r'(```.*?```)',text,flags=re.S)
 for i in range(0,len(chunks),2):
  chunks[i]=re.sub(r'\$\$(.+?)\$\$',lambda m:collect(m,True),chunks[i],flags=re.S)
  chunks[i]=re.sub(r'(?<!\\)\$([^\n$]+?)(?<!\\)\$',collect,chunks[i])
 missing=[v for v in items if not v['path'].exists()]
 if missing:
  data=[{'expression':v['expression'],'display':v['display']} for v in missing]
  result=subprocess.run(['node',str(ROOT/'build-tools/mathjax_render.cjs')],input=json.dumps(data),text=True,capture_output=True,check=True)
  rendered=json.loads(result.stdout)
  for item,svg in zip(missing,rendered):item['path'].write_text(svg)
 body=markdown.markdown(''.join(chunks),extensions=['tables','fenced_code','attr_list','sane_lists'])
 for item in items:
  svg=item['path'].read_text();alignment=re.search(r'vertical-align:\s*([^;" ]+)',svg);vertical=alignment.group(1) if alignment else '-0.25em'
  tag=f'<img src="{item["path"].as_uri()}" alt="{html.escape(item["expression"],quote=True)}" class="'+('math-display-img' if item['display'] else 'math-inline')+'" style="vertical-align:'+vertical+'"/>'
  if item['display']:tag='<div class="math-display">'+tag+'</div>'
  body=body.replace('<p>'+item['token']+'</p>',tag).replace(item['token'],tag)
 body=re.sub(r'<p><img alt="([^"]*)" src="([^"]*)"\s*/></p>',lambda m:f'<figure><img src="{m[2]}" alt="{m[1]}"><figcaption>{m[1]}</figcaption></figure>',body)
 css=CSS_TEXT+'\n.math-display img{display:block;margin:0 auto;vertical-align:baseline!important;}'+'\n.math-display img{max-height:none}.math-inline{max-height:none!important;vertical-align:baseline;}'
 HTML(string='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body>'+body+'</body></html>',base_url=str(source.parent)).write_pdf(str(output),stylesheets=[CSS(string=css)],pdf_variant='pdf/a-3u')
 print(output,output.stat().st_size)
if __name__=='__main__':main()
