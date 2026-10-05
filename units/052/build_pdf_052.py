"""Local cache and display-math alignment wrapper for the shared PDF renderer."""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'shared'))
# Same trusted shared renderer with this unit's isolated scratch cache.
source=(ROOT/'shared/build_pdf_mathjax.py').read_text()
source=source.replace("cache=ROOT/'tmp/mathjax-cache'","cache=ROOT/'tmp/052/mathjax-cache'")
source=source.replace("css=CSS_TEXT+", "css=CSS_TEXT+'\\n.math-display img{display:block;margin:0 auto;vertical-align:baseline!important;}'+")
namespace={'__name__':'__main__','__file__':str(ROOT/'shared/build_pdf_mathjax.py')}
exec(compile(source,str(ROOT/'shared/build_pdf_mathjax.py'),'exec'),namespace)
