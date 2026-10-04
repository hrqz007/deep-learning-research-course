from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
import fitz
from PIL import Image,ImageDraw
src=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True);d=fitz.open(src);tiles=[]
for i,p in enumerate(d):
 pix=p.get_pixmap(matrix=fitz.Matrix(1.3,1.3));pix.save(out/f'page-{i+1:02}.png');im=Image.open(out/f'page-{i+1:02}.png').convert('RGB');im.thumbnail((300,425));tile=Image.new('RGB',(320,450),'#eeeeee');tile.paste(im,((320-im.width)//2,20));ImageDraw.Draw(tile).text((5,3),f'page {i+1}',fill='black');tiles.append(tile)
for s in range(0,len(tiles),6):
 sheet=Image.new('RGB',(960,900),'white')
 for i,im in enumerate(tiles[s:s+6]):sheet.paste(im,((i%3)*320,(i//3)*450))
 sheet.save(out/f'contact-{s//6}.jpg')
print(src.name,len(d),'pages rendered')
