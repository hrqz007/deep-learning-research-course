// Convert trusted course TeX into self-contained SVGs without network access.
const fs = require('fs');
const path = require('path');
const candidates=[process.env.MATHJAX_ROOT,path.resolve(__dirname,'node_modules/mathjax-full/js'),path.resolve(__dirname,'../../../.build-deps/node_modules/mathjax-full/js')].filter(Boolean);
const root=candidates.find(p=>fs.existsSync(p+'/mathjax.js'));
if(!root)throw new Error('Install MathJax 3.2.2 using npm install in build_tools, or set MATHJAX_ROOT to its js directory.');
const {mathjax} = require(root+'/mathjax.js');
const {TeX} = require(root+'/input/tex.js');
const {SVG} = require(root+'/output/svg.js');
const {liteAdaptor} = require(root+'/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require(root+'/handlers/html.js');
const {AllPackages} = require(root+'/input/tex/AllPackages.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result=input.map(({expression,display})=>{
  const node=doc.convert(expression,{display,em:16,ex:8,containerWidth:1100});
  const outer=adaptor.outerHTML(node);
  if(outer.includes('data-mjx-error'))throw new Error('Invalid course math: '+expression);
  return outer.slice(outer.indexOf('<svg'),outer.lastIndexOf('</svg>')+6);
});
process.stdout.write(JSON.stringify(result));
