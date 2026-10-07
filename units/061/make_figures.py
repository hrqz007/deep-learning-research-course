"""Original figures exclusively from saved experiment outputs."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':160})
COL=['#136f8a','#cc6b32','#6554a4'];FIG=ROOT/'figures'
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def scatter(ax,x,title):
    ax.scatter(x[:,0],x[:,1],s=2,alpha=.3,color='#136f8a',rasterized=True);ax.set(xlim=(-3,3),ylim=(-3,3),title=title,xlabel='x1',ylabel='x2');ax.set_aspect('equal')
def main():
    FIG.mkdir(exist_ok=True);r=json.loads((ROOT/'outputs/results.json').read_text());data=np.load(ROOT/'data/mixture.npz')
    fig,ax=plt.subplots(figsize=(10,3.2));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    for x,y,w,txt,c in [(0.2,1.6,1.5,'Noise z','#e8f4fa'),(2.2,1.6,2,'Generator G','#e8f4fa'),(4.7,1.6,1.7,'Fake x','#e8f4fa'),(4.7,.2,1.7,'Real data','#fff1e8'),(7,1,2,'Discriminator D','#eeeaf9')]:
        ax.add_patch(FancyBboxPatch((x,y),w,.7,boxstyle='round,pad=.05',facecolor=c,edgecolor='#4b5563'));ax.text(x+w/2,y+.35,txt,ha='center',va='center')
    for a,b in [((1.7,1.95),(2.2,1.95)),((4.2,1.95),(4.7,1.95)),((6.4,1.95),(7,1.5)),((6.4,.55),(7,1.15))]:ax.annotate('',b,a,arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(2.4,.55,'D step: detach fake\nG step: freeze D weights;\nretain input gradient',ha='center',va='center',fontsize=9);save(fig,'01_game.png')
    fig,axs=plt.subplots(2,3,figsize=(10,6));scatter(axs[0,0],data['test'],'Reference: 8 modes')
    for ax,run in zip([axs[0,1],axs[0,2],axs[1,0]],r['runs']):
        n=run['seed'];x=np.load(ROOT/f'outputs/baseline_seed{n}_samples.npz')['3000'];scatter(ax,x,f'Seed {n}: {run["final"]["valid_fraction"]:.1%} valid')
    stress=np.load(ROOT/'outputs/stress_seed17_samples.npz');step=r['stress']['diagnostic']['step'];scatter(axs[1,1],stress[str(step)],f'Trained collapse: step {step}, 1/8');scatter(axs[1,2],stress['2200'],'Final: outside mode regions\n0/8 covered');fig.tight_layout();save(fig,'02_samples.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.3))
    for run,c in zip(r['runs']+[r['stress']],COL+['#b42342']):
        h=run['history'];label=f'seed {run["seed"]}'+(' stress' if run['stress'] else '')
        axs[0].plot([a['step'] for a in h],[a['coverage'] for a in h],color=c,label=label);axs[1].plot([a['step'] for a in h],[a['valid_fraction'] for a in h],color=c,label=label)
    axs[0].set(ylabel='Covered modes / 8',ylim=(-.2,8.5));axs[1].set(ylabel='Valid fraction',ylim=(-.03,1.05))
    for ax in axs:ax.set_xlabel('Discriminator rounds');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'03_coverage.png')
    fig,axs=plt.subplots(1,4,figsize=(11,3))
    for ax,step in zip(axs,[500,800,1500,2200]):
        scatter(ax,stress[str(step)],f'Stress step {step}');ax.scatter(data['centers'][:,0],data['centers'][:,1],marker='x',color='#cc6b32',s=35)
    fig.tight_layout();save(fig,'04_stress_path.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for ax,run in zip(axs,[r['runs'][0],r['stress']]):
        h=run['history'][1:];ax.plot([a['step'] for a in h],[a['d_loss'] for a in h],label='D loss: sum of two means');ax.plot([a['step'] for a in h],[a['g_loss'] for a in h],label='G loss: non-saturating');ax.axhline(np.log(4),ls=':',c='gray');ax.axhline(np.log(2),ls=':',c='gray');ax.set(title='Stress 5:1' if run['stress'] else 'Baseline seed 11',xlabel='Discriminator rounds',ylabel='Loss');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'05_losses.png')
if __name__=='__main__':main()
