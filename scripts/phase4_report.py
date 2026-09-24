"""Render current measured Phase4 evidence; never fabricate pending results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl
sample=json.loads(Path('artifacts/validation/phase4-v001/manifest.json').read_text())
rows=pl.read_csv('outputs/analysis/P4-PILOT-001/route_utility.csv').to_dicts()
fig,axes=plt.subplots(1,2,figsize=(13,5))
labels=['5k','20k','50k','100k'];rates=[sample['samples'][k]['singleton_rate']*100 for k in 'ABCD']
axes[0].bar(labels,rates,color='#266e93');axes[0].axhline(123247/2206821*100,color='#b25626',linestyle='--',label='Full training prevalence');axes[0].set_ylim(0,7);axes[0].set_ylabel('Singleton entities (%)');axes[0].set_title('Natural-prevalence nested samples');axes[0].legend()
for x,y in zip(labels,rates):axes[0].text(x,y+.1,f'{y:.2f}%',ha='center')
axes[1].plot([r['cumulative_candidates']/1000 for r in rows],[100*r['cumulative_link_recall'] for r in rows],marker='o',color='#266e93');axes[1].set_xlabel('Average candidates per S1');axes[1].set_ylabel('True-link recall (%)');axes[1].set_title('Frozen pilot: diminishing retrieval returns')
for i in [1,3,8]:
 r=rows[i];axes[1].annotate(r['route'],(r['cumulative_candidates']/1000,100*r['cumulative_link_recall']),xytext=(0,-22-i),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_ylim(90,99);axes[1].grid(alpha=.2);fig.suptitle('Phase4 evidence — retrieval plot is development pilot, not OOF');fig.tight_layout();Path('docs/figures').mkdir(exist_ok=True);fig.savefig('docs/figures/phase4_validation_setup.png',dpi=160);plt.close(fig)
