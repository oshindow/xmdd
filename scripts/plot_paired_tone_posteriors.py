#!/usr/bin/env python3
"""Plot paired correct/wrong tone posteriors for sampled error events."""
import argparse, csv, json, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fraction',type=float,default=.1);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path(a.events).read_text().splitlines()];rng=random.Random(a.seed)
    groups=[('confused_correct_in_top3','Confused errors'),('confident_wrong','Confident wrong errors')]
    selected=[]
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharey=True)
    for ax,(category,title) in zip(axes,groups):
        pool=[x for x in rows if x['category']==category];sample=rng.sample(pool,max(1,round(len(pool)*a.fraction)))
        sample.sort(key=lambda x:x['wrong_tone_p']-x['correct_tone_p'])
        x=np.arange(len(sample));correct=np.array([r['correct_tone_p'] for r in sample]);wrong=np.array([r['wrong_tone_p'] for r in sample])
        ax.vlines(x,correct,wrong,color=np.where(wrong>=correct,'#999999','#B279A2'),alpha=.55,linewidth=.8)
        ax.scatter(x,correct,s=12,color='#59A14F',label='Correct phone posterior',zorder=3)
        ax.scatter(x,wrong,s=12,color='#E45756',label='Predicted wrong phone posterior',zorder=3)
        ax.axhline(.8,color='#777777',linestyle='--',linewidth=1,label='0.8 confidence threshold')
        ax.set_title(f'{title}: 10% sample ({len(sample)}/{len(pool)})');ax.set_ylabel('Conditional posterior');ax.set_ylim(-.02,1.02);ax.grid(axis='y',alpha=.22);ax.legend(frameon=False,ncol=3,fontsize=8)
        for rank,row in enumerate(sample):
            selected.append({'category':category,'plot_rank':rank,'id':row['id'],'split':row['split'],'ref_index':row['ref_index'],'correct_phone':row['reference'],'predicted_wrong_phone':row['joint'],'correct_p':row['correct_tone_p'],'wrong_p':row['wrong_tone_p'],'margin_wrong_minus_correct':row['wrong_tone_p']-row['correct_tone_p'],'audio':row['audio']})
    axes[-1].set_xlabel('Error-pair rank (sorted by wrong − correct posterior)');fig.suptitle('Paired phone posteriors: each vertical line is one error event',fontsize=15);fig.tight_layout();fig.savefig(a.output_dir/'paired_tone_posteriors_10pct.png',dpi=220,bbox_inches='tight');fig.savefig(a.output_dir/'paired_tone_posteriors_10pct.pdf',bbox_inches='tight');plt.close(fig)
    with (a.output_dir/'paired_tone_posteriors_10pct.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=selected[0],delimiter='\t');w.writeheader();w.writerows(selected)
    print({c:sum(r['category']==c for r in selected) for c,_ in groups})
if __name__=='__main__':main()
