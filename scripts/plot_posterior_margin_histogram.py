#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--bin-width',type=float,default=.1);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path(a.events).read_text().splitlines()]
    delta=np.asarray([r['wrong_tone_p']-r['correct_tone_p'] for r in rows])
    edges=np.arange(-1,1+a.bin_width*.5,a.bin_width);counts,_=np.histogram(delta,bins=edges)
    labels=[f'[{edges[i]:.1f}, {edges[i+1]:.1f}{"]" if i==len(counts)-1 else ")"}' for i in range(len(counts))]
    summary=[]
    for label,left,right,count in zip(labels,edges[:-1],edges[1:],counts):
        summary.append({'interval':label,'left':left,'right':right,'count':int(count),'percent':100*int(count)/len(rows)})
    with (a.output_dir/'posterior_margin_bins.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    (a.output_dir/'posterior_margin_summary.json').write_text(json.dumps({'total':len(rows),'mean':float(delta.mean()),'median':float(np.median(delta)),'p_wrong_gt_correct':int((delta>0).sum()),'p_wrong_gt_correct_percent':float(100*(delta>0).mean()),'bins':summary},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(14,6));x=np.arange(len(counts));colors=['#59A14F' if edges[i+1]<=0 else '#E45756' for i in range(len(counts))];bars=ax.bar(x,counts,color=colors,edgecolor='white')
    ax.bar_label(bars,labels=[str(v) for v in counts],padding=3,fontsize=8);ax.axvline(9.5,color='#333333',linewidth=1.2);ax.set_xticks(x,labels,rotation=45,ha='right');ax.set_xlabel(r'Posterior margin interval: $P(wrong)-P(correct)$');ax.set_ylabel('Number of error events');ax.set_title(f'Joint model same-final tone errors (n={len(rows)})');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.output_dir/'posterior_margin_histogram.png',dpi=220,bbox_inches='tight');fig.savefig(a.output_dir/'posterior_margin_histogram.pdf',bbox_inches='tight');print(json.dumps({'total':len(rows),'mean':float(delta.mean()),'median':float(np.median(delta)),'p_wrong_gt_correct':int((delta>0).sum()),'p_wrong_gt_correct_percent':float(100*(delta>0).mean())},indent=2))
if __name__=='__main__':main()
