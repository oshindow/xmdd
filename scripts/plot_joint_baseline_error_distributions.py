#!/usr/bin/env python3
import argparse,json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--joint',required=True);p.add_argument('--report-summary',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 b,j=load(a.baseline),load(a.joint); colors={'Baseline':'#4C78A8','Joint':'#E45756'}
 fig,ax=plt.subplots(figsize=(7,5));cats=['confident_wrong','confused_correct_in_top3','history_or_other'];labels=['Confident wrong','Confused (correct in top-3)','History/other'];x=np.arange(3);w=.35
 for k,(name,data) in enumerate([('Baseline',b),('Joint',j)]):
  c=Counter(r['category'] for r in data);vals=[100*c[z]/len(data) for z in cats];ax.bar(x+(k-.5)*w,vals,w,label=f'{name} (n={len(data)})',color=colors[name])
 ax.set_xticks(x,labels);ax.set_ylabel('Errors (%)');ax.set_title('Same-final tone error confidence categories');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.output_dir/'tone_error_categories.png',dpi=220);fig.savefig(a.output_dir/'tone_error_categories.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
 bins=np.linspace(0,1,21)
 for ax,(name,data) in zip(axes,[('Baseline',b),('Joint',j)]):
  ax.hist([r['correct_tone_p'] for r in data],bins=bins,alpha=.65,label='Correct phone',color='#59A14F',density=True)
  ax.hist([r['wrong_tone_p'] for r in data],bins=bins,alpha=.55,label='Predicted wrong phone',color='#E45756',density=True)
  ax.set_title(name);ax.set_xlabel('Conditional tone posterior');ax.grid(alpha=.2);ax.legend(frameon=False)
 axes[0].set_ylabel('Density');fig.suptitle('Posterior distribution on same-final tone errors');fig.tight_layout();fig.savefig(a.output_dir/'tone_posterior_distributions.png',dpi=220);fig.savefig(a.output_dir/'tone_posterior_distributions.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4.5))
 for ax,(name,data) in zip(axes,[('Baseline',b),('Joint',j)]):
  mat=np.zeros((5,5),int)
  for r in data: mat[int(r['correct_tone'])-1,int(r['wrong_tone'])-1]+=1
  im=ax.imshow(mat,cmap='Reds');ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6));ax.set_xlabel('Predicted tone');ax.set_ylabel('Correct tone');ax.set_title(name)
  for y in range(5):
   for x0 in range(5): ax.text(x0,y,str(mat[y,x0]),ha='center',va='center',fontsize=8,color='white' if mat[y,x0]>mat.max()*.55 else 'black')
 fig.suptitle('Same-final tone confusion counts');fig.tight_layout();fig.savefig(a.output_dir/'tone_confusion_matrices.png',dpi=220);fig.savefig(a.output_dir/'tone_confusion_matrices.pdf');plt.close(fig)
 report=json.loads(Path(a.report_summary).read_text())['utterance_categories'];keys=['common_same_error','both_wrong_different','joint_only_error','baseline_only_error','both_correct'];names=['Common same','Both wrong, different','Joint only','Baseline only','Both correct'];vals=[report.get(k,0) for k in keys]
 fig,ax=plt.subplots(figsize=(9,4.8));bars=ax.bar(names,vals,color=['#8C8C8C','#B279A2','#E45756','#4C78A8','#59A14F']);ax.bar_label(bars);ax.set_ylabel('Utterances');ax.set_title('Per-sentence comparison categories');ax.tick_params(axis='x',rotation=18);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.output_dir/'sentence_error_categories.png',dpi=220);fig.savefig(a.output_dir/'sentence_error_categories.pdf');plt.close(fig)
 print(a.output_dir)
if __name__=='__main__':main()
