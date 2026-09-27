#!/usr/bin/env python3
"""Build Icefall-style per-utterance comparison reports and copy error audio."""
import argparse, csv, json, shutil
from collections import Counter
from pathlib import Path


def alignment(ref, hyp):
    n,m=len(ref),len(hyp); d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i
    for j in range(m+1): d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(ref[i-1]!=hyp[j-1]))
    a,b,ops=[],[],[]; i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
            a.append(ref[i-1]);b.append(hyp[j-1]);ops.append('C' if ref[i-1]==hyp[j-1] else 'S');i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            a.append(ref[i-1]);b.append('*');ops.append('D');i-=1
        else:
            a.append('*');b.append(hyp[j-1]);ops.append('I');j-=1
    return list(reversed(a)),list(reversed(b)),list(reversed(ops))


def events_from_alignment(refa,hypa,ops,row,model):
    out=[]; ri=0
    for r,h,op in zip(refa,hypa,ops):
        if op!='C':
            event={"split":row.get("split"),"id":row["id"],"ref_index":ri,
                   "ref_length":sum(x!='*' for x in refa),"relative_position":ri/max(1,sum(x!='*' for x in refa)-1),
                   "reference":r,"baseline":h if model=='baseline' else '',
                   "joint":h,"kind":{"S":"substitution","D":"deletion","I":"insertion"}[op],
                   "same_final_tone_error":bool(op=='S' and r[-1:] in '12345' and h[-1:] in '12345' and r[:-1]==h[:-1]),
                   "audio":row.get("l2_wav")}
            out.append(event)
        if r!='*': ri+=1
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',nargs='+',required=True)
    p.add_argument('--joint',nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--copy-audio',action='store_true');args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    records=[]; bevents=[]; jevents=[]; totals=Counter(); bconf=Counter();jconf=Counter()
    order={'common_same_error':0,'both_wrong_different':1,'joint_only_error':2,'baseline_only_error':3,'both_correct':4}
    for bp,jp in zip(args.baseline,args.joint):
        bs=[json.loads(x) for x in Path(bp).read_text().splitlines()];js=[json.loads(x) for x in Path(jp).read_text().splitlines()]
        for b,j in zip(bs,js):
            assert b['id']==j['id']; ref=j['actual_phones_clean']; bh=b['predicted_phones'];jh=j['predicted_phones']
            bra,bha,bop=alignment(ref,bh);jra,jha,jop=alignment(ref,jh)
            berr=sum(x!='C' for x in bop);jerr=sum(x!='C' for x in jop)
            if not berr and not jerr: cat='both_correct'
            elif berr and jerr and bh==jh: cat='common_same_error'
            elif berr and jerr: cat='both_wrong_different'
            elif jerr: cat='joint_only_error'
            else: cat='baseline_only_error'
            totals[cat]+=1; totals['baseline_errors']+=berr;totals['joint_errors']+=jerr
            for r,h,o in zip(bra,bha,bop):
                if o=='S': bconf[(r,h)]+=1
            for r,h,o in zip(jra,jha,jop):
                if o=='S': jconf[(r,h)]+=1
            rec={'category':cat,'id':j['id'],'split':j.get('split',''),'audio':j.get('l2_wav',''),
                 'ref_length':len(ref),'baseline_errors':berr,'joint_errors':jerr,'delta':jerr-berr,
                 'reference':' '.join(ref),'baseline_hyp':' '.join(bh),'joint_hyp':' '.join(jh),
                 'baseline_ref_aligned':' '.join(bra),'baseline_hyp_aligned':' '.join(bha),'baseline_ops':' '.join(bop),
                 'joint_ref_aligned':' '.join(jra),'joint_hyp_aligned':' '.join(jha),'joint_ops':' '.join(jop)}
            records.append(rec);bevents.extend(events_from_alignment(bra,bha,bop,j,'baseline'));jevents.extend(events_from_alignment(jra,jha,jop,j,'joint'))
    records.sort(key=lambda x:(order[x['category']],-min(x['baseline_errors'],x['joint_errors']),-max(x['baseline_errors'],x['joint_errors']),x['id']))
    fields=list(records[0]);
    with (out/'per_sentence.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(records)
    with (out/'per_sentence.txt').open('w',encoding='utf-8') as f:
        for x in records:
            f.write(f"[{x['category']}] {x['id']}  B_ERR={x['baseline_errors']} J_ERR={x['joint_errors']} DELTA={x['delta']}\n")
            f.write(f"REF: {x['reference']}\nB-H: {x['baseline_hyp']}\nJ-H: {x['joint_hyp']}\n")
            f.write(f"B-A: {x['baseline_ref_aligned']}\n     {x['baseline_ops']}\n     {x['baseline_hyp_aligned']}\n")
            f.write(f"J-A: {x['joint_ref_aligned']}\n     {x['joint_ops']}\n     {x['joint_hyp_aligned']}\nAUDIO: {x['audio']}\n\n")
    for name,events in [('baseline_error_events.csv',bevents),('joint_error_events.csv',jevents)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=events[0]);w.writeheader();w.writerows(events)
    summary={'utterance_categories':dict(totals),'baseline_top_substitutions':[[a,b,n] for (a,b),n in bconf.most_common(50)],
             'joint_top_substitutions':[[a,b,n] for (a,b),n in jconf.most_common(50)]}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if args.copy_audio:
        manifest=[]
        for x in records:
            if x['category']=='both_correct': continue
            src=Path(x['audio']); dest=out/'audio'/x['category']/(x['id']+src.suffix)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if src.is_file() and not dest.exists(): shutil.copy2(src,dest)
            manifest.append({**{k:x[k] for k in ('category','id','split','baseline_errors','joint_errors','delta')},
                             'source_audio':str(src),'copied_audio':str(dest),'exists':dest.is_file()})
        with (out/'audio_manifest.tsv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=manifest[0],delimiter='\t');w.writeheader();w.writerows(manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
