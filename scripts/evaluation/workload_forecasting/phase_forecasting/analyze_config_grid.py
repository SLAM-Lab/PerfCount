#!/usr/bin/env python3
"""Analyze the (source->target) config grid as the two scheduling problems:
DVFS (same core, by frequency distance/direction) and Scheduling (cross core,
same-freq swap vs cross_proc_freq). Reports the phase-aware forecaster (best of
global/per_phase) vs translated-persistence, @all and @transition."""
import argparse, re, numpy as np, pandas as pd
NOPHASE=re.compile(r'^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)$')
def parse_bench(b):
    m=NOPHASE.match(b); return (m.group('cpu'), float(m.group('freq'))) if m else (None,None)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); a=ap.parse_args()
    d=pd.read_csv(a.csv)
    d=d[d.regime=='translated'].copy()
    tc,tf=zip(*d.bench.map(parse_bench)); d['tcpu']=tc; d['tfreq']=tf
    d['scpu']=d.source.str.split(':').str[0]; d['sfreq']=d.source.str.split(':').str[1].astype(float)
    d['problem']=np.where(d.tcpu==d.scpu,'DVFS','Scheduling')
    d['fdist']=(d.tfreq-d.sfreq)           # signed: + = predict higher freq
    # wide per (bench,pair,phase)
    key=['bench','source','tcpu','tfreq','scpu','sfreq','problem','fdist']
    w=d.pivot_table(index=key,columns=['method','phase'],values='mape').reset_index()
    def col(meth,ph): return (meth,ph) if (meth,ph) in w.columns else None
    def fc(ph):  # best of global/per_phase
        cs=[c for c in [col('global',ph),col('per_phase',ph)] if c]
        return w[cs].min(axis=1) if cs else np.nan
    def ps(ph): c=col('persistence',ph); return w[c] if c else np.nan
    for ph in ['all','transition']:
        w[f'fc_{ph}']=fc(ph); w[f'ps_{ph}']=ps(ph)
    print(f"=== Config grid ({d.bench.nunique()} benches) : forecaster(best) vs translated-persistence ===\n")
    # DVFS: by |fdist| and direction
    print("--- DVFS (same core): by frequency step ---")
    print(f"  {'step':14s} {'n':>5} {'fc@all':>7} {'ps@all':>7} {'win%':>5}  {'fc@tr':>7} {'ps@tr':>7} {'win%':>5}")
    dv=w[w.problem=='DVFS']
    for lbl,sub in [('+1 (speed up)',dv[dv.fdist==1]),('+2',dv[dv.fdist==2]),('+3',dv[dv.fdist==3]),
                    ('-1 (slow down)',dv[dv.fdist==-1]),('-2',dv[dv.fdist==-2]),('-3',dv[dv.fdist==-3])]:
        if not len(sub): continue
        wa=(sub.fc_all<sub.ps_all).mean()*100; wt=(sub.fc_transition<sub.ps_transition).mean()*100
        print(f"  {lbl:14s} {len(sub):>5} {sub.fc_all.mean():7.1f} {sub.ps_all.mean():7.1f} {wa:4.0f}%  "
              f"{sub.fc_transition.mean():7.1f} {sub.ps_transition.mean():7.1f} {wt:4.0f}%")
    # Scheduling: cross core, by freq delta
    print("\n--- Scheduling (cross core): same-freq swap vs cross_proc_freq ---")
    print(f"  {'kind':18s} {'n':>5} {'fc@all':>7} {'ps@all':>7} {'win%':>5}  {'fc@tr':>7} {'ps@tr':>7} {'win%':>5}")
    sc=w[w.problem=='Scheduling']
    for lbl,sub in [('same-freq swap',sc[sc.fdist==0]),
                    ('+/-1 freq',sc[sc.fdist.abs()==1]),('+/-2 freq',sc[sc.fdist.abs()==2]),
                    ('+/-3 freq',sc[sc.fdist.abs()==3])]:
        if not len(sub): continue
        wa=(sub.fc_all<sub.ps_all).mean()*100; wt=(sub.fc_transition<sub.ps_transition).mean()*100
        print(f"  {lbl:18s} {len(sub):>5} {sub.fc_all.mean():7.1f} {sub.ps_all.mean():7.1f} {wa:4.0f}%  "
              f"{sub.fc_transition.mean():7.1f} {sub.ps_transition.mean():7.1f} {wt:4.0f}%")
if __name__=='__main__': main()
