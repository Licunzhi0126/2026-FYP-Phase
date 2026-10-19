"""Plots for advanced C experiments; draw exclusively from saved audit CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_advanced(out):
    root=Path(out)
    result=[]
    s=root/'C_strength_per_seed.csv'
    if s.exists():
        x=pd.read_csv(s)
        fig,ax=plt.subplots(figsize=(8.2,5))
        for seed,t in x.groupby('seed'):
            ax.plot(t.strength_setting,t.mean_log2_ratio,marker='o',lw=1.5,alpha=.5,label=f'seed {seed}')
        g=x.groupby('strength_setting').mean_log2_ratio.agg(['mean','std'])
        ax.plot(g.index,g['mean'],marker='D',lw=2.5,label='Mean across seeds',color='black')
        ax.set(xlabel='Assigned imprinting_strength',ylabel='Mean log2(preferred / nonpreferred)',title='Imprinting-strength response (fixed 0.03 leak)')
        ax.grid(alpha=.17);ax.legend(fontsize=8);fig.tight_layout()
        f=root/'C_strength_scan.png';fig.savefig(f,dpi=180);plt.close(fig);result.append(f)
    s=root/'C_neighbors_per_seed.csv'
    if s.exists():
        x=pd.read_csv(s)
        fig,ax=plt.subplots(figsize=(8.2,5))
        groups=['imprinted','exposed_nonimprinted','other_nonimprinted']
        dat=[x.loc[x.group==g,'mean_abs_bias_shift'] for g in groups]
        ax.boxplot(dat,tick_labels=['Imprinted','Same-region non-imprinted','Other non-imprinted'],showmeans=True)
        ax.set(ylabel='Mean |change in log2(paternal / maternal)|',title='Region override spillover (paired seeds)')
        ax.grid(axis='y',alpha=.2);fig.tight_layout()
        f=root/'C_neighbor_spillover.png';fig.savefig(f,dpi=180);plt.close(fig);result.append(f)
    return result
