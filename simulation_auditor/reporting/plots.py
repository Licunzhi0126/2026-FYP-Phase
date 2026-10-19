"""Presentation only. No hidden metric calculation in plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_b(data,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.5))
    for version,g in data.groupby('version'):
        ax[0].hist(np.log2(g.fold_change),bins=np.linspace(0,8,33),alpha=.5,label=version)
        ax[1].scatter(g.promoter_log2_p_over_m,g.log2_p_over_m,alpha=.5,s=17,label=version)
    im=data[(data.version=='V7') & (data.imprinting_status.notna()) & (data.imprinting_status!='none') & (data.imprinting_status!='unknown')]
    if len(im):
        ax[1].scatter(im.promoter_log2_p_over_m,im.log2_p_over_m,s=75,facecolors='none',edgecolors='black',linewidths=1.3,label='V7 imprinted')
    ax[0].set(xlabel='Absolute gene log2(P/M) fold-change',ylabel='Number of genes',title='B1: Allelic imbalance')
    ax[1].set(xlabel='Promoter log2(P/M accessibility)',ylabel='Expression log2(P/M)',title='B2: Association, not causation')
    ax[0].legend();ax[1].legend(fontsize=8)
    ax[1].axhline(0,color='gray',lw=.6);ax[1].axvline(0,color='gray',lw=.6)
    fig.tight_layout();fig.savefig(out,dpi=185);plt.close(fig)

def plot_c(seed_data,out):
    order=['original','promoter_equal_final_on','no_grn_imprint_gate',
           'no_final_log_gate','promoter_equal_final_off','no_expression_imprint_gates','upstream_only']
    order=[n for n in order if n in seed_data.version.unique()]
    values=[seed_data.loc[seed_data.version==n,'mean_log2_ratio'] for n in order]
    fig,ax=plt.subplots(figsize=(11,5))
    ax.boxplot(values,showmeans=True)
    ax.set_xticks(range(1,len(order)+1),[n.replace('_','\n') for n in order],fontsize=8)
    ax.set(ylabel='Mean log2(preferred/nonpreferred expression) per seed',
        title=f'C: V7 imprinting ablation ({seed_data.seed.nunique()} paired seeds)')
    fig.tight_layout();fig.savefig(out,dpi=185);plt.close(fig)
