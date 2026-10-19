"""Use `python -m simulation_auditor --help`. No hidden absolute paths."""
import argparse
from pathlib import Path
from .biological.pipeline import run_b
from .causal.runner import run_c,import_c
from .causal.interventions import REGISTRY
from .causal.advanced import run_strength_scan, run_neighbor_audit
from .reporting.advanced_plots import render_advanced
from .reporting.report import write_report

def main(argv=None):
    p=argparse.ArgumentParser(prog='simulation_auditor',description='ASimAudit — biological (B) and causal-mechanism (C) simulation audit')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('b',help='Audit one or both V6 and V7 simulation datasets')
    b.add_argument('--v6',help='V6 ZIP or extracted directory')
    b.add_argument('--v7',help='V7 ZIP or extracted directory')
    b.add_argument('--out',required=True)
    b.add_argument('--grn-threshold',type=float,default=1e-8)
    c=sub.add_parser('c',help='Re-execute V7 under paired imprinting counterfactual interventions')
    c.add_argument('--generator',required=True,help='Trusted V7 generator source, tree, or ZIP')
    c.add_argument('--out',required=True)
    c.add_argument('--start',type=int,default=42)
    c.add_argument('--seeds',type=int,default=1)
    c.add_argument('--conditions',nargs='+',choices=tuple(REGISTRY),default=None)
    scan=sub.add_parser('strength-scan',help='Dose-response of V7 imprinting strength')
    scan.add_argument('--generator',required=True);scan.add_argument('--out',required=True)
    scan.add_argument('--start',type=int,default=42);scan.add_argument('--seeds',type=int,default=3)
    scan.add_argument('--strengths',nargs='+',type=float,default=[0,.25,.5,.75,.95])
    neigh=sub.add_parser('neighbors',help='Neighbor-gene effects of region imprint overrides')
    neigh.add_argument('--generator',required=True);neigh.add_argument('--out',required=True)
    neigh.add_argument('--start',type=int,default=42);neigh.add_argument('--seeds',type=int,default=3)
    imp=sub.add_parser('import-c',help='Import prior experiments; do not claim new simulation execution')
    imp.add_argument('--main',required=True);imp.add_argument('--promoter',required=True);imp.add_argument('--out',required=True)
    r=sub.add_parser('report',help='Create a Markdown report from B and optional C results')
    r.add_argument('--out',required=True)
    args=p.parse_args(argv)
    if args.command=='b':
        paths={k:getattr(args,k) for k in ('v6','v7') if getattr(args,k)}
        if not paths:p.error('b needs at least --v6 or --v7')
        if args.grn_threshold<0:p.error('GRN threshold must be >=0')
        run_b(paths,args.out,args.grn_threshold)
    elif args.command=='c':
        conditions=args.conditions or tuple(REGISTRY)
        # Ensure unmodified condition is first for paired validation.
        if 'original' in conditions:conditions=('original',)+tuple(i for i in conditions if i!='original')
        run_c(args.generator,args.out,args.start,args.seeds,conditions)
    elif args.command=='strength-scan':
        if args.seeds<1 or any(not 0<=x<=1 for x in args.strengths):p.error('Provide seeds>=1 and strengths between 0 and 1')
        run_strength_scan(args.generator,args.out,args.start,args.seeds,tuple(args.strengths))
        render_advanced(args.out)
    elif args.command=='neighbors':
        if args.seeds<1:p.error('Provide seeds>=1')
        run_neighbor_audit(args.generator,args.out,args.start,args.seeds)
        render_advanced(args.out)
    elif args.command=='import-c':import_c(args.main,args.promoter,args.out)
    elif args.command=='report':write_report(args.out)
