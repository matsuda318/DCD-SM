"""Build simulation and fixed-PC1 empirical tables from audited results.

No model is estimated here. N=1 is omitted from manuscript tables.
Run from any directory: python paper/build_sec4_sec5_tables.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'tables'
SIM_ROOT = ROOT/'results'/'sec4_simulation_dgp'/'common10'
FULL_ROOT = ROOT/'results'/'sec4_simulation_m1_m5'
EMP_ROOT = ROOT/'results'/'sec5_empirical_raw_pc1'
EMP = EMP_ROOT/'tables'
BLOCKS = (('M3','eta'),('M4','even'),('M4','odd'),('M5','even'),('M5','odd'))
EMPIRICAL_BLOCKS = (('M2','location'),('M2','scale')) + BLOCKS
DGPS = ('case0_static','case1_eta','case2_even','case3_eta_even','case4_eta_even_odd')
CASE_NAMES = ('Static',r'$\eta$ only','Even only',r'$\eta$ and even',r'$\eta$, even, and odd')
VARIANTS = ('matched_raw','pc1_residual')
VARIANT_NAMES = ('Raw','PC1 residual')
SOURCE_HASHES = {}
ARTIFACTS = {}


def read(path):
    SOURCE_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    SOURCE_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def full_stage_results():
    """Recompute paired percentage summaries; archived OOS summaries are absolute gains."""
    design = read_json(FULL_ROOT/'design.json')
    validation = read_json(FULL_ROOT/'validation.json')
    baseline = read_json(SIM_ROOT/'design.json')
    assert validation['complete'] and validation['passed']
    assert design['replications']==100 and design['no_truth_in_estimation']
    assert design['generator_source_sha256']==baseline['baseline_source_sha256']
    assert design['calibration']==baseline['original_calibration']
    for key in ('training','oos','burn','seed','common_share'):
        assert design['design'][key]==baseline['design'][key],key
    wald = []
    for row in read(FULL_ROOT/'rejection_summary.csv'):
        if int(row['N']) not in (20,50,100):
            continue
        assert int(row['requested'])==int(row['valid'])==100
        assert int(row['invalid'])==int(row['missing_tests'])==0
        wald.append(dict(row, setting='iid' if row['case']=='iid' else 'common10',
            rate=row['rejection_rate'], failed_or_not_fitted=row['missing_tests']))
    assert len(wald)==300  # 30 design cells, ten block/joint tests each.
    levels = [row for row in read(FULL_ROOT/'oos_levels.csv') if int(row['N']) in (20,50,100)]
    assert len(levels)==15000  # All five models in every retained setting.
    by_model = index(levels,('dataset','model'))
    groups = defaultdict(list)
    for row in levels:
        if row['model']=='M2':
            setting = 'iid' if row['case']=='iid' else 'common10'
            groups[(setting,row['N'],row['dgp'])].append(row['dataset'])
    assert len(groups)==30
    relative = []
    incremental = []
    metrics = ('continuous_log_score','q_loss_0.01')
    for (setting,n,dgp),datasets in groups.items():
        assert len(datasets)==100
        for benchmark,model in (('M2','M3'),('M2','M4'),('M2','M5'),('M3','M5')):
            for metric in metrics:
                parent = [float(by_model[(ds,benchmark)][metric]) for ds in datasets]
                child = [float(by_model[(ds,model)][metric]) for ds in datasets]
                assert all(math.isfinite(x) for x in parent+child)
                if metric=='continuous_log_score':
                    gains = [100*math.expm1(c-p) for p,c in zip(parent,child)]
                    mean = 100*math.expm1(statistics.mean(c-p for p,c in zip(parent,child)))
                else:
                    assert all(p>0 for p in parent)
                    gains = [100*(p-c)/p for p,c in zip(parent,child)]
                    mean = 100*(statistics.mean(parent)-statistics.mean(child))/statistics.mean(parent)
                target = relative if benchmark=='M2' else incremental
                target.append(dict(setting=setting,N=n,dgp=dgp,model=model,metric=metric,
                    valid=100,improvement_percent=mean,median_replication_percent=statistics.median(gains)))
    return wald,relative,incremental



def index(rows,keys):
    result = {tuple(row[k] for k in keys):row for row in rows}
    assert len(result)==len(rows),(keys,'duplicate rows')
    return result


def escape(text):
    return text.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def percent(row):
    count = int(row['valid'])
    assert count==100 and int(row['invalid'])==0 and int(row['failed_or_not_fitted'])==0
    value = 100*int(row['rejected'])/count
    assert abs(value-100*float(row['rate']))<1e-10
    return f'{value:.0f}'


def pvalue(value):
    p = float(value)
    assert 0<=p<=1
    if p>=.001:
        return f'{p:.4f}'
    if p==0:
        return r'$<10^{-16}$'
    mantissa,exponent = f'{p:.2e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def mean_median(row,digits=3):
    assert int(row['valid'])==100
    return (f"{float(row['improvement_percent']):+.{digits}f} / "
            f"{float(row['median_replication_percent']):+.{digits}f}")


def table(name,caption,label,columns,header,rows,note,font_size=10,tabcolsep=4):
    content = [
        '% Generated by build_sec4_sec5_tables.py from audited CSVs; do not edit numbers.',
        r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        rf'\fontsize{{{font_size}pt}}{{{1.2*font_size:g}pt}}\selectfont',
        rf'\setlength{{\tabcolsep}}{{{tabcolsep}pt}}',r'\begin{tabular}{'+columns+'}',r'\hline',
        header+r'\\',r'\hline',*rows,r'\hline',r'\end{tabular}',r'\par\smallskip',
        r'\begin{minipage}{\textwidth}',r'\fontsize{9pt}{10.8pt}\selectfont '+note,
        r'\end{minipage}',r'\end{table}',''
    ]
    text = '\n'.join(content)
    path = OUTPUT/name
    path.write_text(text,encoding='utf-8')
    ARTIFACTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    full_wald,full_rel,full_incremental = full_stage_results()
    sw = index(full_wald,('setting','N','dgp','model','block'))
    sr = index(full_rel,('setting','N','dgp','model','metric'))
    si = index(full_incremental,('setting','N','dgp','model','metric'))
    settings = (('iid','Independent'),('common10',r'Common 10\%'))
    rows = []
    for n in (20,50,100):
        for setting,label in settings:
            values = [percent(sw[(setting,str(n),DGPS[0],m,b)]) for m,b in BLOCKS]
            rows.append(f'{label} & {n} & '+' & '.join(values)+r'\\')
    table('sec4_size.tex','Empirical size of nominal five-percent Wald tests',
        'tab:sim-size','lr'+'r'*5,
        r'Dependence & $N$ & M3 $\eta$ & M4 even & M4 odd & M5 even & M5 odd',rows,
        r'All states are static (case 0). Entries are rejection percentages over 100 replications. '+
        r'All displayed tests are numerically valid. M1--M5 and the M1 Fisher geometry are estimated. '+
        r'The lag-zero covariance holds M2 estimates fixed for M3 and M4, and M3 estimates fixed for M5.')
    rows = []
    for i,dgp in enumerate(DGPS[1:],1):
        for n in (20,50,100):
            for setting,label in settings:
                values = [percent(sw[(setting,str(n),dgp,m,b)]) for m,b in BLOCKS]
                case = f'{i}: {CASE_NAMES[i]}' if n==20 and setting=='iid' else ''
                rows.append(f'{case} & {label} & {n} & '+' & '.join(values)+r'\\')
        if i<4: rows.append(r'\hline')
    table('sec4_rejection.tex','Rejection frequencies under persistent shape dynamics',
        'tab:sim-rejection','llr'+'r'*5,
        r'Case & Dependence & $N$ & M3 $\eta$ & M4 even & M4 odd & M5 even & M5 odd',rows,
        r'Percentages over 100 replications at nominal size 5\%. Columns for active components describe '+
        r'detection; rejections in other columns need not identify a separate true shape component. '+
        r'True coefficients depend on $N$ through the stated calibration. M1--M5 are estimated. '+
        r'Wald calculations hold M2 estimates fixed for M3 and M4, and M3 estimates fixed for M5.',font_size=9)
    rows = []
    for i,dgp in enumerate(DGPS):
        for n in (20,50,100):
            for setting,label in settings:
                values = [f"{float(sr[(setting,str(n),dgp,m,metric)]['median_replication_percent']):+.3f}"
                    for metric in ('continuous_log_score','q_loss_0.01') for m in ('M3','M4','M5')]
                case = f'{i}: {CASE_NAMES[i]}' if n==20 and setting=='iid' else ''
                rows.append(f'{case} & {label} & {n} & '+' & '.join(values)+r'\\')
        if i<4: rows.append(r'\hline')
    table('sec4_oos_density_tail.tex','Median OOS density and tail improvements relative to M2',
        'tab:sim-oos-tail','llr'+'r'*6,
        r'\multicolumn{3}{l}{Criterion} & \multicolumn{3}{c}{Log score} & '+
        r'\multicolumn{3}{c}{1\% quantile loss}\\'+'\n'+
        r'\cline{4-6}\cline{7-9}'+'\n'+
        r'Case & Dependence & $N$ & M3 & M4 & M5 & M3 & M4 & M5',rows,
        r'Median percentage improvements over 100 replications. Within each replication, '+
        r'log-score improvements are $100\{\exp(\overline{\Delta\log g})-1\}$ and '+
        r'quantile-loss improvements are percentage reductions relative to M2. All models, including '+
        r'the M2 benchmark, are estimated.',font_size=9)
    rows = []
    for i,dgp in enumerate(DGPS):
        for n in (20,50,100):
            for setting,label in settings:
                values = [mean_median(si[(setting,str(n),dgp,'M5',metric)])
                    for metric in ('continuous_log_score','q_loss_0.01')]
                case = f'{i}: {CASE_NAMES[i]}' if n==20 and setting=='iid' else ''
                rows.append(f'{case} & {label} & {n} & '+' & '.join(values)+r'\\')
        if i<4: rows.append(r'\hline')
    table('supp_simulation_m5_m3_oos.tex','OOS improvements of M5 relative to M3: mean / median',
        'tab:sim-m5-m3-oos','llrrr',
        r'Case & Dependence & $N$ & Log score (\%) & 1\% quantile loss (\%)',rows,
        r'Percentage improvements over 100 paired replications with M1--M5 estimated. '+
        r'Each replication has 642 OOS dates. Positive values improve on M3. '+
        r'Means use the exponential of the mean log-score difference or the reduction in paired mean losses. '+
        r'Medians use replication-specific percentage improvements.',font_size=9)
    empirical_validation = read_json(EMP_ROOT/'validation.json')
    empirical_audit = read_json(EMP_ROOT/'audit.json')
    assert empirical_validation['all_available_checks_passed']
    assert empirical_audit['all_available_checks_passed'] and not empirical_audit['missing']
    assert empirical_audit['checked']==38
    samples = read(EMP/'samples.csv')
    assert len(samples)==57 and all(int(r['N_min'])>1 for r in samples)
    by_sample = index(samples,('base_dataset','variant'))
    original = [r for r in samples if r['variant']=='original_raw']
    assert len(original)==19
    def panel_label(r):
        label = r['label'].replace('SP500 ','US ').replace('N225 ','JP ')
        return escape(label)
    rows = []
    for source in original:
        key = source['base_dataset']
        r, residual = by_sample[(key,'matched_raw')], by_sample[(key,'pc1_residual')]
        for field in ('actual_start','training_end','actual_oos_start','end','training_days','oos_days',
                      'training_observations','observations','N_min','N_median','N_max'):
            assert r[field]==residual[field],(key,field,'different raw/residual observations')
        rows.append(panel_label(source)+f" & {r['N_min']}--{r['N_max']} & {float(r['N_median']):.0f}"+
            f" & {r['training_days']} & {r['oos_days']}"+r'\\')
    table('sec5_samples.tex','The 19 constituent-return cross sections',
        'tab:empirical-samples','lrrrr',r'Panel & $N_t$ range & Median $N_t$ & Training dates & OOS dates',rows,
        r'US and JP denote S\&P 500 and Nikkei 225 sectors. Raw and PC1 residual use the same '+
        r'constituent--dates, restricted to constituents with estimable training PC1 loadings. '+
        r'The broad-market panels contain constituent returns, rather than the return '+
        r'on an index. The Real Estate panel starts on September 1, 2016; the other panels start on July 5, 2016.')
    wald = read(EMP/'wald_tests.csv')
    ew = index(wald,('base_dataset','variant','model','block'))
    valid_wald = [r for r in wald if r['wald_valid']=='True']
    invalid_wald = [r for r in wald if r['wald_valid']!='True']
    assert len(wald)==570 and len(valid_wald)==empirical_validation['valid_parameter_tests']==570
    assert len(invalid_wald)==empirical_validation['invalid_parameter_tests']==0
    rows = []
    for variant,label in zip(VARIANTS,VARIANT_NAMES):
        for dataset,market in (('sp500',r'S\&P 500'),('n225','Nikkei 225')):
            if rows:
                rows.append(r'\hline')
            rows.append(r'\multicolumn{7}{l}{'+label+'; '+market+r'}\\')
            for m in ('M2','M3','M4','M5'):
                blocks = {'M2': ('location','scale'), 'M3': ('eta',)}.get(m, ('even','odd','joint'))
                assert all(ew[(dataset,variant,m,b)]['wald_valid']=='True' for b in blocks)
                values = [pvalue(ew[(dataset,variant,m,b)]['wald_p']) for b in blocks]
                if m=='M2':
                    values += ['--','--','--','--']
                elif m=='M3':
                    values = ['--','--'] + values + ['--','--','--']
                else:
                    values = ['--','--','--'] + values
                rows.append(f'{m} & '+' & '.join(values)+r'\\')
    table('sec5_market_wald.tex','Broad-market conditional sandwich-Wald $p$-values',
        'tab:market-wald','lrrrrrr',r'Model & Location & Scale & $\eta$ & Even & Odd & Joint',rows,
        r'The reference is symmetric Student-$t$. Each M2 test sets the three HAR coefficients for '+
        r'the indicated component to zero, retaining fitted dynamics of the other component. '+
        r'M3 has one three-coefficient $\eta$ block. '+
        r'M4 and M5 have even and odd three-coefficient blocks and a six-coefficient joint null. '+
        r'All parameter covariances use lag zero and hold M1 estimates fixed for M2, M2 estimates fixed for M3 and M4, '+
        r'and M3 estimates fixed for M5.')
    rows = []
    for variant,label in zip(VARIANTS,VARIANT_NAMES):
        for sectors,universe,count in ((False,'All panels',19),(True,'Sectors',17)):
            values = []
            for m,b in EMPIRICAL_BLOCKS:
                selected = [r for r in wald if r['variant']==variant and r['model']==m and r['block']==b
                    and (not sectors or r['base_dataset'] not in ('sp500','n225'))]
                assert len(selected)==count
                valid = [r for r in selected if r['wald_valid']=='True']
                values.append(f"{sum(float(r['wald_p'])<.05 for r in valid)}/{len(valid)}")
            rows.append(f'{label} & {universe} & '+' & '.join(values)+r'\\')
    table('sec5_wald_counts.tex','Persistence rejections at the five-percent level',
        'tab:wald-counts','ll'+'r'*7,
        r'\rule{0pt}{30pt}Target & Universe & \shortstack{M2\\location} & \shortstack{M2\\scale} & '+
        r'\shortstack{M3\\$\eta$} & \shortstack{M4\\even} & \shortstack{M4\\odd} & '+
        r'\shortstack{M5\\even} & \shortstack{M5\\odd}',rows,
        r'Raw and PC1 residual use the same constituent--dates. Counts are descriptive and unadjusted '+
        r'for multiplicity. All displayed tests are numerically valid. '+
        r'Numerical validity of a Wald statistic does not '+
        r'establish exact finite-sample size.')
    pca = read(EMP/'pca_diagnostics.csv')
    assert all(p['version']=='training-fixed-covariance-pc1-v1' for p in pca)
    comparisons = read(EMP/'oos_comparisons.csv')
    assert len(comparisons)==57*54
    rows = []
    for variant,label in zip(VARIANTS,VARIANT_NAMES):
        for proposed,benchmark in (('M4','M2'),('M5','M3'),('M5','M4')):
            values = []
            for metric in ('continuous_log_score','q_loss_0.01','q_loss_0.05'):
                selected = [r for r in comparisons if r['variant']==variant and
                    r['proposed']==proposed and r['benchmark']==benchmark and r['score']==metric]
                assert len(selected)==19
                positive = [r for r in selected if float(r['improvement_mean'])>0]
                significant = sum(float(r['hac22_two_sided_p'])<.05 for r in positive)
                values.append(f'{len(positive)} ({significant})')
            rows.append(f'{label} & {proposed}--{benchmark} & '+' & '.join(values)+r'\\')
    table('sec5_oos_counts.tex','Panels with positive OOS forecast improvements',
        'tab:oos-pairwise','llrrr',r'Target & Comparison & Log score & 1\% loss & 5\% loss',rows,
        r'Each entry is the number of positive point-estimate improvements out of 19 panels, followed '+
        r'by the number with a two-sided HAC(22) forecast-difference $p$-value below 0.05 in parentheses. '+
        r'The HAC correction here concerns forecast differences, not the lag-zero parameter Wald tests.')
    levels = read(EMP/'oos_levels.csv')
    el = index(levels,('base_dataset','variant','model'))
    ec = index(comparisons,('base_dataset','variant','proposed','benchmark','score'))
    rows = []
    for variant,label in zip(VARIANTS,VARIANT_NAMES):
        for dataset,market in (('sp500',r'S\&P 500'),('n225','Nikkei 225')):
            for proposed,benchmark in (('M4','M2'),('M5','M3')):
                values = []
                for metric in ('continuous_log_score','q_loss_0.01'):
                    delta = float(ec[(dataset,variant,proposed,benchmark,metric)]['improvement_mean'])
                    parent = float(el[(dataset,variant,benchmark)][metric])
                    if metric=='continuous_log_score':
                        value = 100*math.expm1(delta); values.append(f'{value:+.3f}')
                    else:
                        value = 100*delta/parent; values.append(f'{value:+.3f}')
                rows.append(f'{label} & {market} & {proposed}--{benchmark} & '+' & '.join(values)+r'\\')
    table('sec5_market_oos.tex','Broad-market OOS improvements within each target',
        'tab:oos-market','lllrr',r'Target & Market & Comparison & Log score (\%) & 1\% loss (\%)',rows,
        r'Log-score improvements are $100\{\exp(\overline{\Delta\log g})-1\}$, '+
        r'using the proposed model minus the benchmark. Loss improvements are percentage reductions '+
        r'relative to the indicated benchmark. Positive values improve forecasting.')
    audit = dict(description='Main-paper and supplement tables for simulation and empirical evidence; no univariate numerical rows',
        source_sha256=SOURCE_HASHES,artifact_sha256=ARTIFACTS,simulation_replications=100,
        simulation_main='sec4_simulation_m1_m5',
        simulation_sample_sizes=[20,50,100],empirical_base_panels=19,empirical_variants=len(VARIANTS),
        empirical_display_variants=dict(zip(VARIANT_NAMES,VARIANTS)),empirical_archived_variants=3,
        empirical_main_parameter_wald_tests=sum(r['variant'] in VARIANTS for r in wald),
        empirical_main_valid_wald_tests=sum(r['variant'] in VARIANTS for r in valid_wald),
        empirical_main='sec5_empirical_raw_pc1',
        empirical_parameter_wald_tests=570,empirical_valid_wald_tests=len(valid_wald),
        empirical_invalid_wald_tests=[{k:r[k] for k in ('base_dataset','variant','model','block')} for r in invalid_wald])
    (OUTPUT/'sec4_sec5_table_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(f'Generated {len(ARTIFACTS)} TeX tables from {len(SOURCE_HASHES)} audited source files; N=1 excluded.')


if __name__=='__main__':
    main()
