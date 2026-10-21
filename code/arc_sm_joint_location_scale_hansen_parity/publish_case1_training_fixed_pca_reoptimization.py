"""Publish the audited one-panel M3/M5 correction, retaining prior records."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import reoptimize_case1_training_fixed_pca as search

ROOT, DEST, KEY = search.ROOT, search.PUBLISHED, search.DATASET
OUTPUT = search.OUTPUT
ARCHIVE = DEST/'communication_services_pc1_m3_m5_reoptimization'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def update(name, records, match):
    path = DEST/'tables'/(name+'.csv')
    fields, rows = read(path)
    old = [row for row in rows if match(row)]
    backup = ARCHIVE/'before_tables'/(name+'.csv')
    if not backup.exists():
        write(backup, fields, old)
    replacement = iter(records)
    assert len(old) == len(records), (name, len(old), len(records))
    updated = [dict(row, **next(replacement)) if match(row) else row for row in rows]
    write(path, fields, updated)


def main():
    result_path = OUTPUT/KEY/'estimation.json'
    new = json.loads(result_path.read_text())
    old = json.loads((OUTPUT/'before_estimation.json').read_text())
    audit = json.loads((OUTPUT/'audit.json').read_text())
    assert audit['all_available_checks_passed'] and audit['unchanged_models_verified']==['M1','M2','M4']
    assert all(new['models'][m]['wald_diagnostics']['valid'] for m in ('M3','M5'))
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in ('before_estimation.json','M3_search.json','audit.json','reoptimization_provenance.json'):
        shutil.copyfile(OUTPUT/name, ARCHIVE/name)
    for name in ('validation.json','audit.json','run_status.json'):
        backup = ARCHIVE/('before_'+name)
        if not backup.exists():
            shutil.copyfile(DEST/name, backup)
    common = {key:new['dataset'][key] for key in ('dataset','label','family','role')}
    common.update(variant='pc1_residual', base_dataset='sp500_s50')
    def affected(row):
        return row['dataset']==KEY and row['model'] in ('M3','M5')
    update('parameters', [dict(common, model=m,
        coefficients=json.dumps(new['models'][m]['coefficients']),
        free_dimension=new['models'][m]['free_dimension'],lambda_fixed_zero=True)
        for m in ('M3','M5')], affected)
    update('wald_tests', [dict(common, **row) for row in new['wald_tests'] if row['model'] in ('M3','M5')], affected)
    update('training_log_likelihoods', [dict(common,model=m,training_log_likelihood=new['training_log_likelihoods'][m])
                                      for m in ('M3','M5')], affected)
    update('oos_levels', [dict(common,**row) for row in new['oos_summary'] if row['model'] in ('M3','M5')], affected)
    def pair_affected(row):
        return row['dataset']==KEY and (row['proposed'] in ('M3','M5') or row['benchmark'] in ('M3','M5'))
    pairs = [dict(common,**row) for row in new['oos_comparisons'] if pair_affected(dict(common,**row))]
    update('oos_comparisons', pairs, pair_affected)
    _, levels = read(DEST/'tables/oos_levels.csv')
    levels = {(r['dataset'],r['model']):r for r in levels}
    percentages=[]
    for row in pairs:
        metric=row['score']; difference=row['improvement_mean']
        baseline=float(levels[KEY,row['benchmark']][metric])
        gain=100*math.expm1(difference) if metric=='continuous_log_score' else 100*difference/baseline
        percentages.append(dict(row,improvement_percent=gain))
    update('oos_improvement_percent',percentages,pair_affected)
    update('fit_diagnostics',[dict(dataset=KEY,model=m,**{k:new['models'][m]['unrestricted'][k]
        for k in ('projected_gradient_max','successful_starts','total_starts','largest_successful_objective_gap')})
        for m in ('M3','M5')],affected)
    _, tests=read(DEST/'tables/wald_tests.csv')
    fields, rejection=read(DEST/'tables/wald_summary.csv')
    for row in rejection:
        selected=[r for r in tests if r['variant']==row['variant'] and r['model']==row['model'] and r['block']==row['block']
                  and (row['universe']=='all' or r['base_dataset'] not in ('sp500','n225'))]
        valid=[r for r in selected if r['wald_valid']=='True']
        row.update(valid_tests=len(valid),total_panels=len(selected),rejections=sum(float(r['wald_p'])<.05 for r in valid))
    write(DEST/'tables/wald_summary.csv',fields,rejection)
    _, comparisons=read(DEST/'tables/oos_comparisons.csv')
    fields, summary=read(DEST/'tables/oos_summary.csv')
    for row in summary:
        if row['variant']!='pc1_residual' or not(row['proposed'] in ('M3','M5') or row['benchmark'] in ('M3','M5')):
            continue
        selected=[r for r in comparisons if all(r[k]==row[k] for k in ('variant','proposed','benchmark','score'))]
        assert len(selected)==19
        row.update(positive_panels=sum(float(r['improvement_mean'])>0 for r in selected),
            positive_significant_panels=sum(float(r['improvement_mean'])>0 and float(r['hac22_two_sided_p'])<.05 for r in selected),
            negative_significant_panels=sum(float(r['improvement_mean'])<0 and float(r['hac22_two_sided_p'])<.05 for r in selected),
            panel_mean_improvement=sum(float(r['improvement_mean']) for r in selected)/19)
    write(DEST/'tables/oos_summary.csv',fields,summary)
    by_test={(r['dataset'],r['model'],r['block']):r for r in tests}
    fields, rolling=read(DEST/'tables/rolling_fixed_wald_comparison.csv')
    for row in rolling:
        if affected(row):
            current=by_test[row['dataset'],row['model'],row['block']]
            row.update(wald_valid_fixed=current['wald_valid'],wald=current['wald'],wald_p_fixed=current['wald_p'])
    write(DEST/'tables/rolling_fixed_wald_comparison.csv',fields,rolling)
    by_pair={(r['variant'],r['proposed'],r['benchmark'],r['score']):r for r in summary}
    fields, rolling=read(DEST/'tables/rolling_fixed_oos_summary_comparison.csv')
    for row in rolling:
        current=by_pair[tuple(row[k] for k in ('variant','proposed','benchmark','score'))]
        for key in ('positive_panels','positive_significant_panels','negative_significant_panels','panel_mean_improvement'):
            row[key+'_fixed']=current[key]
    write(DEST/'tables/rolling_fixed_oos_summary_comparison.csv',fields,rolling)
    # Retain the original numerical-source fingerprint and explicitly record the new optimizer.
    new['optimizer_reoptimization']=dict(date='2026-10-20',script='reoptimize_case1_training_fixed_pca.py',
        previous_estimate='communication_services_pc1_m3_m5_reoptimization/before_estimation.json',audit='communication_services_pc1_m3_m5_reoptimization/audit.json')
    search.experiment.corrected.archived.dump(DEST/'estimates'/KEY/'estimation.json',new)
    complete_audit=json.loads((DEST/'audit.json').read_text())
    complete_audit['checks']=[audit['checks'][0] if r['dataset']==KEY else r for r in complete_audit['checks']]
    complete_audit['reoptimization_audit']='communication_services_pc1_m3_m5_reoptimization/audit.json'
    search.experiment.corrected.archived.dump(DEST/'audit.json',complete_audit)
    status=json.loads((DEST/'run_status.json').read_text());status['invalid_wald']=sum(r['wald_valid']!='True' for r in tests)
    search.experiment.corrected.archived.dump(DEST/'run_status.json',status)
    validation=json.loads((DEST/'validation.json').read_text())
    validation.update(valid_parameter_tests=sum(r['wald_valid']=='True' for r in tests),invalid_parameter_tests=status['invalid_wald'])
    validation['reoptimization']={'date':'2026-10-20','dataset':KEY,'criterion_unchanged':True,
        'audit':'communication_services_pc1_m3_m5_reoptimization/audit.json','M3_likelihood_gain':audit['M3_likelihood_gain']}
    for name in ('reoptimize_case1_training_fixed_pca.py','audit_case1_training_fixed_pca_reoptimization.py',
                 'publish_case1_training_fixed_pca_reoptimization.py'):
        validation['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    search.experiment.corrected.archived.dump(DEST/'validation.json',validation)
    print('PUBLISHED',json.dumps(validation['reoptimization']))
    print('WALD',json.dumps([r for r in rejection if r['variant']=='pc1_residual' and r['universe']=='all']))
    print('OOS',json.dumps([r for r in summary if r['variant']=='pc1_residual' and r['proposed']=='M5' and r['benchmark']=='M3']))


if __name__=='__main__':
    main()
