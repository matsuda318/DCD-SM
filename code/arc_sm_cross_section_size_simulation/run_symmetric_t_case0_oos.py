"""Score saved known-M2 case0 fits on the existing 642-day OOS sample."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_symmetric_t_known_m2_wald as sim

REPO = sim.REPO
OUTPUT = sim.ROOT/'output_symmetric_t_case0_oos'
PREFIX = 'arc_sm_symmetric_t_case0_oos'
METRICS = tuple(sim.estimator.archived.scoring.SCORE_NAMES)
PAIRS = (('M2','M3'),('M2','M4'),('M3','M5'),('M2','M5'))


def source_hash():
    digest = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n'))
    digest.update(sim.fingerprint().encode())
    return digest.hexdigest()


def references(values, offsets, models):
    """Filter causally from the same zero history, including training days."""
    e = sim.estimator.archived
    knots,masses,helmert,directions,anchor,shape,q = sim.geometry()
    days = len(offsets)-1
    frozen = np.tile(anchor,(days,1))
    ref2 = e.kernels.reference_filter(values,offsets,anchor,np.zeros((3,2)),
        np.ones((3,2)),frozen,shape.projection_shape_on_ms,True,False)
    refs, errors = {'M2':ref2}, {}
    if 'M3' in models:
        try:
            ref3 = e.kernels.reference_filter(values,offsets,anchor,
                np.asarray(models['M3']['coefficients']),np.asarray(models['M3']['parameter_units']),
                frozen,shape.projection_shape_on_ms,True,False)
            assert np.isfinite(ref3[0]).all() and np.isfinite(ref3[3]).all()
            refs['M3'] = ref3
        except Exception as error:
            errors['M3'] = repr(error)
    return refs, errors


def correction(values, offsets, reference, fit):
    e = sim.estimator.archived
    knots,masses,helmert,directions,anchor,shape,q = sim.geometry()
    pit = e.core.dynamic_reference_pit(values,offsets,reference[0])
    phi = np.ascontiguousarray(e.core.normalized_bspline_basis(pit,knots,3,masses))
    result = e.kernels.endpoint_filter(phi,offsets,reference[1],masses,
        directions@helmert.T,directions@q.projection_q_on_hansen,
        np.asarray(fit['coefficients']),np.asarray(fit['parameter_units']),False)
    assert np.isfinite(result[0]).all() and np.isfinite(result[2]).all() and np.isfinite(result[3]).all()
    return result


def score_pair(values, offsets, training, reference, corrected, parent, child):
    e = sim.estimator.archived
    knots,masses,*_ = sim.geometry()
    split = int(offsets[training])
    oos_days = len(offsets)-1-training
    dates = pd.date_range('2000-01-01',periods=oos_days,freq='D')
    # These are day identifiers, not invented real trading-calendar dates.
    if corrected is None:
        corrected = SimpleNamespace(weights=np.tile(masses,(oos_days,1)),contributions=np.zeros(oos_days))
    else:
        corrected = SimpleNamespace(weights=corrected[2][training:],contributions=corrected[3][training:])
    with redirect_stdout(io.StringIO()):
        scores = e.scoring.score_oos_pair(values[split:],offsets[training:]-split,dates,
            values[:split],e.as_reference(reference,training),corrected,
            knots,masses,masses,parent,child)
    scores['oos_day'] = np.repeat(np.arange(oos_days),2)
    return scores.drop(columns='date')


def entries():
    return [sim.load(path) for path in sorted((sim.OUTPUT/'cases').glob('*case0_static*.json'))]


def evaluate(record, output, fingerprint):
    entry = record['entry']; name = entry['dataset']
    path = output/'cases'/(name+'.json')
    if path.exists():
        cached = sim.load(path)
        assert cached['source_sha256']==fingerprint
        return cached
    started = time.monotonic()
    assert record['source_sha256']==sim.fingerprint()
    input_path = REPO/entry['input']
    assert hashlib.sha256(input_path.read_bytes()).hexdigest()==entry['input_sha256']
    with np.load(input_path) as raw:
        values,offsets = raw['values'].copy(),raw['offsets'].copy()
        training = int(raw['training'])
    models = record['models']; counts = np.diff(offsets[:training+1])
    refs, errors = references(values,offsets,models)
    audits = {}
    if 'M3' in refs:
        audits['M3_training_loglik_error'] = abs(float(counts@refs['M3'][3][:training])-models['M3']['layer_log_likelihood'])
    corrected = {}
    for parent,child in (('M2','M4'),('M3','M5')):
        if parent not in refs:
            errors[child] = 'OOS parent M3 unavailable'
        elif child in models:
            try:
                corrected[child] = correction(values,offsets,refs[parent],models[child])
                audits[child+'_training_loglik_error'] = abs(float(counts@corrected[child][3][:training])-models[child]['layer_log_likelihood'])
            except Exception as error:
                errors[child] = repr(error)
    assert max(audits.values(),default=0.)<1e-7, audits
    # Prefix equality independently checks that forecasts do not use future days.
    if entry['replication']==0 and entry['N'] in (1,100):
        prefix_days = training+5
        prefix_refs,_ = references(values[:offsets[prefix_days]],offsets[:prefix_days+1],models)
        for model,ref in prefix_refs.items():
            np.testing.assert_allclose(ref[0],refs[model][0][:prefix_days],rtol=0,atol=0)
        for parent,child in (('M2','M4'),('M3','M5')):
            if child in corrected:
                check = correction(values[:offsets[prefix_days]],offsets[:prefix_days+1],prefix_refs[parent],models[child])
                np.testing.assert_allclose(check[0],corrected[child][0][:prefix_days],rtol=0,atol=0)
        audits['causal_prefix_checked'] = True
    frames = []
    for parent,child in (('M2','M4'),('M3','M5')):
        if parent not in refs:
            continue
        frame = score_pair(values,offsets,training,refs[parent],corrected.get(child),parent,child)
        if child not in corrected:
            frame = frame[frame.model.eq(parent)]
        for model,group in frame.groupby('model'):
            if np.isfinite(group[list(METRICS)].to_numpy()).all():
                expected = refs[parent][3][training:]
                if model==child:
                    expected = expected+corrected[child][3][training:]
                np.testing.assert_allclose(group.continuous_log_score,expected,rtol=1e-10,atol=1e-10)
                frames.append(group.copy())
            else:
                errors[model] = 'nonfinite OOS metric; not scored'
    daily = pd.concat(frames,ignore_index=True)
    assert daily.groupby('model').size().eq(len(offsets)-1-training).all()
    daily_path = output/'daily'/(name+'.csv.gz'); daily_path.parent.mkdir(parents=True,exist_ok=True)
    daily.to_csv(daily_path,index=False,compression='gzip')
    levels = daily.groupby('model')[list(METRICS)].mean()
    differences = []
    for parent,child in PAIRS:
        if parent in levels.index and child in levels.index:
            for metric in METRICS:
                # Positive improvement always favors the child.
                higher = metric=='continuous_log_score' or metric.startswith('categorical_log_score_')
                gain = float((levels.loc[child,metric]-levels.loc[parent,metric])*(1 if higher else -1))
                differences.append({'parent':parent,'child':child,'metric':metric,'improvement':gain})
    result = {'entry':entry,'source_sha256':fingerprint,'fit_source_sha256':record['source_sha256'],
        'training':training,'oos_days':len(offsets)-1-training,'audit':audits,
        'levels':levels.reset_index().to_dict('records'),'comparisons':differences,
        'errors':errors,'unfitted_models':[m for m in sim.MODELS if m not in models],
        'daily_sha256':hashlib.sha256(daily_path.read_bytes()).hexdigest()}
    sim.dump(path,result)
    print(name,'OOS COMPLETE',f'{time.monotonic()-started:.1f}s',errors,flush=True)
    return result


def summarize(output, records, expected):
    levels, comparisons, failures = [], [], []
    for result in records:
        common = {key:result['entry'][key] for key in ('dataset','replication','case','N')}
        levels.extend({**common,**row} for row in result['levels'])
        comparisons.extend({**common,**row} for row in result['comparisons'])
        failures.extend({**common,'model':m,'stage':'unfitted training model'} for m in result['unfitted_models'])
        failures.extend({**common,'model':m,'stage':'OOS','error':error} for m,error in result['errors'].items())
    level_frame = pd.DataFrame(levels); comparison_frame = pd.DataFrame(comparisons)
    level_frame.to_csv(output/'replication_levels.csv',index=False)
    comparison_frame.to_csv(output/'replication_comparisons.csv',index=False)
    pd.DataFrame(failures,columns=['dataset','replication','case','N','model','stage','error']).to_csv(output/'failures.csv',index=False)
    summaries = []
    for keys,group in comparison_frame.groupby(['case','N','parent','child','metric']):
        gains = group.improvement
        summaries.append(dict(zip(('case','N','parent','child','metric'),keys),
            valid=len(gains),mean_improvement=float(gains.mean()),median_improvement=float(gains.median()),
            mc_standard_error=float(gains.std(ddof=1)/np.sqrt(len(gains))) if len(gains)>1 else None,
            improved_count=int((gains>0).sum()),win_rate=float((gains>0).mean())))
    summary = pd.DataFrame(summaries); summary.to_csv(output/'summary.csv',index=False)
    state = {'completed':len(records),'requested':expected,'finished':len(records)==expected,
        'model_oos_records':len(levels),'failure_records':len(failures),'source_sha256':source_hash()}
    sim.dump(output/'status.json',state)
    return summary,state


def audit(output, results):
    assert len(results)==800
    for result in results:
        assert result['source_sha256']==source_hash() and result['fit_source_sha256']==sim.fingerprint()
        assert result['training']==1886 and result['oos_days']==642
        daily_path = output/'daily'/(result['entry']['dataset']+'.csv.gz')
        assert hashlib.sha256(daily_path.read_bytes()).hexdigest()==result['daily_sha256']
        daily = pd.read_csv(daily_path)
        levels = daily.groupby('model')[list(METRICS)].mean()
        for row in result['levels']:
            np.testing.assert_allclose(levels.loc[row['model']], [row[m] for m in METRICS],rtol=1e-10,atol=1e-12)
        for row in result['comparisons']:
            metric = row['metric']; sign = 1 if metric=='continuous_log_score' or metric.startswith('categorical_log_score_') else -1
            gain = sign*(levels.loc[row['child'],metric]-levels.loc[row['parent'],metric])
            np.testing.assert_allclose(gain,row['improvement'],rtol=1e-9,atol=1e-12)
    result = {'passed':True,'cases':len(results),'causal_prefix_checks':sum(bool(r['audit'].get('causal_prefix_checked')) for r in results),
        'maximum_training_loglik_error':max(v for r in results for k,v in r['audit'].items() if k.endswith('error')),
        'fit_source_sha256':sim.fingerprint(),'oos_source_sha256':source_hash()}
    sim.dump(output/'validation.json',result)
    return result


def report(output, summary, state, validation):
    paper = REPO/'paper'
    lines = ['# ケース0：既知M2・対称tのOOS予測比較','',
        '保存済みのケース0（全成分静的）の推定値を使い、学習1886日の後の642日を1日先予測で評価した。係数・既知M2・射影は固定し、日tの密度状態は日t−1までの観測で更新する。再推定や新しいデータ生成は行わない。M2は真の静的周辺密度なので、M1も同じ既知密度になる。','',
        '実証と同じ評価関数・8192予測分位点・カテゴリ50/100/200・下側1%/5%を使う。カテゴリの区切りとBrier閾値は学習観測だけから作る。日別横断平均を642日で平均し、それを反復間で等ウェイト集計する。','',
        '検定の棄却によるモデル選択はせず、推定に成功したモデルを評価する。Wald分散が無効でも予測を計算できれば含める。推定不能・OOS失敗は非改善と数えず、各比較の有効反復数を示す。','',
        '以下は平均改善量。log scoreは子−親、CRPS・quantile loss・Brierは親−子。正なら子モデルが良い。勝数は改善した反復数であり、検定の棄却数ではない。','']
    for metric in METRICS:
        lines += [f'## {metric}','','| 設定 | N | M3−M2 | M4−M2 | M5−M3 | M5−M2 |','|---|---:|---|---|---|---|']
        for case in sim.CASES:
            for n in (1,20,50,100):
                cells = []
                for parent,child in PAIRS:
                    subset = summary[(summary.case==case)&(summary.N==n)&(summary.parent==parent)&(summary.child==child)&(summary.metric==metric)]
                    if subset.empty:
                        cells.append('評価不能（有効0）')
                    else:
                        row = subset.iloc[0]
                        cells.append(f'{row.mean_improvement:+.6g}（勝{row.improved_count}/{row.valid}）')
                lines.append('| '+('iid' if case=='iid' else '共通30%')+f' | {n} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 保存・検証','',f"全{state['completed']}設定を評価。{state['model_oos_records']}モデルのOOS結果を保存。学習尤度の照合、4設定の未来を切り落とした経路との一致、日別スコアと集計の再計算を監査した。",'',
        '[検定実験の設計・結果](arc_sm_symmetric_t_known_m2_wald.md)。','']
    for name in ('summary','replication_levels','replication_comparisons','failures'):
        target = paper/(PREFIX+'_'+name+'.csv'); target.write_bytes((output/(name+'.csv')).read_bytes())
        lines.append(f'- [{target.name}]({target.name})')
    target = paper/(PREFIX+'_audit.json'); sim.dump(target,{'status':state,'validation':validation})
    lines.append(f'- [{target.name}]({target.name})')
    (paper/(PREFIX+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    records = entries(); assert len(records)==800
    if args.pilot:
        records = [r for r in records if r['entry']['replication']==0 and (r['entry']['case'],r['entry']['N']) in (('iid',1),('shared_factor',100))]
    metadata = {'source_sha256':source_hash(),'fit_source_sha256':sim.fingerprint(),'metrics':list(METRICS),'settings':len(records)}
    if (output/'design.json').exists(): assert sim.load(output/'design.json')==metadata
    else: sim.dump(output/'design.json',metadata)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate,r,output,source_hash()) for r in records]
        for future in as_completed(futures):
            results.append(future.result()); summary,state = summarize(output,results,len(records))
    if not args.pilot:
        validation = audit(output,results)
        report(output,summary,state,validation)
    print('OOS ALL COMPLETE',json.dumps(state),flush=True)


if __name__=='__main__':
    main()
