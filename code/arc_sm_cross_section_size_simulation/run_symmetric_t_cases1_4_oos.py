"""Evaluate saved case1--case4 fits with exactly the case0 OOS scoring rules."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_symmetric_t_case0_oos as base

sim = base.sim
REPO = base.REPO
DGPS = sim.DGPS[1:]
OUTPUT = sim.ROOT/'output_symmetric_t_cases1_4_oos'
PREFIX = 'arc_sm_symmetric_t_cases1_4_oos'
METRICS, PAIRS = base.METRICS, base.PAIRS


def source_hash():
    digest = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n'))
    digest.update(base.source_hash().encode())
    return digest.hexdigest()


def evaluate(record, output, fingerprint):
    entry = record['entry']; name = entry['dataset']
    path = output/'cases'/(name+'.json')
    if path.exists():
        cached = sim.load(path)
        assert cached['source_sha256']==fingerprint
        return cached
    started = time.monotonic()
    assert record['source_sha256']==sim.fingerprint()
    result = {'entry':entry,'source_sha256':fingerprint,'fit_source_sha256':record['source_sha256'],
        'training':1886,'oos_days':642,'audit':{},'levels':[],'comparisons':[],
        'failures':[],'daily_sha256':None,'generation_failed':False}
    if 'input' not in entry:
        assert any(f['stage']=='generation' for f in record['failures'])
        result['generation_failed'] = True
        result['failures'] = [{'model':'generation','stage':'generation','error':record['failures'][0]['error']}]
        sim.dump(path,result)
        print(name,'OOS UNAVAILABLE: saved generation failure',flush=True)
        return result
    input_path = REPO/entry['input']
    assert hashlib.sha256(input_path.read_bytes()).hexdigest()==entry['input_sha256']
    with np.load(input_path) as raw:
        values,offsets = raw['values'].copy(),raw['offsets'].copy()
        training = int(raw['training'])
    result['training'],result['oos_days'] = training,len(offsets)-1-training
    models = record['models']; split = int(offsets[training]); counts = np.diff(offsets[:training+1])
    failures,audits = result['failures'],result['audit']
    for model in sim.MODELS:
        if model not in models:
            old = next((f for f in record['failures'] if f['model']==model),{})
            failures.append({'model':model,'stage':'training unavailable','error':old.get('error','not fitted')})
    refs,errors = base.references(values,offsets,models)
    train_refs = None
    if 'M3' in errors:
        failures.append({'model':'M3','stage':'OOS reference','error':errors['M3']})
        # Confirm a full-path failure occurred after the already-fitted sample.
        train_refs,train_errors = base.references(values[:split],offsets[:training+1],models)
        assert not train_errors,train_errors
    if 'M3' in models:
        fitted_ref = refs.get('M3') if 'M3' in refs else train_refs['M3']
        audits['M3_training_loglik_error'] = abs(float(counts@fitted_ref[3][:training])-models['M3']['layer_log_likelihood'])
    corrected = {}
    for parent,child in (('M2','M4'),('M3','M5')):
        if child not in models:
            continue
        if parent not in refs:
            failures.append({'model':child,'stage':'OOS parent unavailable','error':'M3 OOS reference failed'})
            trained = base.correction(values[:split],offsets[:training+1],train_refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@trained[3])-models[child]['layer_log_likelihood'])
            continue
        try:
            corrected[child] = base.correction(values,offsets,refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@corrected[child][3][:training])-models[child]['layer_log_likelihood'])
        except (FloatingPointError,ValueError,OverflowError) as error:
            failures.append({'model':child,'stage':'OOS correction','error':repr(error)})
            if train_refs is None:
                train_refs,train_errors = base.references(values[:split],offsets[:training+1],models)
                assert not train_errors,train_errors
            trained = base.correction(values[:split],offsets[:training+1],train_refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@trained[3])-models[child]['layer_log_likelihood'])
    assert max(audits.values(),default=0.)<1e-7,audits
    if entry['replication']==0 and entry['N'] in (1,100):
        days = training+5
        prefix_refs,_ = base.references(values[:offsets[days]],offsets[:days+1],models)
        for model in refs:
            np.testing.assert_array_equal(prefix_refs[model][0],refs[model][0][:days])
        for parent,child in (('M2','M4'),('M3','M5')):
            if child in corrected:
                prefix = base.correction(values[:offsets[days]],offsets[:days+1],prefix_refs[parent],models[child])
                np.testing.assert_array_equal(prefix[0],corrected[child][0][:days])
        audits['causal_prefix_checked'] = True
    frames = []
    for parent,child in (('M2','M4'),('M3','M5')):
        if parent not in refs:
            continue
        try:
            frame = base.score_pair(values,offsets,training,refs[parent],corrected.get(child),parent,child)
        except (FloatingPointError,ValueError,OverflowError) as error:
            if child not in corrected:
                raise
            failures.append({'model':child,'stage':'OOS scoring','error':repr(error)})
            corrected.pop(child)
            frame = base.score_pair(values,offsets,training,refs[parent],None,parent,child)
        if child not in corrected:
            frame = frame[frame.model.eq(parent)]
        for model,group in frame.groupby('model'):
            if not np.isfinite(group[list(METRICS)].to_numpy()).all():
                failures.append({'model':model,'stage':'OOS scoring','error':'nonfinite OOS metric'})
                continue
            expected = refs[parent][3][training:]
            if model==child:
                expected = expected+corrected[child][3][training:]
            np.testing.assert_allclose(group.continuous_log_score,expected,rtol=1e-10,atol=1e-10)
            frames.append(group.copy())
    assert frames,'no baseline forecast scores'
    daily = pd.concat(frames,ignore_index=True)
    assert daily.groupby('model').size().eq(result['oos_days']).all()
    daily_path = output/'daily'/(name+'.csv.gz'); daily_path.parent.mkdir(parents=True,exist_ok=True)
    daily.to_csv(daily_path,index=False,compression='gzip')
    levels = daily.groupby('model')[list(METRICS)].mean()
    for parent,child in PAIRS:
        if parent in levels.index and child in levels.index:
            for metric in METRICS:
                sign = 1 if metric=='continuous_log_score' or metric.startswith('categorical_log_score_') else -1
                result['comparisons'].append({'parent':parent,'child':child,'metric':metric,
                    'improvement':float(sign*(levels.loc[child,metric]-levels.loc[parent,metric]))})
    result['levels'] = levels.reset_index().to_dict('records')
    result['daily_sha256'] = hashlib.sha256(daily_path.read_bytes()).hexdigest()
    sim.dump(path,result)
    print(name,'OOS COMPLETE',f'{time.monotonic()-started:.1f}s',len(failures),'failures',flush=True)
    return result


def summarize(output, records, expected):
    levels, comparisons, failures = [], [], []
    for result in records:
        common = {key:result['entry'][key] for key in ('dataset','replication','dgp','case','N')}
        levels.extend({**common,**row} for row in result['levels'])
        comparisons.extend({**common,**row} for row in result['comparisons'])
        failures.extend({**common,**row} for row in result['failures'])
    pd.DataFrame(levels).to_csv(output/'replication_levels.csv',index=False)
    frame = pd.DataFrame(comparisons); frame.to_csv(output/'replication_comparisons.csv',index=False)
    pd.DataFrame(failures,columns=['dataset','replication','dgp','case','N','model','stage','error']).to_csv(output/'failures.csv',index=False)
    summaries = []
    if not frame.empty:
        for keys,group in frame.groupby(['dgp','case','N','parent','child','metric']):
            gains = group.improvement
            summaries.append(dict(zip(('dgp','case','N','parent','child','metric'),keys),
                valid=len(gains),mean_improvement=float(gains.mean()),median_improvement=float(gains.median()),
                mc_standard_error=float(gains.std(ddof=1)/np.sqrt(len(gains))) if len(gains)>1 else None,
                improved_count=int((gains>0).sum()),win_rate=float((gains>0).mean())))
    summary = pd.DataFrame(summaries); summary.to_csv(output/'summary.csv',index=False)
    state = {'completed':len(records),'requested':expected,'finished':len(records)==expected,
        'model_oos_records':len(levels),'generation_failed_settings':sum(r['generation_failed'] for r in records),
        'training_missing_models':sum(f['stage']=='training unavailable' for f in failures),
        'oos_failure_records':sum(f['stage'].startswith('OOS') for f in failures),
        'source_sha256':source_hash()}
    sim.dump(output/'status.json',state)
    return summary,state


def audit(output, results):
    assert len(results)==3200
    for result in results:
        assert result['source_sha256']==source_hash() and result['fit_source_sha256']==sim.fingerprint()
        assert result['training']==1886 and result['oos_days']==642
        if result['generation_failed']:
            assert not result['levels'] and not result['comparisons'] and result['daily_sha256'] is None
            original = sim.load(sim.OUTPUT/'cases'/(result['entry']['dataset']+'.json'))
            assert any(f['stage']=='generation' for f in original['failures'])
            continue
        daily_path = output/'daily'/(result['entry']['dataset']+'.csv.gz')
        assert hashlib.sha256(daily_path.read_bytes()).hexdigest()==result['daily_sha256']
        daily = pd.read_csv(daily_path)
        assert daily.groupby('model').size().eq(642).all()
        levels = daily.groupby('model')[list(METRICS)].mean()
        for row in result['levels']:
            np.testing.assert_allclose(levels.loc[row['model']], [row[m] for m in METRICS],rtol=1e-10,atol=1e-12)
        for row in result['comparisons']:
            metric = row['metric']; sign = 1 if metric=='continuous_log_score' or metric.startswith('categorical_log_score_') else -1
            gain = sign*(levels.loc[row['child'],metric]-levels.loc[row['parent'],metric])
            np.testing.assert_allclose(gain,row['improvement'],rtol=1e-9,atol=1e-12)
    summary,frame = pd.read_csv(output/'summary.csv'),pd.read_csv(output/'replication_comparisons.csv')
    for row in summary.itertuples():
        subset = frame[(frame.dgp==row.dgp)&(frame.case==row.case)&(frame.N==row.N)&
            (frame.parent==row.parent)&(frame.child==row.child)&(frame.metric==row.metric)]
        assert len(subset)==row.valid and int((subset.improvement>0).sum())==row.improved_count
        np.testing.assert_allclose(subset.improvement.mean(),row.mean_improvement,atol=1e-12,rtol=1e-9)
    validation = {'passed':True,'cases':len(results),'daily_cases':sum(not r['generation_failed'] for r in results),
        'causal_prefix_checks':sum(bool(r['audit'].get('causal_prefix_checked')) for r in results),
        'maximum_training_loglik_error':max((v for r in results for k,v in r['audit'].items() if k.endswith('error')),default=0.),
        'fit_source_sha256':sim.fingerprint(),'case0_oos_source_sha256':base.source_hash(),'oos_source_sha256':source_hash()}
    sim.dump(output/'validation.json',validation)
    return validation


def report(output, summary, state, validation):
    paper = REPO/'paper'
    lines = ['# ケース1–4：既知M2・対称tのOOS予測比較','',
        'ケース0と同じ評価方法で、保存済みの推定係数を固定して続く642日を1日先予測した。学習1886日、再推定・データ再生成なし。真のηやDCD-SMの状態経路は予測側に与えない。既知M2（静的location/scale/η）からM3–M5を推定した実験の予測評価である。','',
        '| 発生ケース | η | even | odd |','|---|---|---|---|',
        '| 1 | 持続 | 静的 | 静的 |','| 2 | 静的 | 持続 | 静的 |',
        '| 3 | 持続 | 持続 | 静的 |','| 4 | 持続 | 持続 | 持続 |','',
        '実証と同じlog score・CRPS・カテゴリ50/100/200・下側1%/5% quantile loss/Brierを評価。予測分位点8192個。カテゴリ境界とBrier閾値は学習データだけで作る。日別横断平均を642日平均し、同じ反復でのモデル差を反復間で等ウェイト平均する。','',
        '以下の表は平均改善量（中央値はCSV）。log scoreは子−親、損失指標は親−子。正なら子モデルが良い。勝数は改善した反復数であり、有意な検出数ではない。推定成功モデルを棄却の有無によらず評価し、Wald分散が無効なモデルも予測を計算できれば含める。','',
        '生成失敗・学習での推定不能・OOSの予測計算失敗は非改善と数えず、各比較の有効反復数を併記する。将来のreference再帰が既存の数値ガードに抵触した場合はM3をOOS失敗とし、そのreferenceを使うM5も評価不能とする。係数やガードを変更して救済しない。','',
        '共通因子は変換後リターンの共通分散寄与30%。Nは100銘柄masterの先頭Nの入れ子。共通因子の有無で同じ係数でも真の密度経路の変動が変わるため、iidと共通30%の比較は同一経路での純粋な情報量比較ではない。','']
    for dgp in DGPS:
        lines += [f'## {dgp}','']
        for metric in METRICS:
            lines += [f'### {metric}','','| 設定 | N | M3対M2 | M4対M2 | M5対M3 | M5対M2 |','|---|---:|---|---|---|---|']
            for case in sim.CASES:
                for n in (1,20,50,100):
                    cells = []
                    for parent,child in PAIRS:
                        group = summary[(summary.dgp==dgp)&(summary.case==case)&(summary.N==n)&
                            (summary.parent==parent)&(summary.child==child)&(summary.metric==metric)]
                        if group.empty: cells.append('評価不能（有効0）')
                        else:
                            row = group.iloc[0]
                            cells.append(f'{row.mean_improvement:+.6g}（勝{row.improved_count}/{row.valid}）')
                    lines.append('| '+('iid' if case=='iid' else '共通30%')+f' | {n} | '+' | '.join(cells)+' |')
            lines.append('')
    lines += ['## 保存・監査','',
        f"全{state['completed']}設定を記録。生成失敗{state['generation_failed_settings']}設定、学習で未推定のモデル{state['training_missing_models']}件、OOS失敗・親のOOS失敗による評価不能{state['oos_failure_records']}件、評価できたモデル{state['model_oos_records']}件。",'',
        f"日別スコアから全集計を再計算し、入力・source・日別ファイルのhash、保存済み学習尤度、{validation['causal_prefix_checks']}設定の未来を切り落とした予測経路との一致を監査した。学習尤度の最大照合誤差は{validation['maximum_training_loglik_error']:.3g}。",'',
        '[ケース0のOOS結果](arc_sm_symmetric_t_case0_oos.md)。[検定実験の設計・結果](arc_sm_symmetric_t_known_m2_wald.md)。','']
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
    records = [sim.load(p) for p in sorted((sim.OUTPUT/'cases').glob('*.json')) if any(d in p.name for d in DGPS)]
    assert len(records)==3200
    if args.pilot:
        records = [r for r in records if r['entry']['replication']==0 and
            (r['entry']['case'],r['entry']['N']) in (('iid',1),('shared_factor',100))]
    metadata = {'source_sha256':source_hash(),'fit_source_sha256':sim.fingerprint(),
        'case0_oos_source_sha256':base.source_hash(),'metrics':list(METRICS),'settings':len(records)}
    if (output/'design.json').exists(): assert sim.load(output/'design.json')==metadata
    else: sim.dump(output/'design.json',metadata)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate,r,output,source_hash()) for r in records]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%24==0 or len(results)==len(records): summary,state = summarize(output,results,len(records))
    if not args.pilot:
        validation = audit(output,results)
        report(output,summary,state,validation)
    print('OOS ALL COMPLETE',json.dumps(state),flush=True)


if __name__=='__main__':
    main()
