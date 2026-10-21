import sys,json,copy
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'arc_sm_joint_location_scale_hansen_parity'))
import audit_case1_symmetric_t_daily_mean_wald as auditor
import run_case1_training_fixed_pca_wald as experiment
folder=experiment.corrected.REPO/'work/communication_services_pc1_m3_m5_reoptimization'
prior=json.loads((folder/'before_estimation.json').read_text())
search=json.loads((folder/'M3_search.json').read_text())
best=min((r for r in search['starts'] if r['accepted']),key=lambda r:r['mean_nll'])
current=copy.deepcopy(prior)
theta=np.asarray(best['theta']); full=np.zeros(6);full[[0,2,4]]=theta
units=np.asarray(current['models']['M3']['parameter_units'])
current['models']['M3']['coefficients']=(full.reshape(3,2)*units).tolist()
current['models']['M3']['unrestricted']['theta']=full.tolist()
runs,counts=auditor.training_runs(current)
run=experiment.corrected.base.active_run(runs['M3'],[0,2,4])
f=run(theta,True)
print('state_range', f[0][:,2].min(), f[0][:,2].max())
print('total_score',f[4].sum(axis=0))
for h in (1e-4,1e-5,1e-6,5e-7):
 H=np.empty((3,3))
 for j in range(3):
  d=np.zeros(3);d[j]=h
  H[:,j]=-(run(theta+d,True)[4].sum(axis=0)-run(theta-d,True)[4].sum(axis=0))/(2*h)
 H=(H+H.T)/2
 print('step',h,'eigenvalues',np.linalg.eigvalsh(H).tolist())
data=np.load(root/current['dataset']['input'])
k=experiment.corrected.archived.kernels;anchor=np.asarray(current['models']['M1']['state']);dates=data['dates'];offsets=data['offsets'];values=data['values']
fit2=current['models']['M2']
full2=k.reference_filter(values,offsets,anchor,np.asarray(fit2['coefficients']),np.asarray(fit2['parameter_units']),np.zeros((len(dates),4)),np.zeros((2,2)),False,False)
g3=experiment.corrected.archived.shape.shape_geometry(anchor)
full3=k.reference_filter(values,offsets,anchor,np.asarray(current['models']['M3']['coefficients']),units,full2[0],g3.projection_shape_on_ms,True,False)
print('full_state_range',full3[0][:,2].min(),full3[0][:,2].max())
