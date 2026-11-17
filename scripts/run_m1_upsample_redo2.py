"""Checkpointed M1b alpha=.3 and dense alpha=1 redo using fast WERM fits."""
from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]="1"
os.environ.setdefault("XLA_FLAGS","--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax
import jax.numpy as jnp
import numpy as np
from joblib import Parallel, delayed
import WERM.model
from WERM.config import η_to_ψ
from WERM.experiment_utils import data_hash, evaluate_changepoints, trial_rng, weighted_erm_joint_cv
from WERM.signal_priors import SparseGaussianSignal

P=200; MAX_SEGMENTS=4; NOISE_SCALE=float(np.sqrt(.1)); FRACTIONAL_CHANGE_POINTS=np.array([.35,.65])
DELTA_FRACTION=1/20; NUM_FOLDS=2; L1_CONSTANTS=(.03,.3,3.); FIXED_L1_CONSTANT=.3; HUBER_DELTA=1.35
DEFAULT_RESULT_DIR=ROOT/"results"/"M1_upsample_redo2_fastWERM_2fold_100trials_v1"
METHODS={
 "alpha03":("McScan","WERM-unp","WERM-L1-jointCV","WERM-L1-C0.3-CV","DCDP","DPDU","MOSEG"),
 "alpha10":("McScan","WERM-unp"),
}
EXPERIMENTS={
 "alpha03":dict(title="M1b alpha=0.3",alpha=.3,delta_list=np.arange(1.,10.),seed_key="M1b-alpha-0.3"),
 "alpha10":dict(title="M1b alpha=1.0 dense redo",alpha=1.,delta_list=np.arange(2.,10.),seed_key="M1b-alpha-1.0-redo"),
}

def huber(prediction,truth,epsilon=HUBER_DELTA):
 residual=prediction-truth; absolute=jnp.abs(residual)
 return jnp.where(absolute<=epsilon,.5*residual**2,epsilon*(absolute-.5*epsilon))
def absolute_loss(prediction,truth): return jnp.abs(prediction-truth)
def _design_cholesky(p=P,rho=.4):
 i=np.arange(p); return np.linalg.cholesky(rho**np.abs(i[:,None]-i[None,:]))
DESIGN_CHOLESKY=_design_cholesky()
def _clean(values):
 values=np.asarray(values).reshape(-1); return np.asarray([v for v in values if v != -1],dtype=int)

def _run_werm(X,y,penalty_norm,constants):
 selected,cp,diagnostics=weighted_erm_joint_cv(
  X=X,y=y,M=huber,M_scorer=absolute_loss,Lmax=MAX_SEGMENTS,delta_fraction=DELTA_FRACTION,
  l1_constants=tuple(constants),num_folds=NUM_FOLDS,n_jobs=1,standardize_y=True,
  selection_strategy="exhaustive",return_diagnostics=True,penalty_norm=penalty_norm,
  coefficient_loss="huber",huber_delta=HUBER_DELTA,
  coefficient_solver_options={"tol":1e-5,"max_iter":50000})
 pair=tuple(selected); candidates=[tuple(x) for x in diagnostics["candidates"]]
 return dict(change_points=np.asarray(cp,dtype=int).reshape(-1),selected_segments=int(selected[0]),
  selected_num_chgpts=int(selected[0]-1),selected_constant=float(selected[1]),
  cv_score=float(diagnostics["scores"][candidates.index(pair)]/len(y)),
  candidate_fold_fits=int(diagnostics["candidate_fold_fits"]))

def _run_comparison(method,X,y):
 import WERM.comparison as comparison
 if method=="McScan": return _clean(comparison.run_mcscan(X,y.flatten(),num_chgpts=None,method="auto"))
 if method=="DCDP": return _clean(comparison.DCDP(X,y))
 if method=="DPDU": return _clean(comparison.DPDU(X,y,λs=[.01,.1,1,2],ζs=[10.,15.,20.]))
 if method=="MOSEG":
  refined=comparison.MOSEG(X,y,ms=True,num_chgpts=None)[1]
  return np.array([],dtype=int) if refined is comparison.rpy2.rinterface.NULL else _clean(refined)
 raise ValueError(method)

def _run_trial(experiment,i,delta,j):
 spec=EXPERIMENTS[experiment]; rng=trial_rng(spec["seed_key"],i,j); n=int(delta*P)
 signal=SparseGaussianSignal(spec["alpha"],delta,np.sqrt(delta)*np.ones(MAX_SEGMENTS)).sample(P,rng=rng)
 X=rng.normal(0,np.sqrt(1/n),(n,P))@DESIGN_CHOLESKY.T; noise=rng.standard_t(4,size=n)*NOISE_SCALE
 truth=(n*FRACTIONAL_CHANGE_POINTS).astype(int); regimes=η_to_ψ(truth,n); Xs=X*np.sqrt(n)
 theta=Xs@(signal/np.sqrt(n)); y=np.asarray(jax.vmap(WERM.model.q_linear_model,in_axes=(0,0,0))(theta,regimes,noise))
 scale=float(np.std(y)); ys=y/(scale if np.isfinite(scale) and scale>0 else 1.)
 fits={}
 for method in METHODS[experiment]:
  started=time.perf_counter()
  if method=="WERM-unp": fit=_run_werm(Xs,y,"none",(0.,))
  elif method=="WERM-L1-jointCV": fit=_run_werm(Xs,y,"l1",L1_CONSTANTS)
  elif method=="WERM-L1-C0.3-CV": fit=_run_werm(Xs,y,"l1",(FIXED_L1_CONSTANT,))
  else: fit=dict(change_points=_run_comparison(method,Xs,ys),selected_segments=np.nan,selected_num_chgpts=np.nan,selected_constant=np.nan,cv_score=np.nan,candidate_fold_fits=np.nan)
  fit["runtime"]=time.perf_counter()-started; ev=evaluate_changepoints(fit["change_points"],truth,n)
  fit.update(status=ev["status"],haus=ev["haus"],size=ev["size"]); fits[method]=fit
 return dict(experiment=experiment,setting_index=int(i),delta=float(delta),trial_index=int(j),n=n,
  data_hash=data_hash(X,y),true_change_points=truth,fits=fits)

def _atomic_pickle(path,value):
 temporary=path.with_suffix(path.suffix+".tmp")
 with temporary.open("wb") as handle: pickle.dump(value,handle)
 temporary.replace(path)
def _atomic_json(path,value):
 temporary=path.with_suffix(path.suffix+".tmp")
 with temporary.open("w") as handle: json.dump(value,handle,indent=2)
 temporary.replace(path)

def _assemble(experiment,records,num_trials,n_jobs,effective_spec):
 spec=EXPERIMENTS[experiment]; deltas=np.asarray(effective_spec["delta_list"],float); methods=METHODS[experiment]; shape=(len(deltas),num_trials)
 metrics=("haus","size","runtime","selected_segments","selected_num_chgpts","selected_constant","cv_score")
 results=dict(trial_records=records,methods=methods)
 for method in methods:
  for metric in metrics: results[f"{metric}_{method}"]=np.full(shape,np.nan)
  results[f"status_{method}"]=np.full(shape,"not_run",dtype=object); results[f"chgpts_{method}"]=[]
 for record in records:
  i,j=record["setting_index"],record["trial_index"]
  for method in methods:
   fit=record["fits"][method]
   for metric in metrics: results[f"{metric}_{method}"][i,j]=fit[metric]
   results[f"status_{method}"][i,j]=fit["status"]
   results[f"chgpts_{method}"].append((i,j,np.asarray(fit["change_points"],dtype=int)))
 params=dict(experiment=spec["title"],alpha=spec["alpha"],p=P,max_segments=MAX_SEGMENTS,delta_list=deltas,
  noise="Student-t df=4",noise_scale=NOISE_SCALE,num_trials=num_trials,fractional_change_points=FRACTIONAL_CHANGE_POINTS.copy(),
  design_covariance="Toeplitz rho=0.4",fit_loss="Huber epsilon=1.35",cv_score_loss="absolute loss",
  delta_fraction=DELTA_FRACTION,num_folds=NUM_FOLDS,l1_constants=np.asarray(L1_CONSTANTS),fixed_l1_constant=FIXED_L1_CONSTANT,
  selection_strategy="exhaustive full joint CV",coefficient_solver="fast unified weighted solver (FISTA for Huber)",
  n_jobs=n_jobs,random_seed_protocol=f"trial_rng('{spec['seed_key']}', setting, trial)")
 return dict(results=results,params=params)

def run_experiment():
 smoke=os.environ.get("WERM_SMOKE")=="1"
 effective={name:({**spec,"delta_list":np.asarray([spec["delta_list"][0]])} if smoke else spec) for name,spec in EXPERIMENTS.items()}
 num_trials=int(os.environ.get("WERM_NUM_TRIALS",1 if smoke else 100)); n_jobs=max(1,min(int(os.environ.get("WERM_N_JOBS","36")),36))
 if num_trials<1: raise ValueError("WERM_NUM_TRIALS must be positive")
 default=DEFAULT_RESULT_DIR.with_name(DEFAULT_RESULT_DIR.name+"_smoke") if smoke else DEFAULT_RESULT_DIR
 result_dir=Path(os.environ.get("WERM_RESULT_DIR",default)); result_dir.mkdir(parents=True,exist_ok=True)
 checkpoint,status,final=result_dir/"checkpoint.pkl",result_dir/"status.json",result_dir/"results.pkl"
 signature=dict(experiments={name:dict(alpha=spec["alpha"],delta_list=np.asarray(spec["delta_list"]).tolist(),methods=list(METHODS[name])) for name,spec in effective.items()},
  num_trials=num_trials,folds=NUM_FOLDS,l1_constants=list(L1_CONSTANTS),fixed_l1_constant=FIXED_L1_CONSTANT,
  cv_score_loss="absolute",coefficient_solver="fast-unified-weighted-solver-v1")
 if checkpoint.exists():
  with checkpoint.open("rb") as handle: state=pickle.load(handle)
  if state["signature"]!=signature: raise RuntimeError("existing checkpoint has a different experiment signature")
 else: state={"signature":signature,"records":{}}
 tasks=[(name,i,float(delta),j) for name,spec in effective.items() for i,delta in enumerate(spec["delta_list"]) for j in range(num_trials)]
 order=np.random.default_rng(20260901).permutation(len(tasks)); tasks=[tasks[k] for k in order]
 pending=[task for task in tasks if (task[0],task[1],task[3]) not in state["records"]]; started=time.time()
 def write_status(run_state):
  _atomic_json(status,dict(state=run_state,completed=len(state["records"]),total=len(tasks),
   completed_by_experiment={name:sum(k[0]==name for k in state["records"]) for name in effective},
   n_jobs=n_jobs,pid=os.getpid(),elapsed_seconds=time.time()-started,updated=time.time()))
 write_status("running" if pending else "assembling")
 if pending:
  generated=Parallel(n_jobs=n_jobs,backend="loky",batch_size=1,pre_dispatch=n_jobs,return_as="generator_unordered",verbose=10)(
   delayed(_run_trial)(name,i,delta,j) for name,i,delta,j in pending)
  since=0
  for record in generated:
   key=(record["experiment"],record["setting_index"],record["trial_index"]); state["records"][key]=record; since+=1; write_status("running")
   if since>=max(1,n_jobs//2): _atomic_pickle(checkpoint,state); since=0
  _atomic_pickle(checkpoint,state)
 bundles={}
 for name,spec in effective.items():
  records=[state["records"][(name,i,j)] for i in range(len(spec["delta_list"])) for j in range(num_trials)]
  bundles[name]=_assemble(name,records,num_trials,n_jobs,spec)
 bundle=dict(experiments=bundles,run=dict(result_directory=str(result_dir),created=time.time(),signature=signature))
 _atomic_pickle(final,bundle); write_status("complete"); return bundle

if __name__=="__main__": print(run_experiment()["run"]["result_directory"])
