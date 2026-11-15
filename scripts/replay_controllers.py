"""Full archived causal state→action replay; no environment, Torch or training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'v2'))
from controller_registry import load_controller
from quadrl.policy import NumpyPolicy
from quadrl_phase_expert_v29.policy import StatelessMean80

CASES={'direct':('direct111_hover1500',1500),'hybrid':('phase29_route2000',2000)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def replay(name):
    case,expected_rows=CASES[name]
    path=ROOT/'examples'/(case+'.npz');provenance=json.loads(path.with_suffix('.provenance.json').read_text())
    if sha(path)!=provenance['export_sha256']:raise ValueError('Archived example bytes differ')
    with np.load(path,allow_pickle=False)as archive:io={key:archive[key].copy()for key in archive.files}
    if len(io['time'])!=expected_rows:raise ValueError('Full fixed prefix required')
    policy=load_controller(name);policy.reset(float(io['yaw_reference'][0]))
    errors=dict(action=0.,observation=0.,previous_action=0.);counts=[0,0]
    for i in range(expected_rows):
        errors['previous_action']=max(errors['previous_action'],float(np.max(np.abs(policy.previous_action-io['previous_action'][i]))))
        args=[io[key][i]for key in ('position','velocity','rotation','body_rates','target')]
        result=policy.predict_enu(*args,dt=float(io['dt'][i]))
        errors['observation']=max(errors['observation'],float(np.max(np.abs(result['observation']-io['observation'][i]))))
        errors['action']=max(errors['action'],float(np.max(np.abs(result['action']-io['action'][i]))))
        if name=='hybrid':
            diagnostics=policy.diagnostics();expert=int(io['expert_index'][i]);counts[expert]+=1
            assert diagnostics['expert_index']==expert and diagnostics['valid_leg']==bool(io['expert_valid_leg'][i])
            assert diagnostics['encoder_calls']==int(io['expert_encoder_calls'][i])
            assert diagnostics['selected_source_sha256']==str(io['expert_selected_source_sha'][i])
            assert np.array_equal(policy.encoder.raw_observation,io['raw_observation'][i])
            assert np.array_equal(policy.encoder.effective_observation,io['effective_observation'][i])
    assert max(errors.values())<=1e-5,errors
    if name=='hybrid':
        assert policy.expert_forwards==counts and all(c>0 for c in counts) and sum(counts)==expected_rows
        assert policy.expert_index==1,'Completed-route hold must keep the observed-leg expert'
    return dict(controller=name,archived_rows=expected_rows,max_errors=errors,selected_expert_counts=counts if name=='hybrid'else None,
        backend_of_archived_data=provenance['backend'],example_sha256=sha(path),full_causal_prefix=True,
        original_ZIP_verified_locally=False if name=='hybrid'else None)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controller',choices=('direct','hybrid','all'),default='all')
    args=parser.parse_args();names=list(CASES)if args.controller=='all'else[args.controller]
    budget=sum(CASES[n][1]for n in names);assert budget<=3500
    counter={'NN':0};codes={NumpyPolicy.predict.__code__,StatelessMean80.predict.__code__}
    previous_profile=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call'and frame.f_code in codes:
            counter['NN']+=1
            if counter['NN']>budget:raise RuntimeError('Offline NN replay cap exceeded')
    sys.setprofile(profile)
    try:results=[replay(n)for n in names]
    finally:sys.setprofile(previous_profile)
    assert counter['NN']==budget and 'torch'not in sys.modules and 'pybullet'not in sys.modules
    print(json.dumps(dict(status='PASS_OFFLINE_ARCHIVED_CAUSAL_CONTROLLER_REPLAY',cases=results,
        actual_NN_forwards=counter['NN'],new_environment_steps=0,new_physics_substeps=0,optimizer_steps=0,
        Torch_or_PyBullet_imported=False,independent_quality_replication=False),indent=2))

if __name__=='__main__':main()
