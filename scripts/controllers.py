"""Inspect runnable controllers without issuing NN, simulator or hardware commands."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'v2'))
from controller_registry import load_controller

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',choices=('hybrid','direct','historical-hybrid','pid-fast','pid-native'),default='hybrid')
    args=parser.parse_args()
    if args.controller=='pid-native':
        profile=json.loads((ROOT/'v2/artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.public.json').read_text())
        output=dict(controller='pid-native',architecture='pid',implementation='External ArduCopter SITL; not a local NumPy emulation',parameters=profile['parameters'])
    else:
        model=load_controller(args.controller)
        output=dict(controller=args.controller,architecture=model.architecture,
            contract=getattr(model,'contract','Analytical Fast reference, not native ArduCopter'),
            stateful_callable='predict_enu'if hasattr(model,'predict_enu')else'predict',
            output_semantics='Normalized motor actions/fractions, not PWM'if args.controller=='direct'else'Thrust/body rates'if args.controller!='pid-fast'else'Analytical Fast PID action',
            original_ZIP_verified_locally=getattr(model,'original_ZIP_verified_locally',False))
    output.update(new_NN_forwards=0,new_environment_steps=0,optimizer_steps=0,Torch_imported='torch'in sys.modules)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
