import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

folder=Path(sys.argv[1])
def read(name):return [json.loads(s) for s in (folder/name).read_text().splitlines() if s.strip()]
a=read('actuators.jsonl');p=read('physics.jsonl');g=read('groundtruth.jsonl')
print('counts',len(a),len(p),'frameunique',len(set(x['frame'] for x in a)), 'dt',np.unique(np.round(np.diff([x['timestamp'] for x in p]),6),return_counts=True))
print('first',[(x['frame'],p[i]['timestamp']) for i,x in enumerate(a[:3])], 'last',[(x['frame'],p[len(a)-3+i]['timestamp']) for i,x in enumerate(a[-3:])])
tr=np.array([x['sim_time'] for x in g]);R=Rotation.from_quat([x['quaternion_xyzw'] for x in g]);r=(R[:-1].inv()*R[1:]).as_rotvec()/np.diff(tr)[:,None];tm=(tr[1:]+tr[:-1])/2
pt=np.array([x['timestamp'] for x in p]);pr=np.array([x['imu']['gyro'] for x in p])*[1,-1,-1]
starts=[i for i,x in enumerate(a) if x['override'] and (i==0 or not a[i-1]['override'])]
for i in starts:
    j=i
    while j+1<len(a) and a[j+1]['override']:j+=1
    on=p[i]['timestamp'];off=p[j]['timestamp'];shift=np.array(a[i]['output_pwm'])-a[i]['input_pwm']
    print('pulse',i,j,on,off,'shift',shift)
    for name,ts,rates in [('physics',pt,pr),('pose',tm,r)]:
        pre=rates[(ts>=on-.08)&(ts<on)].mean(axis=0)
        for lo,hi in [(0,.02),(.02,.04),(.04,.08),(.08,.1),(.1,.2)]:
            mask=(ts>=on+lo)&(ts<on+hi)
            print(name,lo,hi,'n',mask.sum(),'delta',np.round(rates[mask].mean(axis=0)-pre,5) if mask.any() else None)
