import json,math
from experiments.rmea_replay_parity.differences import differences

assert differences({'ce':1.,'count':[2,3]}, {'ce':1.,'count':[2,3]})==[]
assert differences({'ce':1.}, {'ce':math.nextafter(1.,2.)})[0]['path']=='$.ce'
assert differences({'count':[2,3]}, {'count':[2,4]})==[dict(path='$.count[1]',kind='value',expected=3,actual=4,delta=1)]
assert differences({'x':1}, {})==[dict(path='$.x',kind='missing_key',expected_present=True,actual_present=False)]
assert differences([1,2],[1])[0]['kind']=='length'
assert differences(1,1.)[0]['kind']=='type'
for value in [float('nan'),float('inf')]:
 try:differences(1.,value)
 except AssertionError:pass
 else:raise AssertionError('Nonfinite metric hidden')
print(json.dumps(dict(status='PASS_EXACT_DIFFERENCE_DIAGNOSTICS',cases=8,tolerance=0,torch_imported=False)))
