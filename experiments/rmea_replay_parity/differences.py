"""Exact diagnostic comparison, with no tolerance or acceptance override."""
import math


def differences(expected, actual, path='$'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        rows=[]
        for key in sorted(set(expected)|set(actual)):
            if key not in expected or key not in actual:
                rows.append(dict(path=path+'.'+key,kind='missing_key',
                    expected_present=key in expected,actual_present=key in actual))
            else: rows.extend(differences(expected[key],actual[key],path+'.'+key))
        return rows
    if isinstance(expected,list) and isinstance(actual,list):
        rows=[]
        if len(expected)!=len(actual):rows.append(dict(path=path,kind='length',expected=len(expected),actual=len(actual)))
        for i,(e,a) in enumerate(zip(expected,actual)):rows.extend(differences(e,a,path+'['+str(i)+']'))
        return rows
    for value in [expected,actual]:
        if isinstance(value,float):assert math.isfinite(value),'Nonfinite diagnostic metric'
    if type(expected) is type(actual) and expected==actual:return []
    row=dict(path=path,kind='value' if type(expected) is type(actual) else 'type',expected=expected,actual=actual)
    if type(expected) in (int,float) and type(actual) in (int,float):row['delta']=actual-expected
    return [row]
