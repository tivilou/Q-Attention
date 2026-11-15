"""Exact arithmetic foil to an invalid objective transfer, not a new method/test.

No inputs, fitting, data access, model forward or numerical optimization.
The commuting two-dimensional ensemble has a simple globally optimal
minimum-error measurement with a strictly smaller MacroF1 than another POVM.
"""
import json
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent

def macro(cm):
    return sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in range(2))/2

def encoded(x):
    if isinstance(x,F):return dict(exact=str(x),decimal=float(x))
    if isinstance(x,dict):return {k:encoded(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encoded(v) for v in x]
    return x

def main():
    # Joint table P(Y,X), rows Y1,Y2, columns orthogonal feature states a,b.
    joint=[[F(72,100),F(11,100)],[F(8,100),F(9,100)]]
    pi=list(map(sum,joint));assert pi==[F(83,100),F(17,100)]
    rho=[[v/pi[i] for v in row] for i,row in enumerate(joint)]
    assert all(sum(r)==1 and min(r)>=0 for r in rho)
    # Minimum-error optimal M1=I, M2=0. Dual X=pi1*rho1.
    # X-pi1*rho1=0 and X-pi2*rho2=diag(.64,.02)>=0.
    dual=joint[0];slacks=[[dual[k]-joint[i][k] for k in range(2)] for i in range(2)]
    assert all(min(row)>=0 for row in slacks)
    cm_error=[[pi[0],F(0)],[pi[1],F(0)]]
    cm_split=joint
    acc_error=sum(cm_error[i][i] for i in range(2));acc_split=sum(cm_split[i][i] for i in range(2))
    assert sum(dual)==acc_error==F(83,100) and acc_split==F(81,100)
    assert macro(cm_error)==F(83,183)
    assert macro(cm_split)==F(4131,6031) and macro(cm_split)>macro(cm_error)
    out=dict(schema='qattn.decision-objective-counterexample.v1',status='PASS',scope='Hypothetical commuting2D ensemble; exact known-objective foil, not Q-RMEA/NLP evidence or new theorem.',joint_y_x=joint,priors=pi,class_conditional_diagonals=rho,minimum_error=dict(povm_diagonals=[[1,1],[0,0]],confusion=cm_error,accuracy=acc_error,macro_f1=macro(cm_error),dual_diagonal=dual,dual_slack_diagonals=slacks,primal_dual_gap=F(0)),alternative=dict(povm_diagonals=[[1,0],[0,1]],confusion=cm_split,accuracy=acc_split,macro_f1=macro(cm_split)),macro_advantage_of_alternative=macro(cm_split)-macro(cm_error),conclusion='A global minimum-error certificate does not imply a MacroF1 optimum. This invalidates that universal transfer, not every quantum attention theory or either method in the real cohort.',not_established=['Novel theory','Task utility','Actual circuit/argmax correspondence','Finite-sample or population improvement for Q-RMEA','Global optimality of the alternative for MacroF1'])
    (P/'decision-counterexample.json').write_bytes((json.dumps(encoded(out),indent=2)+'\n').encode())
    print(json.dumps(dict(status='PASS',minimum_error_accuracy=float(acc_error),alternative_accuracy=float(acc_split),minimum_error_macro=float(macro(cm_error)),alternative_macro=float(macro(cm_split)),primal_dual_gap=0,scientific_data_access=False)))

if __name__=='__main__':main()
