"""Frozen observational prediction; no tuned threshold or subgroup override."""
import numpy as np
from pair_math import NAMES

def auc(score,positive):
 s=np.asarray(score,dtype=np.float64);y=np.asarray(positive,dtype=bool)
 pos=int(y.sum());neg=len(y)-pos
 if not pos or not neg or not np.isfinite(s).all():return None
 order=np.argsort(s,kind='stable');values=s[order];labels=y[order]
 ranks=np.empty(len(s),np.float64);lo=0
 while lo<len(s):
  hi=lo+1
  while hi<len(s) and values[hi]==values[lo]:hi+=1
  ranks[lo:hi]=(lo+1+hi)/2;lo=hi
 return float((ranks[labels].sum()-pos*(pos+1)/2)/(pos*neg))

def summarize(scores,labels,predictions,numerical):
 scores=np.asarray(scores);labels=np.asarray(labels)
 q=predictions['quantum_distinct']==labels;g=predictions['gaussian_covariance']==labels
 discord=q!=g;positive=g[discord];rows=scores[discord]
 au={name:auc(rows[:,j],positive) for j,name in enumerate(NAMES[:4])}
 missing=int((~np.isfinite(rows[:,:4]).all(-1)).sum())
 counts=dict(discordants=int(discord.sum()),Gcov_only=int(positive.sum()),Qdistinct_only=int((~positive).sum()),
  both_correct=int((q&g).sum()),both_wrong=int((~q&~g).sum()),primary_undefined=missing,
  all_population_undefined=int((~np.isfinite(scores[:,:4]).all(-1)).sum()))
 sample=counts['discordants']>=100 and counts['Gcov_only']>=30 and counts['Qdistinct_only']>=30 and missing==0
 controls=[au[k] for k in NAMES[1:4]]
 advantage=au['S']-max(controls) if au['S'] is not None and all(x is not None for x in controls) else None
 gates=dict(technical_numeric=bool(numerical),population=bool(sample),
  main_auc_at_least_055=au['S'] is not None and au['S']>=.55,
  advantage_at_least_002=advantage is not None and advantage>=.02)
 strata=[];length=scores[:,2]
 for c in range(40):
  for long in (False,True):
   pick=(labels==c)&((length>35)==long);dd=pick&discord
   strata.append(dict(gold_class=c,length_band='>35' if long else '<=35',rows=int(pick.sum()),
    Gcov_only=int((dd&g).sum()),Qdistinct_only=int((dd&q).sum()),
    S_auc=auc(scores[dd,0],g[dd]),undefined=int((~np.isfinite(scores[dd,:4]).all(-1)).sum())))
 if not numerical:decision='TECHNICAL_NONADMISSION'
 elif all(gates.values()):decision='EXPLORATORY_ASSOCIATION_ONLY_NO_METHOD_ADMISSION'
 else:decision='STOP_THIS_EXACT_ALIASING_PREDICTION'
 return dict(decision=decision,gates=gates,counts=counts,auc=au,advantage_over_best_control=advantage,
  gold_class_length_strata=strata,scope='Already exposed4096 TRAIN/all40. Association only, no causal effect, independent confirmation, method admission or newT.',
  training_admitted=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
