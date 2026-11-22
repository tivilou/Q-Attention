# Native threshold numerator sampling: planning only

For valid token count T and native z_r=b_r+sum u_rt(p_rt+EPS)/sum(p_rt+EPS),
let a_t=b_r+u_rt-s. The denominator is positive, so sign(z_r-s) equals
sign(C_r) for C_r=mean a_t(p_rt+EPS). For t~q and B|t~Bernoulli(p_rt),
X=a_t B/(T q_t) has mean mu=mean a_t p_rt and second moment
sum a_t^2 p_rt/(T^2 q_t). Add EPS mean a_t exactly. q_t=0 is permitted
only for a_t=0; allzero a gives identicallyzero X/C, no strictdecision.

The support range includes0 and every a_t/(T q_t) even when an endpoint
haszero physicalprobability, a conservative knownbound. Uniformq gives X=a_tB.
q proportionalto |a_t| minimizes max|a_t|/(Tq_t) withvalue mean|a_t|,
but doesnotgenerally minimize actualvariance or certifyshotcost improvement.
These are familiar importance-sampling and fractional-sign identities, not newtheory.

The displayed Bernstein-shaped planning width substitutes POPULATIONvariance
for the random empiricalsamplevariance. It is NOT a confidence bound, expected
stoppingtime, coverage guarantee, algorithmiclowerbound or proof ofpossible/
impossibleclassification. No Bernoulli shotsare drawn. All40 meancontrasts
clearing this width is solely a scaleindicator. Oracle midpoint ofexacttop2
native logits favorsplanning but isnot claimed optimal and isnot a usablepilot.

Fixedfirst128TRAIN/all40relations/alloriginalwords, no targetarray/VALID/TEST.
One actual sourceforward, exactnativevalue/probability/thresholdidentity parity,
and an enumerated four-token Bernoullifixture validate the onlynewcalculations.
Nooptimizer/checkpoint/COS writes or wholelifecyclematrix. Privateperrow scale
arrays stayGPU; publicreport containsquantiles/counts/inputhashes/seal only.

No VALIDexperimentaccepted, sampledshotimprovement, originalL4inheritance orL5.
Future source-specific confidencecertificates, if warranted, require separate
prospective fixedscope andreview; sampling/classicalfeature/preparation/readout
costs mustbe counted before any hardwareefficiencyclaim.
