# Fifth-round proof wording correction

The execution-frozen proof says that the two heads' pooled features are
related by a fixed coordinate permutation. At the actual 64-dimensional
value-pooling boundary this should read **an invertible sign change**:
the first coordinate changes sign, the remaining63 coordinates are zero.
The predetermined decoder sign change cancels it. A permutation of the two
active output logits describes the damaged fixed-recipient transplant,
not the relation between the heads' pooled value features.

The original frozen proof and its hash are preserved in the local execution
record. This public copy has a visible correction notice. Source/contract,
two synthetic forward results, CE increment2, maximum error, and the conclusion
about mutual transplant harm remain unchanged. No experiment was rerun and
this correction establishes no novel theorem or L5 acceptance.
