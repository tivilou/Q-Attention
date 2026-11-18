"""Exact conditioning identities only; no circuit, head, data, or utility test."""
from fractions import Fraction as F
import hashlib
import json


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    projection = [[F(1, 2), F(-1, 2), F(1, 2)], [F(-1, 2), F(-1, 2), F(1, 2)]]
    h = [F(2), F(-1), F(3)]
    anchor = [F(1), F(4), F(-2)]
    b = [F(3, 2), F(-2, 3)]
    ph = [dot(row, h) for row in projection]
    pa = [dot(row, anchor) for row in projection]
    interaction = [x * y for x, y in zip(ph, pa)]
    matrix = [[sum(b[k] * projection[k][i] * projection[k][j] for k in range(2)) for j in range(3)] for i in range(3)]
    direct = dot(b, interaction)
    bilinear = sum(h[i] * matrix[i][j] * anchor[j] for i in range(3) for j in range(3))
    assert direct == bilinear and all(matrix[i][j] == matrix[j][i] for i in range(3) for j in range(3))
    w = [F(2, 3), F(1, 5), F(-1, 7)]
    folded = [w[j] + sum(b[k] * projection[k][j] for k in range(2)) for j in range(3)]
    assert dot(w, h) + dot(b, ph) == dot(folded, h)
    # A token-square branch can be non-affine; an extra linear projection cannot.
    values = [F(x * x) for x in range(3)]
    assert values[2] - 2 * values[1] + values[0] == 2
    # Prospective fixed projection rule: label/data independent, no seed search.
    signs = bytes(1 if hashlib.sha256(('ecqp-projection-v1:' + str(i) + ':' + str(j)).encode()).digest()[0] & 1 else 255
        for i in range(16) for j in range(1024))
    assert len(signs) == 16 * 1024
    assert 68200 + (1028 * 12 + 12) + 40 * 12 == 81028
    assert 81028 + 12 * 16 == 81220
    print(json.dumps(dict(status='PASS_ECQP_EXACT_FEATURE_IDENTITIES', bilinear_value=str(direct),
        symmetric_matrix_exact=True, token_linear_control_folds_into_original_affine_map=True,
        token_square_second_difference='2', fixed_projection_signs_sha256=hashlib.sha256(signs).hexdigest(),
        projection_shape=[16, 1024], projection_entry_scale='1/32', original_parameters=81028,
        added_parameters=192, candidate_parameters=81220,
        model_forwards=0, circuit_evaluations=0, training=False, utility_evidence='NOT_RUN', L5='NOT_ACCEPTED'), sort_keys=True))


if __name__ == '__main__':
    main()
