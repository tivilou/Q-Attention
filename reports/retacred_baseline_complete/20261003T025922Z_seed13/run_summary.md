# Re-TACRED Baseline 完整评估

- schema: `q-attention.retacred-baseline-evaluation.v1`
- seed: `13`
- checkpoint: `a7ffa5ed854c9de4e0c49ad7f1260095238fee6795a6b754d667d4d8781bd0db`
- checkpoint selection: `valid macro-F1 then loss`
- test isolation: `test was not used for training or checkpoint selection`

| split | micro-P | micro-R | micro-F1 | accuracy | macro-P | macro-R | macro-F1 | loss | items |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.962063 | 0.962063 | 0.962063 | 0.962063 | 0.893650 | 0.866101 | 0.870789 | 0.124365 | 58465 |
| valid | 0.652880 | 0.652880 | 0.652880 | 0.652880 | 0.359894 | 0.271502 | 0.289704 | 1.690192 | 19584 |
| test | 0.664928 | 0.664928 | 0.664928 | 0.664928 | 0.286991 | 0.227554 | 0.227404 | 1.644870 | 13418 |

## Per-class test metrics

| id | relation | precision | recall | F1 | support |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | no_relation | 0.712355 | 0.854826 | 0.777115 | 7770 |
| 1 | org:alternate_names | 0.377778 | 0.302671 | 0.336079 | 337 |
| 2 | org:city_of_branch | 0.229508 | 0.325581 | 0.269231 | 129 |
| 3 | org:country_of_branch | 0.290043 | 0.403614 | 0.337531 | 166 |
| 4 | org:dissolved | 0.000000 | 0.000000 | 0.000000 | 5 |
| 5 | org:founded | 0.529412 | 0.529412 | 0.529412 | 34 |
| 6 | org:founded_by | 0.000000 | 0.000000 | 0.000000 | 84 |
| 7 | org:member_of | 0.039216 | 0.125000 | 0.059701 | 64 |
| 8 | org:members | 0.058824 | 0.063492 | 0.061069 | 63 |
| 9 | org:number_of_employees/members | 0.000000 | 0.000000 | 0.000000 | 13 |
| 10 | org:political/religious_affiliation | 0.315789 | 0.413793 | 0.358209 | 29 |
| 11 | org:shareholders | 0.000000 | 0.000000 | 0.000000 | 12 |
| 12 | org:stateorprovince_of_branch | 0.215385 | 0.245614 | 0.229508 | 57 |
| 13 | org:top_members/employees | 0.666667 | 0.386441 | 0.489270 | 295 |
| 14 | org:website | 0.083333 | 0.100000 | 0.090909 | 30 |
| 15 | per:age | 0.543103 | 0.302885 | 0.388889 | 208 |
| 16 | per:cause_of_death | 0.500000 | 0.020000 | 0.038462 | 50 |
| 17 | per:charges | 0.275862 | 0.063492 | 0.103226 | 126 |
| 18 | per:children | 0.190476 | 0.145455 | 0.164948 | 55 |
| 19 | per:cities_of_residence | 0.173913 | 0.064000 | 0.093567 | 125 |
| 20 | per:city_of_birth | 0.093750 | 0.200000 | 0.127660 | 15 |
| 21 | per:city_of_death | 0.350000 | 0.269231 | 0.304348 | 26 |
| 22 | per:countries_of_residence | 0.366667 | 0.148649 | 0.211538 | 148 |
| 23 | per:country_of_birth | 0.000000 | 0.000000 | 0.000000 | 0 |
| 24 | per:country_of_death | 0.000000 | 0.000000 | 0.000000 | 14 |
| 25 | per:date_of_birth | 0.217391 | 0.714286 | 0.333333 | 7 |
| 26 | per:date_of_death | 0.444444 | 0.190476 | 0.266667 | 63 |
| 27 | per:employee_of | 0.277512 | 0.174699 | 0.214418 | 332 |
| 28 | per:identity | 0.878226 | 0.651768 | 0.748238 | 2036 |
| 29 | per:origin | 0.437500 | 0.121739 | 0.190476 | 115 |
| 30 | per:other_family | 0.000000 | 0.000000 | 0.000000 | 52 |
| 31 | per:parents | 0.225000 | 0.084906 | 0.123288 | 106 |
| 32 | per:religion | 0.666667 | 0.203390 | 0.311688 | 59 |
| 33 | per:schools_attended | 0.464286 | 0.393939 | 0.426230 | 33 |
| 34 | per:siblings | 0.200000 | 0.030303 | 0.052632 | 66 |
| 35 | per:spouse | 0.066667 | 0.041096 | 0.050847 | 73 |
| 36 | per:stateorprovince_of_birth | 0.388889 | 0.777778 | 0.518519 | 9 |
| 37 | per:stateorprovince_of_death | 0.000000 | 0.000000 | 0.000000 | 16 |
| 38 | per:stateorprovinces_of_residence | 0.520000 | 0.178082 | 0.265306 | 73 |
| 39 | per:title | 0.680995 | 0.575526 | 0.623834 | 523 |
