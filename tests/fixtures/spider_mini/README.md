# spider_mini: a small Spider subset for tests

Five Spider 1.0 **training** databases, so CI can run the data, SQL and evaluation code
without the full download. None of them is a dev database. Point the code at this
folder with `FORGE_DATA_DIR=tests/fixtures/spider_mini`.

FG-101 added the databases. FG-103 adds the questions (60 in total) and the loader.

## Contents

| Database | Spider file | Tables | Foreign keys | Train questions (easy / medium / hard / extra) | Bytes | sha256 |
|---|---|---|---|---|---|---|
| `department_management` | train_spider | 3 | 2 | 16 (5 / 7 / 2 / 2) | 28,672 | `14474b9543dcdf6f8bdd1ce129819af8237b54143ccaef1fd0d8b2d5ea55911f` |
| `dorm_1` | train_spider | 5 | 4 | 100 (20 / 26 / 18 / 36) | 24,576 | `2f836de861ed2b67679990d12b57814feda426f88e62509201277a7200a2c20b` |
| `flight_1` | train_spider | 4 | 3 | 96 (22 / 48 / 16 / 10) | 36,864 | `1b2414f44c04f84bbe30b4dee2eac4c0f24eb39829d694a40e779e8eb069cd01` |
| `college_1` | train_spider | 7 | 9 | 164 (26 / 64 / 38 / 36) | 53,248 | `5bf3ee598b9f10b4d9bef9a2d390f1391bccbcb64986c88241fad24452695d18` |
| `geo` | train_others | 7 | 6 | 564 (224 / 32 / 222 / 86) | 57,344 | `becb9fc6faf0d87b2b396100d962ed6d9de0fd4e5e03ad102bacec73529ffbc8` |

Each file is `database/<db>/<db>.sqlite`, byte-identical to the one in `spider.zip` from
the Hugging Face dataset `xlangai/spider`, revision
`6232cc3fad6d54c62b3ba23a364083a98ff36a17` (zip sha256
`5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1`). Hardness is the
official evaluator's, computed from each example's parsed `sql` field.

`geo` has its tables but no rows, like every `train_others` database, so every gold
query on it returns an empty result or a single row of 0 or NULL. It is here so tests
cover both training files and that case. The other four have data in every table, and
all of their gold queries run.

## License and attribution

The Spider data is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode). These files
are redistributed unchanged under the same license. Spider is by Tao Yu, Rui Zhang, Kai
Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao,
Shanelle Roman, Zilin Zhang and Dragomir Radev (Yale LILY,
<https://yale-lily.github.io/spider>).

```bibtex
@inproceedings{yu-etal-2018-spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
               Semantic Parsing and Text-to-{SQL} Task},
  author    = {Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu
               and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle
               and Zhang, Zilin and Radev, Dragomir},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language
               Processing},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {3911--3921},
  doi       = {10.18653/v1/D18-1425},
  url       = {https://aclanthology.org/D18-1425}
}
```

`geo` comes from the GeoQuery data as prepared by Finegan-Dollak et al. (2018),
"Improving Text-to-SQL Evaluation Methodology", ACL 2018, pages 351-360
(<https://aclanthology.org/P18-1033>). Spider's README also asks users to cite the
original sources: Zelle and Mooney (1996), "Learning to Parse Database Queries Using
Inductive Logic Programming", AAAI 1996, and Iyer et al. (2017), "Learning a Neural
Semantic Parser from User Feedback", ACL 2017 (<https://aclanthology.org/P17-1089>).
