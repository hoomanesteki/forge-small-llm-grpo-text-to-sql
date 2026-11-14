# Spike FG-101: Spider sources, license and the official evaluator

Spider 1.0 can be downloaded by a script with no account, from a pinned Hugging Face
revision with a known sha256. The license is CC BY-SA 4.0. The official evaluator is
Apache-2.0, now vendored in `third_party/test_suite_eval/`, and runs in-process.
`train_others` shares no database with dev, so it goes into training, but its databases
are empty, which matters for execution rewards.

- Date: 2026-11-14. Timebox: 3 hours.
- Checked locally: everything under "Local checks", against the zip and the extracted
  copy in `data/spider/` (`FORGE_DATA_DIR`), with Python 3.12.13 and SQLite 3.53.1.
- Taken from the source pages and not downloaded here: the official Google Drive zip
  (205.8 MB) and the test-suite databases (about 1.2 GB). Those numbers are marked.

## 1. Download source

**Use** the Hugging Face dataset `xlangai/spider`, from the Spider authors' lab, at a
pinned revision:

| Field | Value |
|---|---|
| Repo | `xlangai/spider` (dataset, not gated, license tag `cc-by-sa-4.0`) |
| Revision | `6232cc3fad6d54c62b3ba23a364083a98ff36a17` (last commit before the parquet conversion) |
| File | `data/spider.zip`, 99,736,136 bytes |
| sha256 | `5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1` |
| URL | <https://huggingface.co/datasets/xlangai/spider/resolve/6232cc3fad6d54c62b3ba23a364083a98ff36a17/data/spider.zip> |

A programmatic download works without a token. An anonymous HEAD request on 2026-11-14
returned `x-linked-size: 99736136` and the sha256 above as `x-linked-etag`, and the
downloaded file matches it. With huggingface-hub 2.4.0:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="xlangai/spider",
    repo_type="dataset",
    filename="data/spider.zip",
    revision="6232cc3fad6d54c62b3ba23a364083a98ff36a17",
)
```

The current `main` of `xlangai/spider` only has parquet questions (7,000 train rows, no
`train_others`, no `tables.json`, no databases), so the revision pin is required. Keep a
manual path too (`forge data fetch --from-zip`), for when the Hub is unreachable.

**Fallbacks, in order** (paths inside start with `spider_data/` instead of `spider/`):

1. Hugging Face `HAL-9001/spider-databases`, file `spider_data.zip`, revision
   `4a01bbac6520cd35b216db9e1724e5e1ada60aa4`, 205,800,266 bytes, sha256
   `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`. A third-party
   re-host (created 2026-06-09) of the official Drive file. Its size and sha256 were
   confirmed by a HEAD request; the file was not downloaded.
2. The official Google Drive file linked from <https://yale-lily.github.io/spider>:
   id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, `spider_data.zip`, 205,800,266 bytes (from
   the Drive page). It adds the test set (`test.json`, `test_database/`). Drive puts a
   virus-scan page in front of files this size and throttles popular files, so a
   script needs `gdown` with retries, and may still fail. The older id
   `1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m` returns 404 (taoyds/spider issue #104).

Comparing the two zips' central directories (read with range requests) found the same
CRC32 and size for all 398 files they share, including all 166 `.sqlite` files.

### File list and sizes (`spider.zip`, read locally)

The zip has 575 entries (399 files, 176 folders), 918,683,783 bytes uncompressed.

| Path | Bytes | Needed |
|---|---|---|
| `spider/train_spider.json` | 24,928,884 | yes |
| `spider/train_others.json` | 8,514,621 | yes |
| `spider/dev.json` | 3,629,512 | yes |
| `spider/tables.json` | 810,971 | yes |
| `spider/README.txt` | 5,944 | yes (citations) |
| `spider/train_gold.sql` | 1,176,039 | no (same SQL as the JSON) |
| `spider/dev_gold.sql` | 124,056 | no |
| `spider/.DS_Store` | 6,148 | no |
| `spider/database/<db>/<db>.sqlite` (166 files) | 473,346,048 | yes |
| `spider/database/**/*.sql` (157 schema files) | 390,577,215 | no |
| other files in `database/` (csv, txt, json, xml, one `.py`) | 15,564,345 | no |

`database/` also holds an empty `formula_1/formula_1.splite`, so extraction should
match `*/database/<db>/<db>.sqlite` exactly. Extracting only the needed files takes
about 488 MB on disk.

sha256 of the extracted JSON files, for `data/spider.sha256` in FG-103:

| File | sha256 |
|---|---|
| `train_spider.json` | `c43d0d72e59e1a9e1a60837da9bf70d5a6277226bdb7f634d544f380646f527a` |
| `train_others.json` | `7adb04af470b3c9be653504e03c9a36c1b963a861f308ecf25d436472284e10f` |
| `dev.json` | `30d64a3fccde493226df79687aed9e4a1c0129525baf44f29c0573d914d758a4` |
| `tables.json` | `61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed` |

### SQLite files

All 166 files start with `SQLite format 3`. 164 use 4,096-byte pages; `company_1` uses
1,024 and `icfp_1` 2,048. The "last written by" version in the header varies:

| SQLite version | Files | Examples |
|---|---|---|
| 3.13.0 | 150 | most databases |
| 3.22.0 | 7 | `car_1`, `flight_2` (dev), `csu_1`, `formula_1`, `inn_1`, `student_1`, `wine_1` |
| 3.11.0 | 4 | `wta_1` (dev), `soccer_1`, `epinions_1`, `twitter_1` |
| 3.15.2 | 2 | `flight_4`, `small_bank_1` |
| 3.20.1 | 1 | `voter_1` (dev) |
| not set (written before 3.7.0) | 2 | `company_1`, `icfp_1` |

Every gold query that is valid SQL runs on SQLite 3.53.1, so none of these versions is a
problem. The largest files are `soccer_1` (316.9 MB, train), `wta_1` (104.7 MB, dev) and
`baseball_1` (29.8 MB, train).

## 2. License and attribution

The Spider data is licensed **CC BY-SA 4.0**
(<https://creativecommons.org/licenses/by-sa/4.0/legalcode>), as stated on the Spider
page and the `xlangai/spider` card. The zip itself carries no license file, only the
citations in `README.txt`. The Spider code repo (`taoyds/spider`) is Apache-2.0, a
separate license.

What this means for Forge:

- Training and publishing models is allowed, with attribution. Whether weights count as
  adapted material under CC BY-SA is not settled, so the model card attributes Spider
  and names its license.
- Any data Forge redistributes that is built from Spider (the fixture databases, an SFT
  file, eval outputs that include questions) stays CC BY-SA 4.0 with attribution. Forge's
  own code stays MIT.
- `tests/fixtures/spider_mini/README.md` carries the attribution for the fixture.

Attribution text for the README and model card:

> Spider 1.0 by Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li,
> James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang and Dragomir Radev
> (Yale LILY), licensed under CC BY-SA 4.0. Yu et al., "Spider: A Large-Scale
> Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL
> Task", EMNLP 2018, pages 3911-3921, doi:10.18653/v1/D18-1425.

Spider's README also asks for Finegan-Dollak et al. 2018, "Improving Text-to-SQL
Evaluation Methodology" (ACL 2018, P18-1033), and the original sources of the
`train_others` databases (SQLizer for IMDB and Yelp, Li and Jagadish 2014 for Academic,
Iyer et al. 2017 for Scholar and Geography, Zelle and Mooney 1996 for Geography, and
Tang and Mooney 2000, Popescu et al. 2003 and Giordani and Moschitti 2012 for
Restaurants). Their BibTeX entries are in `README.txt` inside the zip.

## 3. The official test-suite evaluator

| Question | Answer |
|---|---|
| Repo | <https://github.com/taoyds/test-suite-sql-eval>, branch `master`, commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c` (2022-08-03; the last code change was 2021-04-11) |
| License | Apache-2.0. The LICENSE file is the unfilled template (no copyright line) and there is no NOTICE file. Modified files must say so (section 4b) |
| Vendored | `third_party/test_suite_eval/`: `evaluation.py`, `exec_eval.py`, `process_sql.py`, `parse.py`, `LICENSE`, plus `__init__.py`. It ships in the wheel as the top-level package `test_suite_eval`. Every change is listed in its `CHANGES.md` |
| In-process call | `test_suite_eval.exec_eval.eval_exec_match(db, p_str, g_str, plug_value, keep_distinct, progress_bar_for_each_datapoint) -> int` (all arguments required). It returns 1 if the prediction matches the gold on every `.sqlite` file in `db`'s folder, else 0. With the original Spider databases there is one file per folder, so that is classical execution accuracy |
| CLI | `python -m test_suite_eval.evaluation --gold dev_gold.sql --pred pred.txt --db <database dir> --etype exec` (add `--plug_value` or `--keep_distinct` as needed; `--table tables.json` only for `--etype match`). Gold lines are `SQL<TAB>db_id`, prediction lines are SQL. The CLI parses every gold with `get_sql` for its hardness, so it needs `nltk` and its `punkt_tab` data, which Forge does not install; without them it stops with `ModuleNotFoundError` (checked). Forge calls the evaluator in-process instead |
| Python | Runs on Python 3.12.13 here, and CI runs the same tests on 3.11 and 3.12. Upstream needed raw strings for 3.12 and relative imports to work as a package |
| Dependencies | `sqlparse` (locked at 0.6.0) and `tqdm`. `nltk` is only needed by `process_sql.get_sql` on a SQL string, and is now imported only there, so it is not a dependency |
| Hardness | `test_suite_eval.evaluation.Evaluator().eval_hardness(sql)` returns easy, medium, hard or extra for a parsed query. Spider's JSON files carry the parsed query as each example's `sql` field, so no parsing is needed |
| Test-suite databases | Google Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w`, `testsuitedatabases.zip`, about 1.2 GB (from the Drive page); the paper (arXiv 2010.02840) gives 695 dev databases, 3.27 GB. They exist only for dev, not for train |

**Are the test-suite databases needed?** Only for the headline number. Plan metric M1 is
"official test-suite EX" on dev, which needs them. Everything before that (val, CI, the
FG-108 agreement check) uses classical EX on the original databases, because val comes
from train databases, which have no test suites. Download them once for the dev
evaluation (FG-508), on Kaggle as a dataset input, and record their sha256 then. Nothing
in Phase 1 needs them.

**Hardness check.** Computed from the `sql` field, dev comes out as 248 easy, 446
medium, 174 hard and 166 extra. That is the split published for Spider dev, so the
`sql` field gives the official labels. Train (8,659) comes out as 1,983 / 2,999 / 1,922 /
1,755.

**Upstream behaviour confirmed locally** on a scratch database with the unmodified code,
and fixed in the vendored copy:

- A predicted `DROP TABLE` was committed to the database file (the connection was
  read-write). Now the connection is `mode=ro` and the write fails.
- A missing database path was printed and then created as an empty file. Now it raises.
- `timeout` did nothing: a recursive CTE with `timeout=1` was still running after 5 s.
  Now a progress handler stops it at the deadline.

**Upstream behaviour left as is** (details in `CHANGES.md`): `eval_exec_match` calls
`asyncio.run`, which fails inside a running event loop such as a notebook cell. An empty
prediction raises `IndexError` in `remove_distinct`. A comment-only prediction scores 1
against a gold query with an empty result (checked locally). `evaluate()` rewrites every
`value` in a prediction to `1`. FG-108 and the rewards must guard these.

## 4. Should `train_others` be in training?

**Yes.** Its 6 databases (`academic`, `geo`, `imdb`, `restaurants`, `scholar`, `yelp`)
share none with dev, and Spider's README defines the official training set as
`train_spider` plus `train_others`. That gives 8,659 training questions over 146
databases.

One caveat changes how it is used. All 6 databases have their tables but **no rows**.
Run against them, every one of the 1,659 gold queries returns an empty result (1,362) or
one row of 0 or NULL (297). So:

- For SFT, the examples are fine: the target is the gold SQL text.
- For execution rewards they carry no signal, since any query that runs and returns
  nothing matches most golds. FG-505 should drop examples whose gold result is empty or
  all 0/NULL from the GRPO set.
- For val, a row-less database would make EX meaningless. FG-104 should pick val only
  from databases that have rows.

`music_2` in `train_spider` (100 questions) has no rows either.

## Local checks

Counts, from the extracted JSON:

| File | Questions | Databases |
|---|---|---|
| `train_spider.json` | 7,000 | 140 |
| `train_others.json` | 1,659 | 6 |
| train total | 8,659 | 146 |
| `dev.json` | 1,034 | 20 |

- Database overlap: train_spider and dev 0, train_others and dev 0, train_spider and
  train_others 0.
- `tables.json` lists 166 databases, exactly the 166 folders in `database/`, and every
  database a split uses has its `.sqlite` file.
- Every example has the keys `db_id`, `query`, `query_toks`, `query_toks_no_value`,
  `question`, `question_toks` and `sql`, and none is missing `sql`.
- Dev databases: `battle_death`, `car_1`, `concert_singer`, `course_teach`,
  `cre_Doc_Template_Mgt`, `dog_kennels`, `employee_hire_evaluation`, `flight_2`,
  `museum_visit`, `network_1`, `orchestra`, `pets_1`, `poker_player`,
  `real_estate_properties`, `singer`, `student_transcripts_tracking`, `tvshow`,
  `voter_1`, `world_1`, `wta_1`.

Gold queries run read-only on the original databases:

| Split | Rows returned | Empty | One row of 0 or NULL | SQLite error |
|---|---|---|---|---|
| train_spider | 6,668 | 254 | 75 | 3 |
| train_others | 0 | 1,362 | 297 | 0 |
| dev | 959 | 49 | 26 | 0 |

The three failing training golds are items 3153 (`assets_maintenance`: no such table
`Ref_Company_Types`), 4513 and 4514 (`document_management`: ORDER BY before INTERSECT).
The evaluator asserts that the gold runs, so FG-103 has to flag these and FG-108 has to
skip them, or gold-vs-gold can't reach 100% if those databases land in val.

## Notes for later stories

- FG-103: fetch with the `hf_hub_download` call above, accept both `spider/` and
  `spider_data/` prefixes, extract only the JSON files and `database/<db>/<db>.sqlite`,
  and verify the sha256 values in this file. Hardness comes from the `sql` field.
- FG-104: choose val from databases with rows.
- FG-106: only `wta_1` (dev) has text that is not valid UTF-8 (2 values), so the
  bad-UTF-8 test needs its own scratch database. The evaluator decodes with
  `errors="ignore"` and the executor with `errors="replace"`; they can differ there.
- FG-108: call `eval_exec_match` directly (not `evaluate()`), from a script or a worker
  thread, never inside a running event loop; reject empty and comment-only predictions
  first; skip the three broken golds.
- FG-505: drop examples whose gold result is empty or all 0/NULL from GRPO.
- FG-508: download the test-suite databases for the headline dev evaluation.

## Sources

- Spider page: <https://yale-lily.github.io/spider>
- Hugging Face `xlangai/spider`: <https://huggingface.co/datasets/xlangai/spider>, files at
  <https://huggingface.co/datasets/xlangai/spider/tree/6232cc3fad6d54c62b3ba23a364083a98ff36a17>
- Hugging Face `HAL-9001/spider-databases`: <https://huggingface.co/datasets/HAL-9001/spider-databases>
- Dead Drive id: <https://github.com/taoyds/spider/issues/104>
- Evaluator: <https://github.com/taoyds/test-suite-sql-eval>, issues #21 (`value`), #25
  (ORDER BY ties), #33 (comment-only predictions)
- Test-suite paper: Zhong, Yu and Klein, "Semantic Evaluation for Text-to-SQL with
  Distilled Test Suites", EMNLP 2020, <https://arxiv.org/abs/2010.02840>
- Spider paper: <https://aclanthology.org/D18-1425>
