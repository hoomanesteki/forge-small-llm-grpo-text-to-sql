# Changes from upstream

Source: https://github.com/taoyds/test-suite-sql-eval at commit
`e97acc546ecbee8fa27fa8dbf025ef61493a876c` (2022-08-03), licensed Apache-2.0. The
license is kept in `LICENSE`, and each changed file starts with a short notice that
points here (Apache-2.0 section 4b).

Copied: `evaluation.py`, `exec_eval.py`, `process_sql.py`, `parse.py`, `LICENSE`.
Left out: `exec_subprocess.py` (it reads a pickle named by `sys.argv` at import time),
`evaluate_classical.py` (broken upstream: it calls the async `exec_on_db` without
awaiting it), the notebook, `classical_test.pkl`, `tables.json` and the examples.

Added: `__init__.py` (makes the folder the package `test_suite_eval`) and this file.

## Changes

| File | Change | Why |
|---|---|---|
| `exec_eval.py`, `evaluation.py` | `from parse import ...` became `from .parse import ...`, and likewise for `process_sql` and `exec_eval` | Upstream imports are flat and only work with the folder on `sys.path`. A flat `parse` also collides with the PyPI package of that name |
| `exec_eval.py`, `parse.py` | Regex literals became raw strings (`replace_cur_year`, `strip_query`, `extract_typed_value_in_comparison_from_query`) | Python 3.12 emits `SyntaxWarning: invalid escape sequence` for them, which is an error under `-W error`. The patterns are unchanged |
| `process_sql.py` | `from nltk import word_tokenize` moved from module level into `tokenize()` | Execution matching and hardness from Spider's pre-parsed `sql` field then work without nltk and its punkt data. Only `get_sql` on a SQL string needs nltk |
| `exec_eval.py` | `get_cursor_from_path` opens `file:<path>?mode=ro` with `uri=True` | Upstream opens read-write. Python's sqlite3 does not wrap DDL in a transaction, so a predicted `DROP TABLE` was committed and corrupted the database file. Upstream also created an empty database for a missing path; now the connection fails and the query counts as an exception |
| `exec_eval.py` | `get_cursor_from_path` also sets a default-deny authorizer (`ALLOWED_ACTIONS`: select, read, function, recursive) and `LIMITS` with `setlimit`: values of at most 100 kB, LIKE patterns of at most 1,000 characters, SQL of at most 100 kB, and no attached databases. Both are copies of `forge.sql.executor`'s, so this package still imports on its own; a Forge test keeps them equal | A read-only connection still runs `ATTACH` and `VACUUM INTO`, which create the files they name: a prediction could plant an empty `.sqlite` beside a database, and every later evaluation of it then failed on gold. It also let one value take gigabytes and one call run for seconds, past the timeout below. Writes now fail as they are prepared, with `not authorized` |
| `exec_eval.py` | `exec_on_db_` sets a progress handler (every `PROGRESS_STEPS` = 1,000 VM steps) that aborts the statement once `timeout` seconds have passed; `exec_on_db` no longer wraps it in `asyncio.wait_for`, and passes `timeout` through | `asyncio.wait_for` cannot interrupt a blocking sqlite call, so a runaway query hung forever. A timed-out query still returns `("exception", TimeoutError)` as upstream intended. The default stays 60 seconds |
| `exec_eval.py` | Removed the two `print` calls in `get_cursor_from_path` | They wrote to stdout from library code; with the read-only connection the "Openning a new connection" case is an error instead |

Apart from these changes, the `Path` import, the `PROGRESS_STEPS`, `ALLOWED_ACTIONS` and
`LIMITS` constants and the `_authorize` function they use, and the notices, the code is
upstream's, unused imports and globals included.

## Known upstream behaviour left as is

- `evaluate()` and `print_scores()` in `evaluation.py` still print: they are the
  command-line report. Run it as `python -m test_suite_eval.evaluation --gold ...`
  (the relative imports mean `python evaluation.py` no longer works). It parses every
  gold query with `get_sql`, so it needs nltk and its `punkt_tab` data, which Forge does
  not install. Forge calls `eval_exec_match` and `Evaluator.eval_hardness` directly, and
  those print nothing.
- `evaluate()` replaces every `value` in the prediction with `1` before running it,
  which breaks columns named `value` (upstream issue #21). Calling `eval_exec_match`
  directly avoids it.
- `get_schema` (`process_sql.py`) and `isValidSQL` (`evaluation.py`) still open their own
  read-write connections, without the authorizer or the limits, and never close them. A
  missing path is created as an empty file. `get_schema` only reads, so check the path
  first, or compute hardness from the pre-parsed `sql` field. `isValidSQL` runs its SQL
  on a read-write connection and DDL commits there, so `isValidSQL("DROP TABLE t", db)`
  drops the table: never call it on a prediction. Forge calls neither.
- `eval_exec_match` uses `asyncio.run`, which raises inside a running event loop such
  as a notebook cell. Call it from a plain script, a worker thread or a subprocess.
- An empty prediction makes `remove_distinct` raise `IndexError`, and a comment-only
  prediction runs to an empty result that matches an empty gold result (upstream
  issue #33). Callers must guard both.
- `eval_exec_match` rewrites the prediction before it runs it: `postprocess` turns `> =`
  into `>=`, `remove_distinct` drops every `DISTINCT` and keeps only the first statement
  sqlparse finds, and `replace_cur_year` turns `YEAR(CURDATE())` into `2020`. For a
  prediction that starts with `;` the first statement is the bare `;`, which runs to no
  rows. `forge.eval.testsuite` strips what SQLite skips before the first statement, and
  checks the rewritten string as well as the prediction.
