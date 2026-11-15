"""Official Spider test-suite evaluator, vendored for Forge.

Source: https://github.com/taoyds/test-suite-sql-eval at commit
e97acc546ecbee8fa27fa8dbf025ef61493a876c, Apache-2.0 (see LICENSE). The changes from
upstream and the reason for each are listed in CHANGES.md.

Entry points Forge uses:

- ``exec_eval.eval_exec_match(db, p_str, g_str, plug_value, keep_distinct,
  progress_bar_for_each_datapoint)`` returns 1 when the prediction's results match the
  gold query's on every ``.sqlite`` file in ``db``'s folder, else 0.
- ``evaluation.Evaluator().eval_hardness(sql)`` returns easy, medium, hard or extra for
  a parsed query, such as the ``sql`` field of a Spider example.
"""
