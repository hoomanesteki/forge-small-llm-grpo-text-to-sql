# Vendored from https://github.com/taoyds/test-suite-sql-eval at commit
# e97acc546ecbee8fa27fa8dbf025ef61493a876c, Apache-2.0 (see LICENSE).
# Changed for Forge, see CHANGES.md: relative import, raw-string regex, read-only
# connection with an authorizer and limits, a timeout that interrupts sqlite, a bounded
# fetch, no prints.
import os
import re
import asyncio
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Tuple, Any, List, Set
from itertools import product
from collections import defaultdict
import tqdm
import random
from .parse import get_all_preds_for_execution, remove_distinct
import time
import pickle as pkl
import subprocess
from itertools import chain



threadLock = threading.Lock()
TIMEOUT = 60
EXEC_TMP_DIR = 'tmp/'
# Forge: sqlite VM instructions between two deadline checks in exec_on_db_
PROGRESS_STEPS = 1000
# Forge: what get_cursor_from_path allows, the same as forge.sql.executor (a test keeps
# them equal); copied so this package imports without Forge. Default deny: only reads
# pass the authorizer. The limits keep each step cheap, and with no attached database
# allowed, ATTACH and VACUUM INTO can't create files on a read-only connection.
ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
LIMITS = {
    sqlite3.SQLITE_LIMIT_LENGTH: 4_096,
    sqlite3.SQLITE_LIMIT_SQL_LENGTH: 8_192,
    sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH: 100,
    sqlite3.SQLITE_LIMIT_COLUMN: 200,
    sqlite3.SQLITE_LIMIT_ATTACHED: 0,
}
# Forge: exec_on_db_ reads a result in batches and gives up past MAX_ROWS rows or
# MAX_RESULT_BYTES of row objects, counted as forge.sql.executor counts them, where
# upstream's fetchall kept everything. Spider's largest gold result is 20,662 rows.
MAX_ROWS = 100_000
MAX_RESULT_BYTES = 64 * 2**20
MAX_VALUE_SIZE = 4 * LIMITS[sqlite3.SQLITE_LIMIT_LENGTH] + sys.getsizeof("\U00010000")


class ResultTooBig(Exception):
    """Forge: a result past MAX_ROWS rows or MAX_RESULT_BYTES of memory."""


def _authorize(action, arg1, arg2, db_name, source):
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def row_size(row: Tuple) -> int:
    # Forge: the tuple, its values and the row's slot in the result list
    return sys.getsizeof(row) + 8 + sum(map(sys.getsizeof, row))


def fetch_bounded(cursor) -> List[Tuple]:
    # Forge: fetchall with the ceilings above; a batch holds at most a quarter of the
    # byte ceiling however big its values are
    width = len(cursor.description or ())
    largest = sys.getsizeof((None,) * width) + 8 + width * MAX_VALUE_SIZE
    batch = max(1, MAX_RESULT_BYTES // (4 * largest))
    rows, size = [], 0
    while True:
        chunk = cursor.fetchmany(batch)
        if not chunk:
            return rows
        rows.extend(chunk)
        size += sum(map(row_size, chunk))
        if len(rows) > MAX_ROWS:
            raise ResultTooBig(f"more than {MAX_ROWS:,} rows")
        if size > MAX_RESULT_BYTES:
            raise ResultTooBig(f"rows take more than {MAX_RESULT_BYTES // 2**20} MiB")


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    # Forge: read-only, so a predicted DROP TABLE fails instead of being committed to the
    # file, and a missing path raises instead of being created as an empty database
    uri = Path(sqlite_path).absolute().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    # Forge: read-only still lets ATTACH and VACUUM INTO create files, and lets one value
    # take gigabytes and seconds, so the executor's authorizer and limits apply here too
    connection.set_authorizer(_authorize)
    for limit, value in LIMITS.items():
        connection.setlimit(limit, value)
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str, timeout: int = TIMEOUT) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    # Forge: a blocking sqlite call can't be cancelled by asyncio.wait_for; a progress
    # handler that returns True aborts the running statement, so the timeout holds
    deadline = time.monotonic() + timeout
    cursor.connection.set_progress_handler(lambda: time.monotonic() > deadline, PROGRESS_STEPS)
    try:
        cursor.execute(query)
        result = fetch_bounded(cursor)
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        # Forge: report an interrupted statement the way upstream reported a timeout
        if isinstance(e, sqlite3.OperationalError) and time.monotonic() > deadline:
            return "exception", TimeoutError
        return "exception", e

async def exec_on_db(
    sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    try:
        return await exec_on_db_(sqlite_path, query, timeout)
    except Exception as e:
        return ("exception", e)


# postprocess the model predictions to avoid execution errors
# e.g. removing spaces between ">" and "="
def postprocess(query: str) -> str:
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


# approximate whether p_str and g_str are semantically equivalent
# db is the database path
# we are going to evaluate whether they are equivalent in all the databases
# that are in the same directory as db
# 0 if denotationally equivalent
# 1 otherwise
# the meaning of each auxillary argument can be seen in the parser definition in evaluation.py
def eval_exec_match(db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool) -> int:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = 'order by' in g_str.lower()

    # find all databases in the same directory
    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename]

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        # we did not add this line in our EMNLP work
        # this reduces "false negatives" when value is substituted
        preds = chain([p_str], preds)

    for pred in preds:

        pred_passes = 1
        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:
            g_flag, g_denotation = asyncio.run(exec_on_db(db_path, g_str))
            p_flag, p_denotation = asyncio.run(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            assert g_flag != 'exception', 'gold query %s has error on database file %s' % (g_str, db_path)

            # wrong if execution fails
            if p_flag == 'exception':
                pred_passes = 0

            # if denotations are not equivalent, the prediction must be wrong
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        # the model prediction has the same denotation as the gold for all databases
        if pred_passes == 1:
            return 1

    # none of the predictions passed
    return 0
