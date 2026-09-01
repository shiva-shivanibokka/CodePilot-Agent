"""The shared pytest parser."""

from __future__ import annotations

from codepilot.sandbox.pytest_parse import parse

ALL_PASS = """
test_fib.py ....                                                         [100%]

============================== 4 passed in 0.42s ==============================
"""

MIXED = """
test_fib.py .F..                                                         [100%]

=================================== FAILURES ==================================
______________________________ test_fib_negative ______________________________
    def test_fib_negative():
>       assert fib(-1) == 0
E       ValueError

========================= 1 failed, 3 passed in 0.51s =========================
"""

NO_SUITE = """
ERROR: file or directory not found: tests/
"""

NOTHING_COLLECTED = """
collected 0 items

============================ no tests ran in 0.01s ============================
"""

COLLECTION_ERROR = """
==================================== ERRORS ===================================
_________________ ERROR collecting tests/test_fib.py __________________________
ModuleNotFoundError: No module named 'fib'
=========================== 1 error in 0.20s ==================================
"""

#: A traceback that prints the words "9 passed". A scan of the whole output
#: takes the last match it finds, which here is the wrong one.
LYING_TRACEBACK = """
=================================== FAILURES ==================================
_______________________________ test_reporting ________________________________
    def test_reporting():
>       assert summarise() == "9 passed"
E       AssertionError: assert '0 passed' == '9 passed'

========================= 1 failed, 2 passed in 0.30s =========================
"""


def test_all_passing():
    r = parse(ALL_PASS, 0)
    assert (r.passed, r.failed, r.errors) == (4, 0, 0)
    assert r.success and not r.no_tests


def test_mixed_results():
    r = parse(MIXED, 1)
    assert (r.passed, r.failed) == (3, 1)
    assert not r.success
    assert "1 failed" in r.summary


def test_missing_directory_is_no_tests_not_a_failure():
    r = parse(NO_SUITE, 4)
    assert r.no_tests
    assert not r.success
    assert r.summary == "no tests collected"


def test_nothing_collected_is_no_tests():
    assert parse(NOTHING_COLLECTED, 5).no_tests


def test_collection_error_is_a_real_error_not_a_missing_suite():
    r = parse(COLLECTION_ERROR, 2)
    assert r.errors == 1
    assert not r.no_tests, "a broken import is a bug to fix, not a missing suite"
    assert not r.success


def test_the_summary_line_wins_over_a_traceback_that_mentions_passed():
    r = parse(LYING_TRACEBACK, 1)
    assert (r.passed, r.failed) == (2, 1), (
        "a count scraped from a traceback beat pytest's own summary line"
    )


def test_a_timeout_is_never_a_success():
    r = parse(ALL_PASS, 0)
    r.timed_out = True
    assert not r.success
    assert r.summary == "timed out"


def test_output_with_no_summary_line_does_not_crash():
    r = parse("segmentation fault", 139)
    assert (r.passed, r.failed, r.errors) == (0, 0, 0)
    assert not r.success and not r.no_tests


# `pytest -q` prints no decoration at all. This is the default command the
# agent runs, so anchoring only on the "==== ... ====" form reported 0 passed
# for every real run while the raw output plainly said otherwise.
QUIET_PASS = """.                                                                        [100%]
1 passed in 0.02s
"""

QUIET_MIXED = """.F                                                                       [100%]
=================================== FAILURES ===================================
_______________________________ test_two _______________________________________
E       assert 1 == 2
1 failed, 1 passed in 0.03s
"""


def test_quiet_mode_summary_is_parsed():
    r = parse(QUIET_PASS, 0)
    assert (r.passed, r.failed) == (1, 0)
    assert r.success


def test_quiet_mode_mixed_summary_is_parsed():
    r = parse(QUIET_MIXED, 1)
    assert (r.passed, r.failed) == (1, 1)
    assert not r.success


def test_quiet_mode_no_tests_still_reads_as_missing():
    assert parse("no tests ran in 0.01s\n", 5).no_tests
