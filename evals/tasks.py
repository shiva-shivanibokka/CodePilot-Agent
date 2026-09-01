"""
The eval task set.

Each task is a small repository plus a prompt plus a **held-out test file the
agent never sees**. Pass means those tests pass afterwards.

That last part is the whole design. Asking the agent whether it succeeded, or
grading it with another model, measures the agent's self-report. Running tests
it could not have written to measures the work.

Three tiers:
  `single`  — one function in one file
  `multi`   — more than one file, or a change plus its tests
  `debug`   — an existing failing suite to diagnose

Tasks are Python because the sandbox runs pytest; the harness is not otherwise
language-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    tier: str
    prompt: str
    #: The starting repository. Written before the agent runs.
    files: dict[str, str] = field(default_factory=dict)
    #: Written *after* the agent finishes. Never visible to it.
    held_out: dict[str, str] = field(default_factory=dict)
    #: Files the agent must not have deleted. Checked after the run — this is
    #: how experiment 2 detects whole-file rewrites dropping unrelated code.
    must_survive: dict[str, list[str]] = field(default_factory=dict)


HELD_OUT = "tests/test_held_out.py"


TASKS: list[Task] = [
    # ---------------------------------------------------------------- single
    Task(
        id="empty-guard",
        tier="single",
        prompt="mean() crashes on an empty list. Make it raise ValueError instead.",
        files={"stats.py": "def mean(values):\n    return sum(values) / len(values)\n"},
        held_out={
            HELD_OUT: (
                "import pytest\nfrom stats import mean\n\n"
                "def test_normal():\n    assert mean([1, 2, 3]) == 2\n\n"
                "def test_empty_raises():\n"
                "    with pytest.raises(ValueError):\n        mean([])\n"
            )
        },
    ),
    Task(
        id="off-by-one",
        tier="single",
        prompt="last_n(items, n) should return the last n items but is off by one. Fix it.",
        files={
            "slicing.py": "def last_n(items, n):\n    return items[-n + 1:]\n",
        },
        held_out={
            HELD_OUT: (
                "from slicing import last_n\n\n"
                "def test_last_two():\n    assert last_n([1, 2, 3, 4], 2) == [3, 4]\n\n"
                "def test_all():\n    assert last_n([1, 2], 2) == [1, 2]\n\n"
                "def test_one():\n    assert last_n([1, 2, 3], 1) == [3]\n"
            )
        },
    ),
    Task(
        id="case-insensitive",
        tier="single",
        prompt="Make find_user() match the username case-insensitively.",
        files={
            "users.py": (
                "USERS = ['Alice', 'BOB', 'carol']\n\n\n"
                "def find_user(name):\n"
                "    for u in USERS:\n        if u == name:\n            return u\n"
                "    return None\n"
            )
        },
        held_out={
            HELD_OUT: (
                "from users import find_user\n\n"
                "def test_exact():\n    assert find_user('Alice') == 'Alice'\n\n"
                "def test_lower():\n    assert find_user('bob') == 'BOB'\n\n"
                "def test_upper():\n    assert find_user('CAROL') == 'carol'\n\n"
                "def test_missing():\n    assert find_user('dave') is None\n"
            )
        },
    ),
    Task(
        id="retry-backoff",
        tier="single",
        prompt=(
            "Add a retry decorator `retry(times)` in retry.py that retries a "
            "function on any exception up to `times` attempts and re-raises the "
            "last error if all attempts fail."
        ),
        files={"retry.py": '"""Retry helpers."""\n'},
        held_out={
            HELD_OUT: (
                "import pytest\nfrom retry import retry\n\n"
                "def test_succeeds_after_failures():\n"
                "    calls = []\n\n"
                "    @retry(3)\n    def flaky():\n"
                "        calls.append(1)\n"
                "        if len(calls) < 3:\n            raise ValueError('nope')\n"
                "        return 'ok'\n\n"
                "    assert flaky() == 'ok'\n    assert len(calls) == 3\n\n"
                "def test_reraises_after_exhausting():\n"
                "    @retry(2)\n    def always():\n        raise KeyError('boom')\n\n"
                "    with pytest.raises(KeyError):\n        always()\n"
            )
        },
    ),
    Task(
        id="parse-duration",
        tier="single",
        prompt=(
            "Write parse_duration(text) in duration.py. It accepts strings like "
            "'90s', '5m', '2h' and returns the number of seconds as an int. "
            "Raise ValueError on anything else."
        ),
        files={"duration.py": '"""Duration parsing."""\n'},
        held_out={
            HELD_OUT: (
                "import pytest\nfrom duration import parse_duration\n\n"
                "def test_seconds():\n    assert parse_duration('90s') == 90\n\n"
                "def test_minutes():\n    assert parse_duration('5m') == 300\n\n"
                "def test_hours():\n    assert parse_duration('2h') == 7200\n\n"
                "def test_bad():\n"
                "    with pytest.raises(ValueError):\n        parse_duration('soon')\n"
            )
        },
    ),
    # ----------------------------------------------------------------- multi
    Task(
        id="extract-helper",
        tier="multi",
        prompt=(
            "report.py duplicates the same currency formatting in three places. "
            "Extract it into a helper in money.py and use it everywhere."
        ),
        files={
            "money.py": '"""Money helpers."""\n',
            "report.py": (
                "def line_total(n):\n    return f'${n:,.2f}'\n\n\n"
                "def subtotal(n):\n    return f'${n:,.2f}'\n\n\n"
                "def grand_total(n):\n    return f'${n:,.2f}'\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "import inspect\nimport money, report\n\n"
                "def test_still_formats():\n"
                "    assert report.line_total(1234.5) == '$1,234.50'\n"
                "    assert report.subtotal(0) == '$0.00'\n"
                "    assert report.grand_total(9.999) == '$10.00'\n\n"
                "def test_helper_exists_and_is_used():\n"
                "    helpers = [n for n, _ in inspect.getmembers(money, inspect.isfunction)]\n"
                "    assert helpers, 'no helper was created in money.py'\n"
                "    source = inspect.getsource(report)\n"
                "    assert 'money' in source, 'report.py does not use the helper'\n"
            )
        },
        must_survive={"report.py": ["line_total", "subtotal", "grand_total"]},
    ),
    Task(
        id="add-validation",
        tier="multi",
        prompt=(
            "Add input validation to Account.withdraw: refuse a negative amount "
            "and refuse to overdraw, raising ValueError in both cases. Do not "
            "change any other behaviour."
        ),
        files={
            "account.py": (
                "class Account:\n"
                "    def __init__(self, balance=0):\n        self.balance = balance\n\n"
                "    def deposit(self, amount):\n        self.balance += amount\n"
                "        return self.balance\n\n"
                "    def withdraw(self, amount):\n        self.balance -= amount\n"
                "        return self.balance\n"
            )
        },
        held_out={
            HELD_OUT: (
                "import pytest\nfrom account import Account\n\n"
                "def test_normal_withdraw():\n"
                "    a = Account(100)\n    assert a.withdraw(40) == 60\n\n"
                "def test_negative_refused():\n"
                "    a = Account(100)\n"
                "    with pytest.raises(ValueError):\n        a.withdraw(-5)\n"
                "    assert a.balance == 100\n\n"
                "def test_overdraw_refused():\n"
                "    a = Account(50)\n"
                "    with pytest.raises(ValueError):\n        a.withdraw(80)\n"
                "    assert a.balance == 50\n\n"
                "def test_deposit_unchanged():\n"
                "    a = Account(0)\n    assert a.deposit(10) == 10\n"
            )
        },
        must_survive={"account.py": ["deposit", "__init__"]},
    ),
    Task(
        id="two-modules",
        tier="multi",
        prompt=(
            "Add a `slugify(text)` function to text_utils.py that lowercases, "
            "replaces runs of non-alphanumeric characters with a single hyphen, "
            "and strips leading and trailing hyphens. Then use it in "
            "pages.py's page_url() instead of the inline logic there."
        ),
        files={
            "text_utils.py": '"""Text helpers."""\n',
            "pages.py": (
                "def page_url(title):\n"
                "    return '/p/' + title.lower().replace(' ', '-')\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "import inspect\nimport pages\nfrom text_utils import slugify\n\n"
                "def test_slugify():\n"
                "    assert slugify('Hello, World!') == 'hello-world'\n"
                "    assert slugify('  A  B  ') == 'a-b'\n\n"
                "def test_page_url_uses_it():\n"
                "    assert pages.page_url('Hello, World!') == '/p/hello-world'\n"
                "    assert 'slugify' in inspect.getsource(pages)\n"
            )
        },
    ),
    Task(
        id="config-defaults",
        tier="multi",
        prompt=(
            "load_config() should merge user settings over DEFAULTS instead of "
            "replacing them, so a partial config keeps the default for anything "
            "it does not mention. Nested dictionaries should merge too."
        ),
        files={
            "config.py": (
                "DEFAULTS = {\n"
                "    'host': 'localhost',\n    'port': 8080,\n"
                "    'log': {'level': 'info', 'file': None},\n}\n\n\n"
                "def load_config(user):\n    return user\n"
            )
        },
        held_out={
            HELD_OUT: (
                "from config import load_config\n\n"
                "def test_fills_defaults():\n"
                "    c = load_config({'port': 9000})\n"
                "    assert c['port'] == 9000\n    assert c['host'] == 'localhost'\n\n"
                "def test_merges_nested():\n"
                "    c = load_config({'log': {'level': 'debug'}})\n"
                "    assert c['log']['level'] == 'debug'\n"
                "    assert c['log']['file'] is None\n\n"
                "def test_empty():\n"
                "    assert load_config({})['host'] == 'localhost'\n"
            )
        },
    ),
    Task(
        id="cli-flag",
        tier="multi",
        prompt=(
            "Add a --upper flag to the CLI in tool.py that uppercases the output "
            "of greet(). Keep the existing behaviour when the flag is absent."
        ),
        files={
            "tool.py": (
                "import argparse\n\n\n"
                "def greet(name):\n    return f'hello, {name}'\n\n\n"
                "def main(argv=None):\n"
                "    p = argparse.ArgumentParser()\n"
                "    p.add_argument('name')\n"
                "    args = p.parse_args(argv)\n"
                "    return greet(args.name)\n"
            )
        },
        held_out={
            HELD_OUT: (
                "from tool import main\n\n"
                "def test_default():\n    assert main(['ada']) == 'hello, ada'\n\n"
                "def test_upper():\n    assert main(['ada', '--upper']) == 'HELLO, ADA'\n"
            )
        },
        must_survive={"tool.py": ["greet"]},
    ),
    # ----------------------------------------------------------------- debug
    Task(
        id="debug-mutation",
        tier="debug",
        prompt="The test suite fails. Find the cause and fix the code, not the test.",
        files={
            "cart.py": (
                "def add_items(cart, items):\n"
                "    cart.extend(items)\n"
                "    return cart\n"
                "\n\n"
                "def totals(carts, extra):\n"
                "    out = []\n"
                "    for cart in carts:\n"
                "        out.append(add_items(extra, cart))\n"
                "    return out\n"
            ),
            "tests/test_cart.py": (
                "from cart import totals\n"
                "\n"
                "def test_each_cart_gets_the_extras():\n"
                "    out = totals([[], ['a']], ['x'])\n"
                "    assert out[0] == ['x']\n"
                "    assert out[1] == ['a', 'x']\n"
                "\n"
                "def test_extras_not_mutated():\n"
                "    extra = ['x']\n"
                "    totals([[], []], extra)\n"
                "    assert extra == ['x'], 'the extras list was mutated'\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "from cart import totals\n"
                "\n"
                "def test_extras_untouched():\n"
                "    extra = ['x', 'y']\n"
                "    totals([[], [], []], extra)\n"
                "    assert extra == ['x', 'y']\n"
                "\n"
                "def test_results_are_distinct_lists():\n"
                "    out = totals([[], ['a']], ['z'])\n"
                "    assert out[0] == ['z']\n"
                "    assert out[1] == ['a', 'z']\n"
                "    assert out[0] is not out[1]\n"
            )
        },
    ),
    Task(
        id="debug-rounding",
        tier="debug",
        prompt="The test suite fails. Diagnose the cause and fix the implementation.",
        files={
            "invoice.py": (
                "def line_total(price, qty):\n    return price * qty\n\n\n"
                "def invoice_total(lines):\n"
                "    return round(sum(line_total(p, q) for p, q in lines), 2)\n"
            ),
            "tests/test_invoice.py": (
                "from invoice import invoice_total\n\n"
                "def test_rounds_each_line_not_just_the_sum():\n"
                "    # Each line should be rounded to cents before summing.\n"
                "    assert invoice_total([(0.005, 1), (0.005, 1)]) == 0.02\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "from invoice import invoice_total, line_total\n\n"
                "def test_line_rounds():\n"
                "    assert line_total(0.005, 1) == 0.01\n\n"
                "def test_total_of_rounded_lines():\n"
                "    assert invoice_total([(0.005, 1), (0.005, 1)]) == 0.02\n\n"
                "def test_plain_case():\n"
                "    assert invoice_total([(2.50, 2), (1.00, 3)]) == 8.0\n"
            )
        },
    ),
    Task(
        id="debug-import",
        tier="debug",
        prompt="The tests fail to even run. Work out why and fix it.",
        files={
            "pkg/__init__.py": "",
            "pkg/core.py": "from helpers import double\n\n\ndef quadruple(n):\n    return double(double(n))\n",
            "pkg/helpers.py": "def double(n):\n    return n * 2\n",
            "tests/test_core.py": (
                "from pkg.core import quadruple\n\n"
                "def test_quadruple():\n    assert quadruple(3) == 12\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "from pkg.core import quadruple\nfrom pkg.helpers import double\n\n"
                "def test_double():\n    assert double(5) == 10\n\n"
                "def test_quadruple():\n    assert quadruple(2) == 8\n"
            )
        },
    ),
    Task(
        id="debug-boundary",
        tier="debug",
        prompt="The suite fails on an edge case. Fix the code so every test passes.",
        files={
            "chunk.py": (
                "def chunks(items, size):\n"
                "    return [items[i:i + size] for i in range(0, len(items), size)]\n"
            ),
            "tests/test_chunk.py": (
                "import pytest\nfrom chunk import chunks\n\n"
                "def test_even():\n    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n\n"
                "def test_size_zero_raises():\n"
                "    with pytest.raises(ValueError):\n        chunks([1, 2], 0)\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "import pytest\nfrom chunk import chunks\n\n"
                "def test_uneven():\n    assert chunks([1, 2, 3], 2) == [[1, 2], [3]]\n\n"
                "def test_empty():\n    assert chunks([], 3) == []\n\n"
                "def test_zero():\n"
                "    with pytest.raises(ValueError):\n        chunks([1], 0)\n\n"
                "def test_negative():\n"
                "    with pytest.raises(ValueError):\n        chunks([1], -1)\n"
            )
        },
    ),
    Task(
        id="debug-shared-default",
        tier="debug",
        prompt="Two tests fail for the same underlying reason. Fix the cause.",
        files={
            "registry.py": (
                "def register(name, into={}):\n"
                "    into[name] = True\n    return into\n"
            ),
            "tests/test_registry.py": (
                "from registry import register\n\n"
                "def test_first():\n    assert register('a') == {'a': True}\n\n"
                "def test_second():\n    assert register('b') == {'b': True}\n"
            ),
        },
        held_out={
            HELD_OUT: (
                "from registry import register\n\n"
                "def test_independent_calls():\n"
                "    assert register('x') == {'x': True}\n"
                "    assert register('y') == {'y': True}\n\n"
                "def test_explicit_target_still_works():\n"
                "    target = {'seed': True}\n"
                "    assert register('z', target) is target\n"
                "    assert target == {'seed': True, 'z': True}\n"
            )
        },
    ),
]


def by_tier(tier: str | None = None) -> list[Task]:
    return [t for t in TASKS if tier is None or t.tier == tier]


def by_ids(ids: list[str]) -> list[Task]:
    known = {t.id: t for t in TASKS}
    missing = [i for i in ids if i not in known]
    if missing:
        raise KeyError(f"unknown task ids: {missing}. Known: {sorted(known)}")
    return [known[i] for i in ids]
