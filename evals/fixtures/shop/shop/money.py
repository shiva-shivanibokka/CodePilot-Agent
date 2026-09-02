"""
Money as integer minor units.

Floats are not used anywhere in this package for amounts. A cent is an int;
a currency is a three-letter code; the two travel together and refuse to mix.
"""

from __future__ import annotations

from dataclasses import dataclass

from shop.errors import CurrencyMismatch, ValidationError

#: Minor units per major unit. Not every currency uses two.
EXPONENT = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "KWD": 3,
}


def normalise(code: str) -> str:
    """Upper-case and check a currency code."""
    if not isinstance(code, str) or len(code) != 3:
        raise ValidationError("currency", "must be a three-letter code")
    upper = code.upper()
    if upper not in EXPONENT:
        raise ValidationError("currency", f"unsupported currency {upper}")
    return upper


@dataclass(frozen=True)
class Money:
    """An amount in minor units of a single currency."""

    minor: int
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not isinstance(self.minor, int) or isinstance(self.minor, bool):
            raise ValidationError("minor", "must be an integer number of minor units")
        object.__setattr__(self, "currency", normalise(self.currency))

    # -- construction -----------------------------------------------------
    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        return cls(0, currency)

    @classmethod
    def from_major(cls, amount: float | int | str, currency: str = "USD") -> Money:
        """Build from a major-unit amount, rounding half away from zero."""
        code = normalise(currency)
        scale = 10 ** EXPONENT[code]
        value = float(amount) * scale
        return cls(round_half_up(value), code)

    # -- arithmetic -------------------------------------------------------
    def _check(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(f"{self.currency} and {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor - other.minor, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.minor, self.currency)

    def __mul__(self, factor: int) -> Money:
        if not isinstance(factor, int) or isinstance(factor, bool):
            raise ValidationError("factor", "money multiplies by whole units only")
        return Money(self.minor * factor, self.currency)

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor < other.minor

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self.minor <= other.minor

    def is_negative(self) -> bool:
        return self.minor < 0

    def scale(self, numerator: int, denominator: int) -> Money:
        """Multiply by a rational fraction, rounding half away from zero.

        Percentages go through here rather than through float multiplication,
        so 5% of 1005 cents is 50 rather than 50.249999999999996.
        """
        if denominator == 0:
            raise ValidationError("denominator", "cannot be zero")
        return Money(round_half_up(self.minor * numerator / denominator), self.currency)

    # -- presentation -----------------------------------------------------
    def major(self) -> float:
        return self.minor / (10 ** EXPONENT[self.currency])

    def format(self) -> str:
        digits = EXPONENT[self.currency]
        if digits == 0:
            return f"{self.minor} {self.currency}"
        sign = "-" if self.minor < 0 else ""
        whole, part = divmod(abs(self.minor), 10 ** digits)
        return f"{sign}{whole}.{part:0{digits}d} {self.currency}"

    def __str__(self) -> str:
        return self.format()


def round_half_up(value: float) -> int:
    """Round half away from zero.

    Python's built-in `round` rounds half to even, which is correct for
    statistics and wrong for invoices: a customer charged 2.5 cents expects
    3, not 2, and expects the same answer every time.
    """
    if value >= 0:
        return int(value + 0.5)
    return -int(-value + 0.5)


def total(amounts: list[Money], currency: str = "USD") -> Money:
    """Sum a list of amounts, defaulting to zero in `currency` when empty."""
    if not amounts:
        return Money.zero(currency)
    result = amounts[0]
    for amount in amounts[1:]:
        result = result + amount
    return result


def allocate(amount: Money, weights: list[int]) -> list[Money]:
    """Split `amount` across `weights` so the parts sum back to the whole.

    Naive proportional division loses or invents a cent. The remainder is
    handed out one minor unit at a time, largest weight first.
    """
    if not weights:
        raise ValidationError("weights", "cannot be empty")
    if any(w < 0 for w in weights):
        raise ValidationError("weights", "cannot be negative")
    denominator = sum(weights)
    if denominator == 0:
        raise ValidationError("weights", "must not all be zero")

    parts = [amount.minor * w // denominator for w in weights]
    remainder = amount.minor - sum(parts)
    order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    step = 1 if remainder >= 0 else -1
    for i in range(abs(remainder)):
        parts[order[i % len(order)]] += step
    return [Money(p, amount.currency) for p in parts]
