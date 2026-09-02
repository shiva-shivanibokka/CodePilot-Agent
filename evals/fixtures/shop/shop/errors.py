"""Exception types shared across the package."""

from __future__ import annotations


class ShopError(Exception):
    """Base class, so callers can catch everything this package raises."""


class ValidationError(ShopError):
    """A value failed a rule before it reached the domain layer."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class CurrencyMismatch(ShopError):
    """Two amounts in different currencies met in an arithmetic operation."""


class OutOfStock(ShopError):
    def __init__(self, sku: str, wanted: int, available: int) -> None:
        super().__init__(f"{sku}: wanted {wanted}, {available} available")
        self.sku = sku
        self.wanted = wanted
        self.available = available


class NotFound(ShopError):
    """A lookup by identifier found nothing."""


class RuleConflict(ShopError):
    """Two pricing rules cannot both apply to the same line."""
