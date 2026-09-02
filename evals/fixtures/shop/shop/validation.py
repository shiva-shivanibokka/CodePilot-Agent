"""
Input rules, applied at the edge.

Every rule is a class with a `validate` method so they can be composed into a
`RuleSet` and reported together, rather than failing on the first problem and
making the caller fix one field per round trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from shop.errors import ValidationError
from shop.models import Address, Customer, LineItem, Order
from shop.money import Money

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)
SKU = re.compile(r"^[A-Z]{2,4}-\d{3,6}$")
POSTCODE = {
    "US": re.compile(r"^\d{5}(\d{4})?$"),
    "GB": re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\d[A-Z]{2}$"),
    "DE": re.compile(r"^\d{5}$"),
}


class Rule:
    """A single check. Subclasses raise `ValidationError` from `validate`."""

    name = "rule"

    def validate(self, value) -> None:  # noqa: ANN001 - deliberately duck-typed
        raise NotImplementedError


class EmailRule(Rule):
    name = "email"

    def validate(self, value: str) -> None:
        if not value or not EMAIL.match(value):
            raise ValidationError("email", f"{value!r} is not a valid address")


class SkuRule(Rule):
    name = "sku"

    def validate(self, value: str) -> None:
        if not value or not SKU.match(value):
            raise ValidationError("sku", f"{value!r} does not match AA-000")


class PostcodeRule(Rule):
    name = "postcode"

    def __init__(self, country: str) -> None:
        self.country = country.upper()

    def validate(self, value: str) -> None:
        pattern = POSTCODE.get(self.country)
        if pattern is None:
            return
        if not pattern.match(value.upper().replace(" ", "")):
            raise ValidationError("postcode", f"{value!r} is not valid for {self.country}")


class QuantityRule(Rule):
    name = "quantity"

    def __init__(self, maximum: int = 999) -> None:
        self.maximum = maximum

    def validate(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError("quantity", "must be a whole number")
        if value <= 0:
            raise ValidationError("quantity", "must be positive")
        if value > self.maximum:
            raise ValidationError("quantity", f"cannot exceed {self.maximum}")


class AmountRule(Rule):
    name = "amount"

    def __init__(self, minimum: Money | None = None) -> None:
        self.minimum = minimum

    def validate(self, value: Money) -> None:
        if value.is_negative():
            raise ValidationError("amount", "cannot be negative")
        if self.minimum is not None and value < self.minimum:
            raise ValidationError("amount", f"must be at least {self.minimum.format()}")


class CouponCodeRule(Rule):
    name = "coupon"

    def validate(self, value: str) -> None:
        if not value or not value.isupper() or " " in value:
            raise ValidationError("coupon", "must be upper-case with no spaces")
        if len(value) < 4 or len(value) > 20:
            raise ValidationError("coupon", "must be 4 to 20 characters")


@dataclass
class RuleSet:
    """Runs several rules and reports every failure, not just the first."""

    rules: list[tuple[Rule, object]] = field(default_factory=list)

    def check(self, rule: Rule, value) -> RuleSet:  # noqa: ANN001
        self.rules.append((rule, value))
        return self

    def validate(self) -> None:
        problems: list[str] = []
        for rule, value in self.rules:
            try:
                rule.validate(value)
            except ValidationError as exc:
                problems.append(str(exc))
        if problems:
            raise ValidationError("request", "; ".join(problems))


class AddressValidator:
    """Composite rules for a postal address."""

    def validate(self, address: Address) -> None:
        clean = address.normalise()
        rules = RuleSet()
        if not clean.line1:
            raise ValidationError("line1", "is required")
        if not clean.city:
            raise ValidationError("city", "is required")
        rules.check(PostcodeRule(clean.country), clean.postcode)
        rules.validate()


class CustomerValidator:
    def __init__(self) -> None:
        self.addresses = AddressValidator()

    def validate(self, customer: Customer) -> None:
        RuleSet().check(EmailRule(), customer.email).validate()
        if customer.address is not None:
            self.addresses.validate(customer.address)


class LineValidator:
    def __init__(self, max_quantity: int = 999) -> None:
        self.quantities = QuantityRule(max_quantity)

    def validate(self, line: LineItem) -> None:
        (
            RuleSet()
            .check(SkuRule(), line.product.sku)
            .check(self.quantities, line.quantity)
            .check(AmountRule(), line.product.unit_price)
            .validate()
        )


class OrderValidator:
    """The one callers actually use."""

    def __init__(self, max_quantity: int = 999) -> None:
        self.customers = CustomerValidator()
        self.lines = LineValidator(max_quantity)

    def validate(self, order: Order) -> None:
        if not order.lines:
            raise ValidationError("lines", "an order needs at least one line")
        self.customers.validate(order.customer)
        for line in order.lines:
            self.lines.validate(line)
            if line.product.unit_price.currency != order.currency:
                raise ValidationError(
                    "currency",
                    f"{line.product.sku} is priced in "
                    f"{line.product.unit_price.currency}, order is {order.currency}",
                )
        if order.coupon is not None:
            RuleSet().check(CouponCodeRule(), order.coupon).validate()
