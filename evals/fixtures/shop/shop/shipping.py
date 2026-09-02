"""
Shipping quotes.

Same shape as pricing: a list of rules, each with `applies_to` and `apply`,
run in order until one of them answers. Unlike discounts, shipping rules do
not stack — the first match wins, so the order of the list is the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from shop.errors import NotFound, ValidationError
from shop.models import Order
from shop.money import Money

#: Zones by country. Anything unlisted falls into "rest".
ZONES = {
    "US": "domestic",
    "CA": "near",
    "MX": "near",
    "GB": "europe",
    "DE": "europe",
    "FR": "europe",
    "JP": "far",
    "AU": "far",
}

#: Base cost in minor units per zone, then per-kilo on top.
ZONE_RATES = {
    "domestic": (500, 120),
    "near": (1200, 260),
    "europe": (1800, 340),
    "far": (2600, 520),
    "rest": (3200, 640),
}


def zone_for(country: str) -> str:
    return ZONES.get(country.upper(), "rest")


class ShippingRule:
    """One way of answering "what does delivery cost"."""

    name = "rule"
    order = 100

    def applies_to(self, order: Order) -> bool:
        return True

    def apply(self, order: Order) -> Money:
        raise NotImplementedError


class FreeOverThreshold(ShippingRule):
    """Free delivery once the basket is big enough."""

    name = "free-over"
    order = 10

    def __init__(self, threshold: Money) -> None:
        self.threshold = threshold

    def applies_to(self, order: Order) -> bool:
        return self.threshold <= order.subtotal()

    def apply(self, order: Order) -> Money:
        return Money.zero(order.currency)


class PartnerFreeShipping(ShippingRule):
    name = "partner"
    order = 20

    def applies_to(self, order: Order) -> bool:
        return order.customer.tier == "partner"

    def apply(self, order: Order) -> Money:
        return Money.zero(order.currency)


class WeightBanded(ShippingRule):
    """Base rate for the zone plus a per-kilo charge, rounded up to the kilo."""

    name = "weight"
    order = 50

    def applies_to(self, order: Order) -> bool:
        return order.customer.address is not None

    def apply(self, order: Order) -> Money:
        address = order.customer.address
        if address is None:
            raise ValidationError("address", "a shipping quote needs an address")
        zone = zone_for(address.normalise().country)
        base, per_kilo = ZONE_RATES[zone]
        kilos = -(-order.weight_grams() // 1000)
        return Money(base + per_kilo * kilos, order.currency)


class FlatRate(ShippingRule):
    """The fallback, so a quote is always possible."""

    name = "flat"
    order = 900

    def __init__(self, amount: Money) -> None:
        self.amount = amount

    def apply(self, order: Order) -> Money:
        return Money(self.amount.minor, order.currency)


@dataclass
class ShippingQuoter:
    rules: list[ShippingRule]

    @classmethod
    def standard(cls, currency: str = "USD") -> ShippingQuoter:
        return cls(
            rules=[
                FreeOverThreshold(Money.from_major(75, currency)),
                PartnerFreeShipping(),
                WeightBanded(),
                FlatRate(Money.from_major(9.99, currency)),
            ]
        )

    def quote(self, order: Order) -> Money:
        for rule in sorted(self.rules, key=lambda r: r.order):
            if rule.applies_to(order):
                return rule.apply(order)
        raise NotFound("no shipping rule matched, and there is no fallback")

    def explain(self, order: Order) -> tuple[str, Money]:
        for rule in sorted(self.rules, key=lambda r: r.order):
            if rule.applies_to(order):
                return rule.name, rule.apply(order)
        raise NotFound("no shipping rule matched, and there is no fallback")
