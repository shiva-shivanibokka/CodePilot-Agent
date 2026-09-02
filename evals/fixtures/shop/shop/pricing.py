"""
The pricing engine.

An order goes in with nothing but quantities and list prices; it comes out with
a discount and a tax figure on every line. The rules are ordered, and the order
matters: percentage discounts stack multiplicatively against the running price,
fixed discounts come off the end, and tax is computed last against whatever is
left.

Everything here works in integer minor units. See `shop.money` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from shop.errors import RuleConflict, ValidationError
from shop.models import Coupon, CustomerTier, LineItem, Order
from shop.money import Money, allocate

# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

#: Basis points, because a third of a percent is a real rate and 0.333 is not
#: a number this package is willing to hold.
BASE_TAX_BPS = {
    "US": 0,
    "GB": 2000,
    "DE": 1900,
    "FR": 2000,
    "JP": 1000,
}

#: Categories taxed differently from the base rate in a given country.
CATEGORY_TAX_BPS = {
    ("GB", "book"): 0,
    ("GB", "food"): 0,
    ("DE", "book"): 700,
    ("DE", "food"): 700,
    ("FR", "book"): 550,
}


@dataclass(frozen=True)
class TaxTable:
    """Looks up a rate in basis points for a country and category."""

    base: dict[str, int] = field(default_factory=lambda: dict(BASE_TAX_BPS))
    by_category: dict[tuple[str, str], int] = field(
        default_factory=lambda: dict(CATEGORY_TAX_BPS)
    )

    def rate_bps(self, country: str, category: str) -> int:
        key = (country.upper(), category.lower())
        if key in self.by_category:
            return self.by_category[key]
        return self.base.get(country.upper(), 0)

    def apply(self, amount: Money, country: str, category: str) -> Money:
        """Tax due on `amount`. Not the gross — just the tax."""
        bps = self.rate_bps(country, category)
        if bps == 0:
            return Money.zero(amount.currency)
        return amount.scale(bps, 10_000)


# ---------------------------------------------------------------------------
# Discount rules
# ---------------------------------------------------------------------------


class DiscountRule:
    """One reason a line might cost less than its list price.

    `apply` returns the amount to take *off* the line, never the new price.
    Returning the reduction rather than the result is what lets the engine sum
    several rules, cap the total, and report each one separately on the
    invoice.
    """

    #: Rules with a lower number run first.
    order = 100
    name = "rule"

    def applies_to(self, order: Order, line: LineItem) -> bool:
        return True

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        raise NotImplementedError


#: Quantity thresholds and their reduction in basis points, richest first.
#: A line takes the best single tier it qualifies for; tiers do not stack.
VOLUME_TIERS: list[tuple[int, int]] = [
    (100, 1500),
    (50, 1000),
    (10, 500),
]


class VolumeDiscount(DiscountRule):
    """Buy more, pay less per unit."""

    order = 10
    name = "volume"

    def __init__(self, tiers: list[tuple[int, int]] | None = None) -> None:
        self.tiers = sorted(tiers or VOLUME_TIERS, reverse=True)

    def tier_for(self, quantity: int) -> tuple[int, int] | None:
        for threshold, bps in self.tiers:
            if quantity >= threshold:
                return (threshold, bps)
        return None

    def applies_to(self, order: Order, line: LineItem) -> bool:
        return line.product.discountable and self.tier_for(line.quantity) is not None

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        tier = self.tier_for(line.quantity)
        if tier is None:
            return Money.zero(running.currency)
        return running.scale(tier[1], 10_000)


#: What each customer tier takes off, in basis points.
TIER_BPS = {
    CustomerTier.STANDARD: 0,
    CustomerTier.SILVER: 300,
    CustomerTier.GOLD: 700,
    CustomerTier.PARTNER: 1500,
}


class TierDiscount(DiscountRule):
    """A standing reduction attached to the customer, not the basket."""

    order = 20
    name = "tier"

    def __init__(self, table: dict[CustomerTier, int] | None = None) -> None:
        self.table = table or dict(TIER_BPS)

    def bps_for(self, order: Order) -> int:
        return self.table.get(order.customer.tier, 0)

    def applies_to(self, order: Order, line: LineItem) -> bool:
        return line.product.discountable and self.bps_for(order) > 0

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        return running.scale(self.bps_for(order), 10_000)


class CategoryDiscount(DiscountRule):
    """A promotion on a whole category for a window of dates."""

    order = 30
    name = "category"

    def __init__(
        self,
        category: str,
        bps: int,
        starts: date | None = None,
        ends: date | None = None,
    ) -> None:
        if bps < 0 or bps > 10_000:
            raise ValidationError("bps", "must be between 0 and 10000")
        self.category = category.lower()
        self.bps = bps
        self.starts = starts
        self.ends = ends

    def in_window(self, today: date) -> bool:
        if self.starts is not None and today < self.starts:
            return False
        if self.ends is not None and today > self.ends:
            return False
        return True

    def applies_to(self, order: Order, line: LineItem) -> bool:
        if not line.product.discountable:
            return False
        if line.product.category.lower() != self.category:
            return False
        return self.in_window(order.placed_on or date.today())

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        return running.scale(self.bps, 10_000)


class CouponDiscount(DiscountRule):
    """A code the customer typed. Percentage or fixed, never both."""

    order = 40
    name = "coupon"

    def __init__(self, coupon: Coupon) -> None:
        if coupon.percent_off and coupon.amount_off is not None:
            raise RuleConflict(f"{coupon.code} is both a percentage and a fixed amount")
        self.coupon = coupon

    def applies_to(self, order: Order, line: LineItem) -> bool:
        if order.coupon != self.coupon.code:
            return False
        if not self.coupon.is_live(order.placed_on or date.today()):
            return False
        if self.coupon.minimum is not None and order.subtotal() < self.coupon.minimum:
            return False
        return line.product.discountable or self.coupon.amount_off is not None

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        if self.coupon.percent_off:
            return running.scale(self.coupon.percent_off, 100)
        # A fixed amount belongs to the order, not to any one line, so it is
        # split across the lines in proportion to what they cost. Handing the
        # whole thing to the first line would make the invoice wrong even
        # though the total came out right.
        return Money.zero(running.currency)


class BundleDiscount(DiscountRule):
    """Buy every SKU in a set, take a reduction on all of them."""

    order = 50
    name = "bundle"

    def __init__(self, skus: list[str], bps: int) -> None:
        if len(skus) < 2:
            raise ValidationError("skus", "a bundle needs at least two products")
        self.skus = {s.upper() for s in skus}
        self.bps = bps

    def complete(self, order: Order) -> bool:
        present = {line.product.sku.upper() for line in order.lines}
        return self.skus.issubset(present)

    def applies_to(self, order: Order, line: LineItem) -> bool:
        return (
            line.product.discountable
            and line.product.sku.upper() in self.skus
            and self.complete(order)
        )

    def apply(self, order: Order, line: LineItem, running: Money) -> Money:
        return running.scale(self.bps, 10_000)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

#: No stack of rules may take more than this off a line. Without a cap, a
#: partner buying two hundred discounted books during a promotion with a
#: coupon reaches a negative price, and a negative line is a refund nobody
#: authorised.
MAX_DISCOUNT_BPS = 6000


@dataclass
class PricedLine:
    """What one line cost and why."""

    line: LineItem
    subtotal: Money
    discount: Money
    tax: Money
    reasons: list[tuple[str, Money]] = field(default_factory=list)

    def net(self) -> Money:
        return self.subtotal - self.discount

    def gross(self) -> Money:
        return self.net() + self.tax


@dataclass
class Quote:
    """The whole priced order."""

    order: Order
    lines: list[PricedLine]
    shipping: Money

    def subtotal(self) -> Money:
        result = Money.zero(self.order.currency)
        for priced in self.lines:
            result = result + priced.subtotal
        return result

    def discount(self) -> Money:
        result = Money.zero(self.order.currency)
        for priced in self.lines:
            result = result + priced.discount
        return result

    def tax(self) -> Money:
        result = Money.zero(self.order.currency)
        for priced in self.lines:
            result = result + priced.tax
        return result

    def total(self) -> Money:
        return self.subtotal() - self.discount() + self.tax() + self.shipping

    def reasons(self) -> dict[str, Money]:
        """Every rule that fired, and what it took off in total."""
        merged: dict[str, Money] = {}
        for priced in self.lines:
            for name, amount in priced.reasons:
                if name in merged:
                    merged[name] = merged[name] + amount
                else:
                    merged[name] = amount
        return merged


class PriceEngine:
    """Applies the rules in order and produces a `Quote`."""

    def __init__(
        self,
        rules: list[DiscountRule] | None = None,
        taxes: TaxTable | None = None,
        max_discount_bps: int = MAX_DISCOUNT_BPS,
    ) -> None:
        self.rules = sorted(rules or default_rules(), key=lambda r: r.order)
        self.taxes = taxes or TaxTable()
        self.max_discount_bps = max_discount_bps

    # -- discounts --------------------------------------------------------
    def discounts_for(self, order: Order, line: LineItem) -> tuple[Money, list[tuple[str, Money]]]:
        """Every reduction that applies to one line, capped.

        Percentages apply to the running price rather than to the list price,
        so two 10% rules take 19% and not 20%. That is what a shop means by
        "stacking" and it is why the running total is threaded through
        `apply`.
        """
        running = line.subtotal()
        taken = Money.zero(order.currency)
        reasons: list[tuple[str, Money]] = []

        for rule in self.rules:
            if not rule.applies_to(order, line):
                continue
            amount = rule.apply(order, line, running)
            if amount.minor <= 0:
                continue
            taken = taken + amount
            running = running - amount
            reasons.append((rule.name, amount))

        cap = line.subtotal().scale(self.max_discount_bps, 10_000)
        if cap < taken:
            reasons = self._rescale(reasons, taken, cap)
            taken = cap
        return taken, reasons

    def _rescale(
        self, reasons: list[tuple[str, Money]], taken: Money, cap: Money
    ) -> list[tuple[str, Money]]:
        """Shrink each reason proportionally so the parts still sum to the cap."""
        if not reasons or taken.minor == 0:
            return reasons
        shares = allocate(cap, [amount.minor for _, amount in reasons])
        return [(name, share) for (name, _), share in zip(reasons, shares, strict=True)]

    # -- fixed-amount coupons --------------------------------------------
    def _fixed_coupon(self, order: Order) -> Money | None:
        for rule in self.rules:
            if isinstance(rule, CouponDiscount) and rule.coupon.amount_off is not None:
                if any(rule.applies_to(order, line) for line in order.lines):
                    return rule.coupon.amount_off
        return None

    def _spread_fixed(self, order: Order, priced: list[PricedLine], amount: Money) -> None:
        """Split a fixed coupon across the lines in proportion to their net."""
        weights = [max(p.net().minor, 0) for p in priced]
        if sum(weights) == 0:
            return
        capped = amount if amount.minor <= sum(weights) else Money(sum(weights), amount.currency)
        for p, share in zip(priced, allocate(capped, weights), strict=True):
            if share.minor <= 0:
                continue
            p.discount = p.discount + share
            p.reasons.append(("coupon", share))

    # -- tax --------------------------------------------------------------
    def tax_for(self, order: Order, priced: PricedLine) -> Money:
        if order.customer.tax_exempt:
            return Money.zero(order.currency)
        address = order.customer.address
        if address is None or not priced.line.product.taxable:
            return Money.zero(order.currency)
        country = address.normalise().country
        return self.taxes.apply(priced.net(), country, priced.line.product.category)

    # -- the whole order --------------------------------------------------
    def quote(self, order: Order, shipping: Money | None = None) -> Quote:
        priced: list[PricedLine] = []
        for line in order.lines:
            discount, reasons = self.discounts_for(order, line)
            priced.append(
                PricedLine(
                    line=line,
                    subtotal=line.subtotal(),
                    discount=discount,
                    tax=Money.zero(order.currency),
                    reasons=reasons,
                )
            )

        fixed = self._fixed_coupon(order)
        if fixed is not None:
            self._spread_fixed(order, priced, fixed)

        for p in priced:
            p.tax = self.tax_for(order, p)
            p.line.discount = p.discount
            p.line.tax = p.tax

        ship = shipping or order.shipping or Money.zero(order.currency)
        return Quote(order=order, lines=priced, shipping=ship)


def default_rules() -> list[DiscountRule]:
    """The rules every shop starts with. Coupons are added per order."""
    return [VolumeDiscount(), TierDiscount()]


def price(order: Order, extra: list[DiscountRule] | None = None) -> Quote:
    """Convenience wrapper for the common case."""
    engine = PriceEngine(rules=default_rules() + list(extra or []))
    return engine.quote(order)
