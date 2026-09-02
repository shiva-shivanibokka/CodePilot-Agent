"""
Summaries over a set of orders.

Read-only. Nothing here changes an order; if a number looks wrong the cause is
upstream, in pricing or in the service layer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from shop.models import Order, OrderStatus
from shop.money import Money, allocate, total

#: Statuses that count as revenue. A placed order is not money yet.
EARNED = (OrderStatus.PAID, OrderStatus.SHIPPED)


@dataclass
class Row:
    key: str
    orders: int
    units: int
    revenue: Money

    def average(self) -> Money:
        if self.orders == 0:
            return Money.zero(self.revenue.currency)
        return Money(self.revenue.minor // self.orders, self.revenue.currency)

    def format(self) -> str:
        return f"{self.key:<20} {self.orders:>5} {self.units:>7} {self.revenue.format():>14}"


def earned(orders: list[Order]) -> list[Order]:
    return [o for o in orders if o.status in EARNED]


def revenue(orders: list[Order], currency: str = "USD") -> Money:
    return total([o.total() for o in earned(orders)], currency)


def by_status(orders: list[Order]) -> dict[OrderStatus, int]:
    counts: dict[OrderStatus, int] = defaultdict(int)
    for order in orders:
        counts[order.status] += 1
    return dict(counts)


def by_customer(orders: list[Order], currency: str = "USD") -> list[Row]:
    grouped: dict[str, list[Order]] = defaultdict(list)
    for order in earned(orders):
        grouped[order.customer.id].append(order)
    rows = [
        Row(
            key=customer_id,
            orders=len(group),
            units=sum(o.quantity() for o in group),
            revenue=total([o.total() for o in group], currency),
        )
        for customer_id, group in grouped.items()
    ]
    return sorted(rows, key=lambda r: r.revenue.minor, reverse=True)


def by_category(orders: list[Order], currency: str = "USD") -> list[Row]:
    grouped: dict[str, list[tuple[Order, int, Money]]] = defaultdict(list)
    for order in earned(orders):
        for line in order.lines:
            grouped[line.product.category].append((order, line.quantity, line.total()))
    rows = [
        Row(
            key=category,
            orders=len({id(o) for o, _, _ in entries}),
            units=sum(q for _, q, _ in entries),
            revenue=total([m for _, _, m in entries], currency),
        )
        for category, entries in grouped.items()
    ]
    return sorted(rows, key=lambda r: r.revenue.minor, reverse=True)


def by_month(orders: list[Order], currency: str = "USD") -> list[Row]:
    grouped: dict[str, list[Order]] = defaultdict(list)
    for order in earned(orders):
        when: date = order.placed_on or date.min
        grouped[f"{when.year:04d}-{when.month:02d}"].append(order)
    return [
        Row(
            key=month,
            orders=len(group),
            units=sum(o.quantity() for o in group),
            revenue=total([o.total() for o in group], currency),
        )
        for month, group in sorted(grouped.items())
    ]


def split_commission(orders: list[Order], shares: dict[str, int], currency: str = "USD") -> dict[str, Money]:
    """Divide revenue between named partners without losing a minor unit."""
    if not shares:
        return {}
    names = list(shares)
    parts = allocate(revenue(orders, currency), [shares[n] for n in names])
    return dict(zip(names, parts, strict=True))


def render(rows: list[Row]) -> str:
    header = f"{'key':<20} {'orders':>5} {'units':>7} {'revenue':>14}"
    return "\n".join([header, "-" * len(header)] + [r.format() for r in rows])
