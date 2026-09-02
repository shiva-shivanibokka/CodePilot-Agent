"""
Stock levels and reservations.

A reservation holds units for an order that has not been paid for yet. It is
the difference between "we have 5" and "we have 5 and four of them are already
spoken for", which is the difference between a sale and an apology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shop.errors import NotFound, OutOfStock, ValidationError
from shop.models import Order, Product


@dataclass
class StockLevel:
    sku: str
    on_hand: int = 0
    reserved: int = 0

    def available(self) -> int:
        return max(self.on_hand - self.reserved, 0)

    def validate(self) -> None:
        if self.on_hand < 0:
            raise ValidationError("on_hand", "cannot be negative")
        if self.reserved < 0:
            raise ValidationError("reserved", "cannot be negative")


@dataclass
class Reservation:
    order_id: str
    lines: dict[str, int] = field(default_factory=dict)

    def units(self) -> int:
        return sum(self.lines.values())


class Inventory:
    """In-memory stock. A real deployment swaps this for the database."""

    def __init__(self, levels: dict[str, int] | None = None) -> None:
        self.levels: dict[str, StockLevel] = {
            sku: StockLevel(sku=sku, on_hand=count) for sku, count in (levels or {}).items()
        }
        self.reservations: dict[str, Reservation] = {}

    # -- levels -----------------------------------------------------------
    def level(self, sku: str) -> StockLevel:
        key = sku.upper()
        if key not in self.levels:
            raise NotFound(f"no stock record for {sku}")
        return self.levels[key]

    def available(self, sku: str) -> int:
        try:
            return self.level(sku).available()
        except NotFound:
            return 0

    def receive(self, sku: str, count: int) -> StockLevel:
        if count <= 0:
            raise ValidationError("count", "must be positive")
        key = sku.upper()
        level = self.levels.setdefault(key, StockLevel(sku=key))
        level.on_hand += count
        level.validate()
        return level

    def stock(self, product: Product, count: int) -> StockLevel:
        return self.receive(product.sku, count)

    # -- reservations -----------------------------------------------------
    def check(self, order: Order) -> None:
        """Raise on the first line that cannot be filled."""
        wanted: dict[str, int] = {}
        for line in order.lines:
            wanted[line.product.sku.upper()] = (
                wanted.get(line.product.sku.upper(), 0) + line.quantity
            )
        for sku, count in wanted.items():
            have = self.available(sku)
            if have < count:
                raise OutOfStock(sku, count, have)

    def reserve(self, order: Order) -> Reservation:
        if order.id in self.reservations:
            raise ValidationError("order", f"{order.id} already has a reservation")
        self.check(order)
        held: dict[str, int] = {}
        for line in order.lines:
            sku = line.product.sku.upper()
            self.level(sku).reserved += line.quantity
            held[sku] = held.get(sku, 0) + line.quantity
        reservation = Reservation(order_id=order.id, lines=held)
        self.reservations[order.id] = reservation
        return reservation

    def release(self, order_id: str) -> None:
        reservation = self.reservations.pop(order_id, None)
        if reservation is None:
            return
        for sku, count in reservation.lines.items():
            level = self.level(sku)
            level.reserved = max(level.reserved - count, 0)

    def fulfil(self, order_id: str) -> None:
        """Turn a reservation into a shipment: stock leaves the building."""
        reservation = self.reservations.pop(order_id, None)
        if reservation is None:
            raise NotFound(f"no reservation for {order_id}")
        for sku, count in reservation.lines.items():
            level = self.level(sku)
            level.reserved = max(level.reserved - count, 0)
            level.on_hand -= count
            level.validate()

    def low_stock(self, threshold: int = 5) -> list[StockLevel]:
        return sorted(
            (lvl for lvl in self.levels.values() if lvl.available() <= threshold),
            key=lambda lvl: lvl.available(),
        )
