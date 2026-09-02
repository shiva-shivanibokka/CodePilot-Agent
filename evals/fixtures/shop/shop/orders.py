"""
The service layer.

Everything a caller can do to an order goes through here, so the ordering of
validate -> price -> reserve -> persist happens once rather than in every
handler that ever touches a basket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from shop.errors import NotFound, OutOfStock, ValidationError
from shop.inventory import Inventory
from shop.models import Customer, Order, OrderStatus, Product
from shop.money import Money
from shop.pricing import CouponDiscount, PriceEngine, Quote, default_rules
from shop.shipping import ShippingQuoter
from shop.storage import Store
from shop.validation import OrderValidator

#: Which states an order may move to from each state it might be in.
TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.DRAFT: {OrderStatus.PLACED, OrderStatus.CANCELLED},
    OrderStatus.PLACED: {OrderStatus.PAID, OrderStatus.CANCELLED},
    OrderStatus.PAID: {OrderStatus.SHIPPED, OrderStatus.REFUNDED},
    OrderStatus.SHIPPED: {OrderStatus.REFUNDED},
    OrderStatus.CANCELLED: set(),
    OrderStatus.REFUNDED: set(),
}


@dataclass
class PlacedOrder:
    order: Order
    quote: Quote

    def total(self) -> Money:
        return self.quote.total()


class OrderService:
    def __init__(
        self,
        store: Store | None = None,
        inventory: Inventory | None = None,
        validator: OrderValidator | None = None,
        shipping: ShippingQuoter | None = None,
    ) -> None:
        self.store = store or Store()
        self.inventory = inventory or Inventory()
        self.validator = validator or OrderValidator()
        self.shipping = shipping or ShippingQuoter.standard()

    # -- building ---------------------------------------------------------
    def start(self, order_id: str, customer: Customer, currency: str = "USD") -> Order:
        if self.store.orders.find(order_id) is not None:
            raise ValidationError("order_id", f"{order_id} already exists")
        order = Order(id=order_id, customer=customer, currency=currency)
        return self.store.orders.add(order)

    def add_line(self, order_id: str, sku: str, quantity: int = 1) -> Order:
        order = self.load(order_id)
        if not order.is_editable():
            raise ValidationError("status", f"a {order.status} order cannot be changed")
        product = self.store.products.by_sku(sku)
        order.add(product, quantity)
        return order

    def apply_coupon(self, order_id: str, code: str) -> Order:
        order = self.load(order_id)
        if self.store.coupons.by_code(code) is None:
            raise NotFound(f"no coupon {code}")
        order.coupon = code.upper()
        return order

    def load(self, order_id: str) -> Order:
        return self.store.orders.get(order_id)  # type: ignore[return-value]

    # -- pricing ----------------------------------------------------------
    def engine_for(self, order: Order) -> PriceEngine:
        rules = default_rules()
        if order.coupon:
            coupon = self.store.coupons.by_code(order.coupon)
            if coupon is not None:
                rules.append(CouponDiscount(coupon))
        return PriceEngine(rules=rules)

    def quote(self, order_id: str) -> Quote:
        order = self.load(order_id)
        self.validator.validate(order)
        shipping = self.shipping.quote(order)
        return self.engine_for(order).quote(order, shipping=shipping)

    # -- transitions ------------------------------------------------------
    def can_move(self, order: Order, to: OrderStatus) -> bool:
        return to in TRANSITIONS[order.status]

    def move(self, order: Order, to: OrderStatus) -> Order:
        if not self.can_move(order, to):
            raise ValidationError("status", f"cannot go from {order.status} to {to}")
        order.status = to
        return order

    def place(self, order_id: str, today: date | None = None) -> PlacedOrder:
        order = self.load(order_id)
        self.validator.validate(order)
        order.placed_on = today or date.today()
        try:
            self.inventory.reserve(order)
        except OutOfStock:
            raise
        quote = self.engine_for(order).quote(order, shipping=self.shipping.quote(order))
        order.shipping = quote.shipping
        self.move(order, OrderStatus.PLACED)
        return PlacedOrder(order=order, quote=quote)

    def pay(self, order_id: str) -> Order:
        return self.move(self.load(order_id), OrderStatus.PAID)

    def ship(self, order_id: str) -> Order:
        order = self.move(self.load(order_id), OrderStatus.SHIPPED)
        self.inventory.fulfil(order_id)
        return order

    def cancel(self, order_id: str) -> Order:
        order = self.move(self.load(order_id), OrderStatus.CANCELLED)
        self.inventory.release(order_id)
        return order

    # -- catalogue --------------------------------------------------------
    def stock(self, product: Product, count: int) -> Product:
        self.store.products.add(product)
        self.inventory.receive(product.sku, count)
        return product
