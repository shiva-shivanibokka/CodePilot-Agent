"""
Persistence, such as it is.

An in-memory repository with the same method names a database-backed one
would have, so the service layer never learns which it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shop.errors import NotFound, ValidationError
from shop.models import Coupon, Customer, Order, OrderStatus, Product


@dataclass
class Repository:
    """A keyed collection with the four operations everything needs."""

    items: dict[str, object] = field(default_factory=dict)
    label: str = "item"

    def put(self, key: str, value: object) -> object:
        if not key:
            raise ValidationError("key", "cannot be empty")
        self.items[key] = value
        return value

    def get(self, key: str) -> object:
        if key not in self.items:
            raise NotFound(f"no {self.label} with id {key}")
        return self.items[key]

    def find(self, key: str) -> object | None:
        return self.items.get(key)

    def all(self) -> list[object]:
        return list(self.items.values())

    def delete(self, key: str) -> None:
        self.items.pop(key, None)

    def __len__(self) -> int:
        return len(self.items)


class ProductRepository(Repository):
    def __init__(self) -> None:
        super().__init__(label="product")

    def add(self, product: Product) -> Product:
        self.put(product.sku.upper(), product)
        return product

    def by_sku(self, sku: str) -> Product:
        return self.get(sku.upper())  # type: ignore[return-value]

    def by_category(self, category: str) -> list[Product]:
        wanted = category.lower()
        return [p for p in self.all() if p.category.lower() == wanted]  # type: ignore[attr-defined]


class CustomerRepository(Repository):
    def __init__(self) -> None:
        super().__init__(label="customer")

    def add(self, customer: Customer) -> Customer:
        self.put(customer.id, customer)
        return customer

    def by_email(self, email: str) -> Customer | None:
        wanted = email.strip().lower()
        for customer in self.all():
            if customer.email.strip().lower() == wanted:  # type: ignore[attr-defined]
                return customer  # type: ignore[return-value]
        return None


class OrderRepository(Repository):
    def __init__(self) -> None:
        super().__init__(label="order")

    def add(self, order: Order) -> Order:
        self.put(order.id, order)
        return order

    def by_customer(self, customer_id: str) -> list[Order]:
        return [o for o in self.all() if o.customer.id == customer_id]  # type: ignore[attr-defined]

    def by_status(self, status: OrderStatus) -> list[Order]:
        return [o for o in self.all() if o.status == status]  # type: ignore[attr-defined]


class CouponRepository(Repository):
    def __init__(self) -> None:
        super().__init__(label="coupon")

    def add(self, coupon: Coupon) -> Coupon:
        self.put(coupon.code.upper(), coupon)
        return coupon

    def by_code(self, code: str) -> Coupon | None:
        return self.find(code.upper())  # type: ignore[return-value]


@dataclass
class Store:
    """Everything a service needs, in one object."""

    products: ProductRepository = field(default_factory=ProductRepository)
    customers: CustomerRepository = field(default_factory=CustomerRepository)
    orders: OrderRepository = field(default_factory=OrderRepository)
    coupons: CouponRepository = field(default_factory=CouponRepository)
