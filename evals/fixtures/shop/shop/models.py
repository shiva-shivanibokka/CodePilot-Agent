"""
The domain objects.

These are data, not behaviour. Anything that needs a rule engine, a database
or a clock lives in another module; this one only knows shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from shop.errors import ValidationError
from shop.money import Money, normalise


class OrderStatus(StrEnum):
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class CustomerTier(StrEnum):
    STANDARD = "standard"
    SILVER = "silver"
    GOLD = "gold"
    PARTNER = "partner"


@dataclass(frozen=True)
class Address:
    line1: str
    city: str
    postcode: str
    country: str

    def normalise(self) -> Address:
        """Trim and upper-case the parts that are matched against tables."""
        return Address(
            line1=self.line1.strip(),
            city=self.city.strip(),
            postcode=self.postcode.strip().upper().replace(" ", ""),
            country=self.country.strip().upper(),
        )

    def is_domestic(self, home: str = "US") -> bool:
        return self.normalise().country == home.upper()


@dataclass
class Customer:
    id: str
    email: str
    tier: CustomerTier = CustomerTier.STANDARD
    address: Address | None = None
    tax_exempt: bool = False
    since: date | None = None

    def label(self) -> str:
        return f"{self.id} <{self.email}> ({self.tier})"


@dataclass
class Product:
    sku: str
    name: str
    unit_price: Money
    category: str = "general"
    weight_grams: int = 0
    taxable: bool = True
    #: Products flagged here never take a percentage discount. Clearance
    #: stock is already below cost.
    discountable: bool = True

    def describe(self) -> str:
        return f"{self.sku} {self.name} @ {self.unit_price.format()}"


@dataclass
class LineItem:
    product: Product
    quantity: int
    #: Set by the pricing engine. Left alone by everything else.
    discount: Money | None = None
    tax: Money | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValidationError("quantity", "must be positive")

    def subtotal(self) -> Money:
        return self.product.unit_price * self.quantity

    def discounted(self) -> Money:
        if self.discount is None:
            return self.subtotal()
        return self.subtotal() - self.discount

    def total(self) -> Money:
        base = self.discounted()
        return base if self.tax is None else base + self.tax

    def weight_grams(self) -> int:
        return self.product.weight_grams * self.quantity


@dataclass
class Order:
    id: str
    customer: Customer
    lines: list[LineItem] = field(default_factory=list)
    currency: str = "USD"
    status: OrderStatus = OrderStatus.DRAFT
    placed_on: date | None = None
    shipping: Money | None = None
    coupon: str | None = None

    def __post_init__(self) -> None:
        self.currency = normalise(self.currency)

    def add(self, product: Product, quantity: int = 1) -> LineItem:
        line = LineItem(product=product, quantity=quantity)
        self.lines.append(line)
        return line

    def quantity(self) -> int:
        return sum(line.quantity for line in self.lines)

    def subtotal(self) -> Money:
        result = Money.zero(self.currency)
        for line in self.lines:
            result = result + line.subtotal()
        return result

    def discount_total(self) -> Money:
        result = Money.zero(self.currency)
        for line in self.lines:
            if line.discount is not None:
                result = result + line.discount
        return result

    def tax_total(self) -> Money:
        result = Money.zero(self.currency)
        for line in self.lines:
            if line.tax is not None:
                result = result + line.tax
        return result

    def total(self) -> Money:
        result = Money.zero(self.currency)
        for line in self.lines:
            result = result + line.total()
        if self.shipping is not None:
            result = result + self.shipping
        return result

    def weight_grams(self) -> int:
        return sum(line.weight_grams() for line in self.lines)

    def is_editable(self) -> bool:
        return self.status in (OrderStatus.DRAFT, OrderStatus.PLACED)


@dataclass(frozen=True)
class Coupon:
    code: str
    percent_off: int = 0
    amount_off: Money | None = None
    minimum: Money | None = None
    expires: date | None = None

    def is_live(self, today: date) -> bool:
        return self.expires is None or self.expires >= today
