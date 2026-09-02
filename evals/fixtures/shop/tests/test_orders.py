import pytest

from shop.errors import NotFound, OutOfStock, ValidationError
from shop.inventory import Inventory
from shop.models import Address, Coupon, Customer, CustomerTier, OrderStatus, Product
from shop.money import Money, allocate, total
from shop.orders import OrderService
from shop.reporting import by_category, by_customer, by_status, revenue, split_commission
from shop.shipping import ShippingQuoter, WeightBanded, zone_for
from shop.storage import Store

US = Address(line1="1 Main St", city="Austin", postcode="78701", country="US")
DE = Address(line1="1 Hauptstr", city="Berlin", postcode="10115", country="DE")


def service():
    svc = OrderService(store=Store(), inventory=Inventory())
    svc.stock(Product(sku="WD-1001", name="Widget", unit_price=Money.from_major(10), weight_grams=400), 500)
    svc.stock(Product(sku="BK-2001", name="Book", unit_price=Money.from_major(20), category="book", weight_grams=300), 20)
    svc.store.coupons.add(Coupon(code="TAKE10", percent_off=10))
    return svc


def customer(tier=CustomerTier.STANDARD, address=US):
    return Customer(id="c1", email="a@b.com", tier=tier, address=address)


class TestMoney:
    def test_scale_rounds_half_away_from_zero(self):
        assert Money(1005).scale(500, 10_000) == Money(50)

    def test_from_major_rounds_half_up(self):
        assert Money.from_major(0.125).minor == 13

    def test_currencies_do_not_mix(self):
        from shop.errors import CurrencyMismatch

        with pytest.raises(CurrencyMismatch):
            Money(100, "USD") + Money(100, "EUR")

    def test_yen_has_no_minor_units(self):
        assert Money.from_major(1000, "JPY").format() == "1000 JPY"

    def test_allocate_never_loses_a_unit(self):
        parts = allocate(Money(100), [1, 1, 1])
        assert sum(p.minor for p in parts) == 100

    def test_allocate_favours_the_larger_weight(self):
        assert [p.minor for p in allocate(Money(10), [7, 3])] == [7, 3]

    def test_total_of_nothing_is_zero(self):
        assert total([], "EUR") == Money.zero("EUR")


class TestFlow:
    def test_place_reserves_stock(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 3)
        svc.place("o1")
        assert svc.inventory.available("WD-1001") == 497

    def test_place_prices_the_order(self):
        svc = service()
        svc.start("o1", customer(tier=CustomerTier.GOLD))
        svc.add_line("o1", "WD-1001", 10)
        placed = svc.place("o1")
        assert placed.quote.discount().minor > 0
        assert placed.total() == placed.quote.total()

    def test_cancel_releases_stock(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 5)
        svc.place("o1")
        svc.cancel("o1")
        assert svc.inventory.available("WD-1001") == 500

    def test_ship_removes_stock_for_good(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 5)
        svc.place("o1")
        svc.pay("o1")
        svc.ship("o1")
        assert svc.inventory.level("WD-1001").on_hand == 495

    def test_cannot_ship_before_paying(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 1)
        svc.place("o1")
        with pytest.raises(ValidationError):
            svc.ship("o1")

    def test_out_of_stock_is_refused(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "BK-2001", 50)
        with pytest.raises(OutOfStock):
            svc.place("o1")

    def test_a_placed_order_cannot_gain_lines_after_shipping(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 1)
        svc.place("o1")
        svc.pay("o1")
        svc.ship("o1")
        with pytest.raises(ValidationError):
            svc.add_line("o1", "WD-1001", 1)

    def test_empty_order_will_not_price(self):
        svc = service()
        svc.start("o1", customer())
        with pytest.raises(ValidationError):
            svc.quote("o1")

    def test_unknown_coupon_is_refused(self):
        svc = service()
        svc.start("o1", customer())
        with pytest.raises(NotFound):
            svc.apply_coupon("o1", "NOPE")


class TestShipping:
    def test_zones(self):
        assert zone_for("US") == "domestic"
        assert zone_for("DE") == "europe"
        assert zone_for("ZZ") == "rest"

    def test_free_over_threshold(self):
        svc = service()
        svc.start("o1", customer())
        svc.add_line("o1", "WD-1001", 10)
        assert svc.shipping.explain(svc.load("o1")) == ("free-over", Money.zero())

    def test_partners_ship_free(self):
        svc = service()
        svc.start("o1", customer(tier=CustomerTier.PARTNER))
        svc.add_line("o1", "WD-1001", 1)
        assert svc.shipping.explain(svc.load("o1"))[0] == "partner"

    def test_weight_banded_rounds_up_to_the_kilo(self):
        svc = service()
        svc.start("o1", customer(address=DE))
        svc.add_line("o1", "WD-1001", 3)
        name, amount = ShippingQuoter(rules=[WeightBanded()]).explain(svc.load("o1"))
        assert name == "weight"
        assert amount == Money(1800 + 340 * 2)


class TestReporting:
    def build(self):
        svc = service()
        for n, (oid, qty) in enumerate([("o1", 2), ("o2", 4), ("o3", 1)]):
            svc.start(oid, customer())
            svc.add_line(oid, "WD-1001", qty)
            svc.place(oid)
            if n < 2:
                svc.pay(oid)
        return svc

    def test_only_paid_orders_count(self):
        svc = self.build()
        assert by_status(svc.store.orders.all())[OrderStatus.PLACED] == 1
        assert revenue(svc.store.orders.all()).minor > 0

    def test_grouped_by_customer(self):
        rows = by_customer(self.build().store.orders.all())
        assert [r.key for r in rows] == ["c1"]
        assert rows[0].orders == 2

    def test_grouped_by_category(self):
        rows = by_category(self.build().store.orders.all())
        assert rows[0].key == "general"

    def test_commission_splits_without_loss(self):
        svc = self.build()
        parts = split_commission(svc.store.orders.all(), {"a": 2, "b": 1})
        assert parts["a"] + parts["b"] == revenue(svc.store.orders.all())
