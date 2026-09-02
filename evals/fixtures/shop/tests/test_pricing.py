from datetime import date

import pytest

from shop.models import Address, Coupon, Customer, CustomerTier, Order, Product
from shop.money import Money
from shop.pricing import (
    BundleDiscount,
    CategoryDiscount,
    CouponDiscount,
    PriceEngine,
    TaxTable,
    TierDiscount,
    VolumeDiscount,
    default_rules,
    price,
)

UK = Address(line1="1 High St", city="London", postcode="SW1A1AA", country="GB")
US = Address(line1="1 Main St", city="Austin", postcode="78701", country="US")


def widget(price_major=10.00, **kw):
    return Product(sku="WD-1001", name="Widget", unit_price=Money.from_major(price_major), **kw)


def order_of(quantity=1, tier=CustomerTier.STANDARD, address=US, product=None):
    customer = Customer(id="c1", email="a@b.com", tier=tier, address=address)
    order = Order(id="o1", customer=customer)
    order.add(product or widget(), quantity)
    return order


class TestVolume:
    def test_below_first_tier_is_free_of_discount(self):
        assert VolumeDiscount().tier_for(9) is None

    def test_ten_takes_five_percent(self):
        assert VolumeDiscount().tier_for(10) == (10, 500)

    def test_fifty_takes_ten_percent(self):
        assert VolumeDiscount().tier_for(50) == (50, 1000)

    def test_hundred_takes_fifteen_percent(self):
        assert VolumeDiscount().tier_for(100) == (100, 1500)

    def test_best_tier_wins_not_the_sum(self):
        order = order_of(quantity=100)
        quote = PriceEngine(rules=[VolumeDiscount()]).quote(order)
        assert quote.discount() == Money.from_major(150)

    def test_clearance_stock_takes_no_percentage(self):
        order = order_of(quantity=100, product=widget(discountable=False))
        assert price(order).discount() == Money.zero()


class TestTier:
    @pytest.mark.parametrize(
        "tier,bps",
        [
            (CustomerTier.STANDARD, 0),
            (CustomerTier.SILVER, 300),
            (CustomerTier.GOLD, 700),
            (CustomerTier.PARTNER, 1500),
        ],
    )
    def test_table(self, tier, bps):
        assert TierDiscount().bps_for(order_of(tier=tier)) == bps

    def test_gold_on_a_single_unit(self):
        order = order_of(quantity=1, tier=CustomerTier.GOLD)
        assert price(order).discount() == Money.from_major(0.70)


class TestStacking:
    def test_percentages_stack_against_the_running_price(self):
        # 10 units at 10.00 = 100.00. Volume takes 5% (5.00), leaving 95.00.
        # Gold then takes 7% of 95.00, not of 100.00.
        order = order_of(quantity=10, tier=CustomerTier.GOLD)
        quote = price(order)
        assert quote.discount() == Money.from_major(5.00) + Money.from_major(6.65)

    def test_reasons_name_each_rule(self):
        order = order_of(quantity=10, tier=CustomerTier.GOLD)
        assert set(price(order).reasons()) == {"volume", "tier"}

    def test_cap_limits_the_stack(self):
        engine = PriceEngine(
            rules=[VolumeDiscount(), TierDiscount(), CategoryDiscount("general", 5000)],
            max_discount_bps=6000,
        )
        order = order_of(quantity=100, tier=CustomerTier.PARTNER)
        quote = engine.quote(order)
        assert quote.discount() == order.subtotal().scale(6000, 10_000)

    def test_capped_reasons_still_sum_to_the_cap(self):
        engine = PriceEngine(
            rules=[VolumeDiscount(), TierDiscount(), CategoryDiscount("general", 5000)],
            max_discount_bps=6000,
        )
        quote = engine.quote(order_of(quantity=100, tier=CustomerTier.PARTNER))
        parts = sum(m.minor for m in quote.reasons().values())
        assert parts == quote.discount().minor


class TestCoupons:
    def test_percentage_coupon(self):
        order = order_of(quantity=1)
        order.coupon = "TAKE10"
        coupon = Coupon(code="TAKE10", percent_off=10)
        quote = PriceEngine(rules=[CouponDiscount(coupon)]).quote(order)
        assert quote.discount() == Money.from_major(1.00)

    def test_fixed_coupon_spreads_across_lines(self):
        customer = Customer(id="c1", email="a@b.com", address=US)
        order = Order(id="o1", customer=customer, coupon="FIVE")
        order.add(widget(10.00), 1)
        order.add(Product(sku="WD-1002", name="Other", unit_price=Money.from_major(30.00)), 1)
        coupon = Coupon(code="FIVE", amount_off=Money.from_major(5.00))
        quote = PriceEngine(rules=[CouponDiscount(coupon)]).quote(order)
        assert quote.discount() == Money.from_major(5.00)
        assert [p.discount.minor for p in quote.lines] == [125, 375]

    def test_expired_coupon_does_nothing(self):
        order = order_of()
        order.coupon = "OLD"
        order.placed_on = date(2026, 6, 1)
        coupon = Coupon(code="OLD", percent_off=50, expires=date(2026, 1, 1))
        assert PriceEngine(rules=[CouponDiscount(coupon)]).quote(order).discount() == Money.zero()

    def test_minimum_blocks_a_small_basket(self):
        order = order_of()
        order.coupon = "BIG"
        coupon = Coupon(code="BIG", percent_off=50, minimum=Money.from_major(500))
        assert PriceEngine(rules=[CouponDiscount(coupon)]).quote(order).discount() == Money.zero()


class TestBundle:
    def test_incomplete_bundle_does_nothing(self):
        order = order_of()
        rule = BundleDiscount(["WD-1001", "WD-9999"], 1000)
        assert PriceEngine(rules=[rule]).quote(order).discount() == Money.zero()

    def test_complete_bundle_discounts_both(self):
        customer = Customer(id="c1", email="a@b.com", address=US)
        order = Order(id="o1", customer=customer)
        order.add(widget(10.00), 1)
        order.add(Product(sku="WD-1002", name="Other", unit_price=Money.from_major(10.00)), 1)
        rule = BundleDiscount(["WD-1001", "WD-1002"], 1000)
        assert PriceEngine(rules=[rule]).quote(order).discount() == Money.from_major(2.00)


class TestTax:
    def test_us_has_no_federal_rate(self):
        assert TaxTable().rate_bps("US", "general") == 0

    def test_uk_general(self):
        assert TaxTable().rate_bps("GB", "general") == 2000

    def test_uk_books_are_zero_rated(self):
        assert TaxTable().rate_bps("GB", "book") == 0

    def test_german_books_take_the_reduced_rate(self):
        assert TaxTable().rate_bps("DE", "book") == 700

    def test_tax_is_charged_on_the_discounted_price(self):
        order = order_of(quantity=10, address=UK)
        quote = price(order)
        assert quote.tax() == quote.lines[0].net().scale(2000, 10_000)

    def test_exempt_customer_pays_none(self):
        order = order_of(address=UK)
        order.customer.tax_exempt = True
        assert price(order).tax() == Money.zero()

    def test_non_taxable_product_pays_none(self):
        order = order_of(address=UK, product=widget(taxable=False))
        assert price(order).tax() == Money.zero()


class TestQuoteTotals:
    def test_total_is_subtotal_less_discount_plus_tax_and_shipping(self):
        order = order_of(quantity=10, tier=CustomerTier.GOLD, address=UK)
        engine = PriceEngine(rules=default_rules())
        quote = engine.quote(order, shipping=Money.from_major(4.99))
        expected = quote.subtotal() - quote.discount() + quote.tax() + Money.from_major(4.99)
        assert quote.total() == expected

    def test_pricing_writes_back_onto_the_lines(self):
        order = order_of(quantity=10, tier=CustomerTier.GOLD, address=UK)
        price(order)
        assert order.lines[0].discount is not None
        assert order.lines[0].tax is not None
        assert order.discount_total() == order.lines[0].discount
