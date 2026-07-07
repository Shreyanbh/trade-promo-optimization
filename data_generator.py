"""
DataGenerator — creates realistic raw source files in multiple formats.

Structured   : data/raw/structured/  (CSV, Excel, JSON)
Unstructured : data/raw/unstructured/ (TXT reviews, call transcripts, emails)

All IDs are consistent across sources so joins work cleanly downstream.
"""
import os
import json
import random
import string
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("data_generator")

RAW_STRUCTURED   = os.path.join(PATHS.get("raw_data", "data/raw"), "structured")
RAW_UNSTRUCTURED = os.path.join(PATHS.get("raw_data", "data/raw"), "unstructured")

REGIONS   = ["North", "South", "East", "West"]
CITIES    = {"North": ["Manchester", "Leeds", "Newcastle"],
             "South": ["London", "Brighton", "Southampton"],
             "East":  ["Cambridge", "Norwich", "Ipswich"],
             "West":  ["Bristol", "Cardiff", "Exeter"]}
CATEGORIES = ["beverages", "snacks", "dairy", "produce", "household"]
SUBCATS    = {"beverages": ["hot drinks", "soft drinks", "juices"],
              "snacks":    ["crisps", "biscuits", "confectionery"],
              "dairy":     ["milk", "cheese", "yogurt"],
              "produce":   ["fruits", "vegetables", "salads"],
              "household": ["cleaning", "personal care", "kitchen"]}
CHANNELS   = ["in-store", "online", "mobile-app", "click-and-collect"]
PAYMENT    = ["card", "cash", "digital-wallet", "loyalty-points"]
LOYALTY    = ["bronze", "silver", "gold", "platinum"]
INCOME     = ["<30k", "30-50k", "50-80k", "80-120k", ">120k"]
FIRST_NAMES = ["James","Emma","Oliver","Sophia","William","Ava","Ethan","Isabella",
               "Mason","Mia","Lucas","Charlotte","Aiden","Amelia","Liam","Harper",
               "Noah","Evelyn","Logan","Abigail"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
               "Davis","Wilson","Taylor","Anderson","Thomas","Jackson","White","Harris"]
BRANDS     = ["FreshCo","NaturePlus","HomeEssentials","DailyBest","PureChoice"]
PROMO_NAMES = ["Summer Flash Sale","Loyalty Reward","Bundle Bonanza","Weekend Deal",
               "New Member Offer","Clearance Event","Seasonal Discount","VIP Preview",
               "Flash Friday","Bulk Buy Saver","Category Week","Brand Spotlight",
               "Digital Exclusive","Senior Saver","Student Deal","Family Pack",
               "Happy Hour","Refer a Friend","Birthday Bonus","Re-engagement Offer"]

POSITIVE_PHRASES = [
    "Absolutely love this product! Fast delivery and great quality.",
    "Exceeded my expectations. Will definitely reorder soon.",
    "Fantastic value for money. The quality is outstanding.",
    "Perfect product. Arrived on time and exactly as described.",
    "Really impressed. This has become my go-to brand.",
    "Five stars all around. Superb quality and packaging.",
    "Great product, works exactly as advertised. Very happy.",
    "Excellent purchase. My whole family loves it.",
]
NEGATIVE_PHRASES = [
    "Very disappointed. Product didn't match the description at all.",
    "Poor quality for the price. Expected much better.",
    "Arrived damaged and the packaging was completely crushed.",
    "Not what I ordered. Customer service was unhelpful.",
    "Returned the product. It broke within a week of use.",
    "Terrible experience. Will not be purchasing again.",
    "Product tasted off. Not fresh at all. Very unhappy.",
]
NEUTRAL_PHRASES = [
    "Decent product. Does what it says on the tin.",
    "Average quality. Nothing special but gets the job done.",
    "It's okay. Not as good as the previous version.",
    "Product is fine. Delivery took longer than expected.",
    "Satisfactory. I've had better but also much worse.",
    "Reasonable quality. Might try a different brand next time.",
]


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_all(n_customers: int = 500, n_products: int = 50,
                 n_transactions: int = 5000, n_promos: int = 20,
                 force: bool = False) -> dict:
    """Generate all raw source files. Returns dict of file paths created."""
    os.makedirs(RAW_STRUCTURED, exist_ok=True)
    os.makedirs(RAW_UNSTRUCTURED, exist_ok=True)

    rng    = _rng()
    paths  = {}
    cids   = [f"C{i:04d}" for i in range(n_customers)]
    pids   = [f"P{i:03d}" for i in range(n_products)]

    paths["customers_csv"]     = _customers_csv(cids, rng, force)
    paths["transactions_csv"]  = _transactions_csv(cids, pids, n_transactions, rng, force)
    paths["products_xlsx"]     = _products_xlsx(pids, rng, force)
    paths["promotions_json"]   = _promotions_json(pids, n_promos, rng, force)
    paths["reviews_txt"]       = _reviews_txt(cids, pids, rng, force)
    paths["transcripts_txt"]   = _transcripts_txt(cids, pids, rng, force)
    paths["emails_txt"]        = _emails_txt(cids, pids, rng, force)

    log.info(f"Generated {len(paths)} raw source files")
    return paths


# ── Structured sources ────────────────────────────────────────────────────────

def _customers_csv(cids, rng, force):
    path = os.path.join(RAW_STRUCTURED, "customers.csv")
    if os.path.exists(path) and not force:
        return path
    n = len(cids)
    regions = rng.choice(REGIONS, n)
    rows = []
    for i, cid in enumerate(cids):
        region = regions[i]
        city   = rng.choice(CITIES[region])
        rows.append({
            "customer_id":    cid,
            "first_name":     rng.choice(FIRST_NAMES),
            "last_name":      rng.choice(LAST_NAMES),
            "email":          f"customer_{cid.lower()}@email.com",
            "age":            int(rng.integers(18, 75)),
            "region":         region,
            "city":           city,
            "income_bracket": rng.choice(INCOME),
            "loyalty_tier":   rng.choice(LOYALTY),
            "signup_date":    (datetime(2020, 1, 1) +
                               timedelta(days=int(rng.integers(0, 1460)))).strftime("%Y-%m-%d"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"  Wrote {path} ({len(df)} rows, CSV)")
    return path


def _transactions_csv(cids, pids, n, rng, force):
    path = os.path.join(RAW_STRUCTURED, "transactions.csv")
    if os.path.exists(path) and not force:
        return path
    base = datetime(2023, 1, 1)
    rows = [{
        "transaction_id":    f"TXN{i:05d}",
        "customer_id":       rng.choice(cids),
        "product_id":        rng.choice(pids),
        "quantity":          int(rng.integers(1, 6)),
        "unit_price":        round(float(rng.uniform(1, 50)), 2),
        "amount":            round(float(rng.uniform(2, 200)), 2),
        "date":              (base + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d"),
        "channel":           rng.choice(CHANNELS),
        "store_id":          f"STORE{int(rng.integers(1, 21)):02d}",
        "payment_method":    rng.choice(PAYMENT),
        "promo_code_applied": f"PROMO{int(rng.integers(1, 21)):03d}" if rng.random() < 0.3 else "",
    } for i in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"  Wrote {path} ({len(df)} rows, CSV)")
    return path


def _products_xlsx(pids, rng, force):
    path = os.path.join(RAW_STRUCTURED, "products.xlsx")
    if os.path.exists(path) and not force:
        return path
    categories = [rng.choice(CATEGORIES) for _ in pids]
    products = pd.DataFrame([{
        "product_id":   pid,
        "product_name": f"Product_{pid[1:]}",
        "category":     cat,
        "subcategory":  rng.choice(SUBCATS[cat]),
        "brand":        rng.choice(BRANDS),
        "price":        round(float(rng.uniform(1, 50)), 2),
        "cost_price":   round(float(rng.uniform(0.5, 25)), 2),
        "sku":          "SKU-" + "".join(rng.choice(list(string.ascii_uppercase), 4)),
    } for pid, cat in zip(pids, categories)])
    inventory = pd.DataFrame([{
        "product_id":         pid,
        "stock_level":        int(rng.integers(0, 500)),
        "reorder_threshold":  int(rng.integers(20, 100)),
        "warehouse_location": f"WH{int(rng.integers(1, 4))}-AISLE{int(rng.integers(1, 10))}",
    } for pid in pids])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        products.to_excel(w,  sheet_name="Products",  index=False)
        inventory.to_excel(w, sheet_name="Inventory", index=False)
    log.info(f"  Wrote {path} (2 sheets: Products + Inventory, XLSX)")
    return path


def _promotions_json(pids, n, rng, force):
    path = os.path.join(RAW_STRUCTURED, "promotions.json")
    if os.path.exists(path) and not force:
        return path
    base   = datetime(2023, 1, 1)
    promos = []
    for i in range(n):
        start = base + timedelta(days=int(rng.integers(0, 300)))
        cats  = list(rng.choice(CATEGORIES, size=int(rng.integers(1, 4)), replace=False))
        prods = [str(p) for p in rng.choice(pids, size=int(rng.integers(3, 10)), replace=False)]
        promos.append({
            "promo_id":              f"PROMO{i+1:03d}",
            "name":                  PROMO_NAMES[i % len(PROMO_NAMES)],
            "type":                  rng.choice(["percentage", "fixed", "bogo"]),
            "discount_value":        round(float(rng.uniform(5, 40)), 1),
            "start_date":            start.strftime("%Y-%m-%d"),
            "end_date":              (start + timedelta(days=int(rng.integers(7, 45)))).strftime("%Y-%m-%d"),
            "eligible_categories":   cats,
            "eligible_products":     prods,
            "min_purchase_amount":   round(float(rng.uniform(10, 50)), 2),
            "target_loyalty_tiers":  list(rng.choice(LOYALTY, size=int(rng.integers(1, 3)), replace=False)),
            "max_uses_per_customer": int(rng.integers(1, 5)),
            "channel":               rng.choice(CHANNELS + ["all"]),
        })
    with open(path, "w") as f:
        json.dump(promos, f, indent=2)
    log.info(f"  Wrote {path} ({n} promos, JSON)")
    return path


# ── Unstructured sources ──────────────────────────────────────────────────────

def _reviews_txt(cids, pids, rng, force):
    path = os.path.join(RAW_UNSTRUCTURED, "customer_reviews.txt")
    if os.path.exists(path) and not force:
        return path
    lines = []
    for i in range(200):
        cid    = rng.choice(cids)
        pid    = rng.choice(pids)
        rating = int(rng.integers(1, 6))
        d      = (datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        if rating >= 4:
            text = rng.choice(POSITIVE_PHRASES)
        elif rating <= 2:
            text = rng.choice(NEGATIVE_PHRASES)
        else:
            text = rng.choice(NEUTRAL_PHRASES)
        lines += [
            f"REVIEW [R{i+1:03d}] | CUSTOMER: {cid} | PRODUCT: {pid} | DATE: {d} | RATING: {rating}/5",
            f'"{text}"',
            "---",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} (200 reviews, TXT)")
    return path


def _transcripts_txt(cids, pids, rng, force):
    path = os.path.join(RAW_UNSTRUCTURED, "call_transcripts.txt")
    if os.path.exists(path) and not force:
        return path
    agent_names = ["Sarah", "Michael", "Emily", "James", "Priya", "Tom"]
    issues = [
        ("damaged product",    "Replacement order created and escalated to logistics."),
        ("missing item",       "Refund issued and repeat order dispatched."),
        ("billing query",      "Billing corrected and credit applied to account."),
        ("delivery delay",     "Tracked shipment, customer updated with new ETA."),
        ("product complaint",  "Return label sent, full refund processed."),
        ("account issue",      "Password reset completed, account verified."),
        ("promo code failed",  "Discount manually applied and order re-processed."),
    ]
    sentiments = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    durations  = [f"{m}m {s}s" for m in range(2, 20) for s in [15, 30, 45]]
    lines = []
    for i in range(60):
        cid     = rng.choice(cids)
        pid     = rng.choice(pids)
        agent   = rng.choice(agent_names)
        issue, resolution = rng.choice(issues)
        sent    = rng.choice(sentiments, p=[0.45, 0.35, 0.20])
        d       = (datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        dur     = rng.choice(durations)
        lines += [
            f"CALL [T{i+1:03d}] | DATE: {d} | AGENT: {agent} | CUSTOMER: {cid} | DURATION: {dur}",
            f"PRODUCT REFERENCED: {pid}",
            f"Issue: {issue.title()}",
            f"Customer: 'I have an issue with {issue} regarding product {pid}.'",
            f"Agent: 'I understand your frustration. Let me look into that for you.'",
            f"RESOLUTION: {resolution}",
            f"SENTIMENT: {sent}",
            "---",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} (60 call transcripts, TXT)")
    return path


def _emails_txt(cids, pids, rng, force):
    path = os.path.join(RAW_UNSTRUCTURED, "support_emails.txt")
    if os.path.exists(path) and not force:
        return path
    subjects = [
        "Order Inquiry", "Missing Delivery", "Return Request",
        "Billing Dispute", "Product Feedback", "Account Problem",
        "Promo Code Issue", "Damaged Goods", "Subscription Query",
    ]
    lines = []
    for i in range(100):
        cid  = rng.choice(cids)
        pid  = rng.choice(pids)
        subj = rng.choice(subjects)
        d    = (datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        senti = rng.choice(["POSITIVE", "NEUTRAL", "NEGATIVE"], p=[0.2, 0.45, 0.35])
        lines += [
            f"EMAIL [E{i+1:03d}] | DATE: {d} | CUSTOMER: {cid} | SUBJECT: {subj}",
            f"FROM: customer_{cid.lower()}@email.com",
            f"PRODUCT: {pid}",
            f"---",
            f"Dear Support Team,",
            f"I am writing regarding my recent experience with product {pid}. "
            f"[Subject: {subj}]",
            f"SENTIMENT_LABEL: {senti}",
            f"RESPONSE: Issue acknowledged. Customer notified within 24h.",
            "===",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} (100 support emails, TXT)")
    return path
