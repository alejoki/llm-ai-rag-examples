"""
Demo 8 – Resumable AI Procurement Agent (LangGraph Persistence + Interrupt)

Scenario: An AI agent handles purchase requests. When a purchase exceeds
€10,000 it must pause for manager approval — which may come hours or days later.

The graph:

  START → lookup_vendors → fetch_pricing → compare_quotes
        → request_approval (INTERRUPTS here — process exits!)
        → submit_purchase_order → notify_employee → END

To simulate a real-world "late second invocation" across process restarts,
we use SqliteSaver (file-based checkpoint) and two CLI modes:

  python demo8.1-purchase-agent.py              # First run  — steps 1-3, then suspends
  python demo8.1-purchase-agent.py --resume     # Second run — manager approves, steps 5-6

Between the two runs the Python process exits completely.  The full agent
state (vendor data, pricing, chosen quote) survives on disk in SQLite.
"""

import re
import sys
import os
import sqlite3
import time
import requests
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    request: str
    quantity: int
    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict
    approval_status: str
    po_number: str
    notification: str


# ─── LLM (used only for the notification step to make it feel "agentic") ─────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ─── Helpers ─────────────────────────────────────────────────────────────────

LAPTOPS_API_URL = "https://dummyjson.com/products/category/laptops"

SHIPPING_DAYS = {
    "ships overnight": 1,
    "ships in 1 week": 7,
    "ships in 2 weeks": 14,
    "ships in 1 month": 30,
    "ships in 1-2 business days": 2,
    "ships in 3-5 business days": 5,
}

def parse_shipping_days(shipping_info: str) -> int:
    """Convert shippingInformation text to an integer number of days."""
    return SHIPPING_DAYS.get(shipping_info.lower(), 30)


def fetch_laptops_from_api() -> list[dict]:
    """Fetch laptop products from dummyjson.com. Returns [] on failure."""
    try:
        resp = requests.get(LAPTOPS_API_URL, timeout=10)
        resp.raise_for_status()
        return resp.json().get("products", [])
    except Exception as e:
        print(f"   WARNING: API request failed: {e}")
        return []


# ─── Tools ───────────────────────────────────────────────────────────────────

# Populated at runtime by lookup_vendors; keyed by product title
_product_prices: dict[str, float] = {}

@tool
def get_unit_price(product: str) -> float:
    """Look up the current unit price for a given product. Returns the price in euros."""
    price = _product_prices.get(product)
    if price is None:
        return -1.0  # unknown product
    return price

llm_with_tools = llm.bind_tools([get_unit_price])


# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    """Step 1: Fetch laptops from dummyjson API, filter by availability within 2 weeks."""
    print("\n[Step 1] Fetching laptop catalogue from dummyjson.com...")

    products = fetch_laptops_from_api()

    if not products:
        # Fallback: use hardcoded defaults and log a warning
        print("   WARNING: No products from API — using hardcoded fallback")
        vendors = [
            {"name": "Generic Laptop", "brand": "Unknown", "unit_price": 999.99,
             "shipping_days": 14, "rating": 3.0, "stock": 100},
        ]
        _product_prices["Generic Laptop"] = 999.99
        for v in vendors:
            print(f"   [fallback] {v['name']} - EUR {v['unit_price']}")
        return {"vendors": vendors}

    # Filter: in stock and ships within 2 weeks (14 days)
    eligible = []
    for p in products:
        days = parse_shipping_days(p.get("shippingInformation", ""))
        if p.get("availabilityStatus") == "In Stock" and days <= 14:
            eligible.append({
                "name": p["title"],
                "brand": p.get("brand", "Unknown"),
                "unit_price": p["price"],
                "shipping_days": days,
                "rating": p.get("rating", 0),
                "stock": p.get("stock", 0),
            })
            # Register price so the tool can look it up
            _product_prices[p["title"]] = p["price"]

    if not eligible:
        print("   WARNING: No laptops available within 2 weeks — using cheapest as fallback")
        cheapest = min(products, key=lambda p: p["price"])
        days = parse_shipping_days(cheapest.get("shippingInformation", ""))
        eligible = [{
            "name": cheapest["title"],
            "brand": cheapest.get("brand", "Unknown"),
            "unit_price": cheapest["price"],
            "shipping_days": days,
            "rating": cheapest.get("rating", 0),
            "stock": cheapest.get("stock", 0),
        }]
        _product_prices[cheapest["title"]] = cheapest["price"]

    # Sort by price (cheapest first)
    eligible.sort(key=lambda v: v["unit_price"])

    for v in eligible:
        print(f"   Found: {v['name']} ({v['brand']}) "
              f"- EUR {v['unit_price']} | ships in {v['shipping_days']}d "
              f"| rating {v['rating']} | stock {v['stock']}")

    return {"vendors": eligible}


def fetch_pricing(state: ProcurementState) -> dict:
    """Step 2: Use LLM tool calls to fetch unit prices, then compute totals."""
    print("\n[Step 2] Fetching pricing from suppliers via tool calls...")

    # --- Parse quantity from the request string ---
    match = re.search(r"(\d+)", state["request"])
    quantity = int(match.group(1)) if match else 50
    print(f"   Parsed quantity: {quantity}")

    # --- Ask the LLM to call get_unit_price for each product ---
    product_names = [v["name"] for v in state["vendors"]]
    prompt = (
        f"I need the unit price for each of these products: {', '.join(product_names)}. "
        f"Call the get_unit_price tool once for each product."
    )
    ai_msg = llm_with_tools.invoke([HumanMessage(content=prompt)])

    # --- Execute every tool call the LLM made ---
    tool_results = {}
    tool_messages = []
    for tc in ai_msg.tool_calls:
        product_arg = tc["args"]["product"]
        price = get_unit_price.invoke(tc["args"])
        tool_results[product_arg] = price
        tool_messages.append(
            ToolMessage(content=str(price), tool_call_id=tc["id"])
        )
        print(f"   Tool call: get_unit_price({product_arg!r}) -> EUR {price}")

    # --- Build quotes from the tool results + vendor metadata ---
    vendor_map = {v["name"]: v for v in state["vendors"]}
    quotes = []
    for product_name, unit_price in tool_results.items():
        vendor_info = vendor_map.get(product_name, {})
        total = unit_price * quantity
        delivery = vendor_info.get("shipping_days", 14)
        quotes.append({
            "vendor": product_name,
            "brand": vendor_info.get("brand", "Unknown"),
            "unit_price": unit_price,
            "total": total,
            "delivery_days": delivery,
        })
        print(f"   {product_name}: EUR {unit_price}/unit x {quantity} = EUR {total:,.2f} "
              f"({delivery}d delivery)")

    return {"quantity": quantity, "quotes": quotes}


def compare_quotes(state: ProcurementState) -> dict:
    """Step 3: Compare quotes and pick the best one."""
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)
    best = min(state["quotes"], key=lambda q: q["total"])
    print(f"   Best quote: {best['vendor']} at EUR {best['total']:,.2f}")
    if len(state["quotes"]) > 1:
        most_expensive = max(q["total"] for q in state["quotes"])
        print(f"   (Saves EUR {most_expensive - best['total']:,.2f} vs most expensive option)")
    return {"best_quote": best}


def request_approval(state: ProcurementState) -> dict:
    """Step 4: Human-in-the-loop — request manager approval for orders > €10,000."""
    best = state["best_quote"]
    qty = state.get("quantity", 50)
    print("\n[Step 4] Order exceeds EUR 10,000 -- manager approval required!")
    print(f"   Sending approval request to manager...")
    amount_str = f"EUR {best['total']:,.2f}"
    delivery_str = f"{best['delivery_days']} days"
    product_name = best["vendor"]
    brand = best.get("brand", "")

    w = 50  # box inner width
    print(f"   +{'-' * w}+")
    print(f"   | {'APPROVAL NEEDED':<{w}}|")
    print(f"   | {'Product: ' + product_name:<{w}}|")
    if brand:
        print(f"   | {'Brand:   ' + brand:<{w}}|")
    print(f"   | {'Amount:  ' + amount_str:<{w}}|")
    print(f"   | {'Items:   ' + str(qty) + ' units':<{w}}|")
    print(f"   | {'Delivery: ' + delivery_str:<{w}}|")
    print(f"   +{'-' * w}+")

    # ── THIS IS WHERE THE MAGIC HAPPENS ──
    # interrupt() freezes the entire graph state into the checkpoint store.
    # The process can now exit completely. When resumed later (even days later),
    # execution continues right here with the resume value.
    decision = interrupt({
        "message": f"Approve purchase of {qty}x {product_name} for EUR {best['total']:,.2f}?",
        "product": product_name,
        "amount": best["total"],
    })

    print(f"\n[Step 4] Manager responded: {decision}")
    return {"approval_status": decision}


def submit_purchase_order(state: ProcurementState) -> dict:
    """Step 5: Submit the purchase order to the ERP system."""
    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)
    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Product: {state['best_quote']['vendor']}")
    print(f"   Amount: EUR {state['best_quote']['total']:,.2f}")
    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    """Step 6: Use LLM to draft and send a notification to the employee."""
    print("\n[Step 6] Notifying employee...")

    qty = state.get("quantity", 50)
    approval = state.get("approval_status", "")
    rejected = approval and "reject" in approval.lower()
    product_name = state["best_quote"]["vendor"]

    if rejected:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request for {qty}x {product_name} was rejected by the manager. "
            f"Reason given: \"{approval}\". Be empathetic but concise."
        )
    else:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request has been approved and processed. "
            f"Details: {qty}x {product_name}, "
            f"EUR {state['best_quote']['total']:,.2f}, PO number {state['po_number']}, "
            f"delivery in {state['best_quote']['delivery_days']} days."
        )

    response = llm.invoke(prompt)
    notification = response.content
    print(f"   Employee notification sent:")
    print(f"   \"{notification}\"")
    return {"notification": notification}


# ─── Routing ─────────────────────────────────────────────────────────────────

def needs_approval(state: ProcurementState) -> str:
    """Route after compare_quotes: approval required only when total > €10,000."""
    total = state["best_quote"]["total"]
    if total > 10_000:
        print(f"\n   Total EUR {total:,.0f} exceeds EUR 10,000 -- routing to manager approval")
        return "request_approval"
    print(f"\n   Total EUR {total:,.0f} is under EUR 10,000 -- auto-approved, skipping manager")
    return "submit_purchase_order"


def check_approval(state: ProcurementState) -> str:
    """Route after request_approval: approved → purchase order, rejected → notify."""
    status = state.get("approval_status", "")
    if "reject" in status.lower():
        print(f"\n   Manager rejected — skipping purchase order")
        return "notify_employee"
    print(f"\n   Manager approved — proceeding to purchase order")
    return "submit_purchase_order"


# ─── Build the graph ─────────────────────────────────────────────────────────
#
#   START → lookup_vendors → fetch_pricing → compare_quotes
#         ──(>€10k)──→ request_approval (INTERRUPT)
#                       ──(approved)──→ submit_purchase_order → notify_employee → END
#                       ──(rejected)──→ notify_employee → END
#         ──(≤€10k)──→ submit_purchase_order → notify_employee → END

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")
builder.add_conditional_edges("compare_quotes", needs_approval)
builder.add_conditional_edges("request_approval", check_approval)
builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Checkpointer (SQLite — survives process restarts!) ──────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    """First run: employee submits request, agent does steps 1-3, then suspends."""
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)
    print("\nEmployee request: \"Order 67 laptops for the sales team\"")

    result = graph.invoke(
        {"request": "Order 67 laptops for the sales team"},
        config,
    )

    # Check whether the graph was interrupted (needs approval) or ran to completion
    saved_state = graph.get_state(config)
    if saved_state and saved_state.next:
        # Graph was interrupted — waiting for manager
        print("\n" + "=" * 60)
        print("AGENT SUSPENDED — waiting for manager approval")
        print("=" * 60)
        print("\n  The agent process can now exit completely.")
        print("  All state (vendors, pricing, best quote) is frozen in SQLite.")
        print(f"  Checkpoint DB: {DB_PATH}")
        print(f"  Thread ID: {THREAD_ID}")
        print("\n  In a real system, the manager gets a Slack/email notification.")
        print("  They might respond hours or even days later.\n")
        print("  To resume, run:")
        print(f"    python {os.path.basename(__file__)} --resume\n")
    else:
        # Graph completed without interrupt — order was auto-approved
        print("\n" + "=" * 60)
        print("PROCUREMENT COMPLETE (auto-approved, no manager needed)")
        print("=" * 60)
        print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
        print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
        print(f"  Total:        EUR {result.get('best_quote', {}).get('total', 0):,.0f}")
        print()


def run_second_invocation(graph):
    """Second run: manager approves, agent wakes up at step 5 with full context."""
    print("=" * 60)
    print("  SECOND INVOCATION — Manager approves (maybe days later!)")
    print("=" * 60)

    # Show that the state survived the process restart
    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")
    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,}")
    print(f"\n  Steps 1-3 are NOT re-executed — their output is in the checkpoint!\n")

    # Resume with the manager's decision
    # Use --reject flag to simulate a rejection, otherwise approve
    if "--reject" in sys.argv:
        decision = "Rejected — over budget"
        print("Manager clicks [REJECT] ...")
    else:
        decision = "Approved — go ahead with the purchase."
        print("Manager clicks [APPROVE] ...")
    time.sleep(1)

    result = graph.invoke(
        Command(resume=decision),
        config,
    )

    approval = result.get("approval_status", "")
    rejected = approval and "reject" in approval.lower()

    print("\n" + "=" * 60)
    if rejected:
        print("PROCUREMENT REJECTED")
    else:
        print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  Approval:     {approval}")
    if not rejected:
        print(f"  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Total:        EUR {result.get('best_quote', {}).get('total', 0):,.0f}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    # Clean start if not resuming
    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            run_second_invocation(graph)
        else:
            run_first_invocation(graph)
    finally:
        conn.close()
