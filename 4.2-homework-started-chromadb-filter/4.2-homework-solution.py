"""
ChromaDB Search & Filter Exercises
===================================
Starter code — shared knowledge base for all four exercises.

Install dependency:
    pip install chromadb

Run:
    python chroma_exercises_starter.py
"""

import chromadb

# ---------------------------------------------------------------------------
# Knowledge base — a fictional university IT helpdesk
# ---------------------------------------------------------------------------

documents = [
    # VPN
    "To connect to the university VPN, install the GlobalProtect client from software.uni.fi and authenticate with your student credentials.",
    "The VPN service is maintained every Tuesday between 06:00 and 07:00. Connections will be interrupted during this window.",
    "If the VPN disconnects repeatedly, try switching the server region from 'EU-West' to 'EU-North' in GlobalProtect settings.",

    # Email
    "Student email accounts are hosted on Microsoft 365. Your address is firstname.lastname@student.uni.fi.",
    "Email accounts are deactivated 12 months after graduation. Export your data before that deadline using the IT portal.",
    "The maximum email attachment size is 25 MB. Use OneDrive to share larger files.",

    # Software
    "MATLAB is available for all students via the campus licence. Download it from software.uni.fi using your student ID.",
    "Microsoft Office 365 is included in your student account at no cost. Install up to five devices.",
    "Adobe Creative Cloud is available for ICT and Design students only. Request access through the IT helpdesk portal.",

    # Network / Wi-Fi
    "The campus Wi-Fi network is called UniNet. Use your student credentials to authenticate via the captive portal.",
    "Eduroam is available on campus and at partner universities worldwide. Configure it with your full student email address.",
    "Wired ethernet ports in library study rooms operate at 1 Gbps. Contact IT if a port is inactive.",

    # Accounts & passwords
    "Passwords must be at least 12 characters and include uppercase, lowercase, a digit, and a special character.",
    "Your student account is created within 24 hours of enrolment confirmation. Check your personal email for the activation link.",
    "Multi-factor authentication (MFA) is mandatory for all student accounts from September 2025 onwards.",

    # Printing
    "Campus printers accept UniPrint credits. Load credits at any campus info desk or online at print.uni.fi.",
    "The default print quota is 200 pages per semester. ICT students receive an additional 100 pages.",
    "Colour printing costs 0.15 EUR per page. Black-and-white printing costs 0.04 EUR per page.",
]

metadatas = [
    # VPN
    {"category": "vpn",      "priority": "high",   "year": 2025, "verified": True},
    {"category": "vpn",      "priority": "medium", "year": 2025, "verified": True},
    {"category": "vpn",      "priority": "low",    "year": 2024, "verified": True},
    # Email
    {"category": "email",    "priority": "high",   "year": 2025, "verified": True},
    {"category": "email",    "priority": "medium", "year": 2024, "verified": True},
    {"category": "email",    "priority": "low",    "year": 2025, "verified": False},
    # Software
    {"category": "software", "priority": "medium", "year": 2025, "verified": True},
    {"category": "software", "priority": "low",    "year": 2025, "verified": True},
    {"category": "software", "priority": "low",    "year": 2024, "verified": False},
    # Network
    {"category": "network",  "priority": "high",   "year": 2025, "verified": True},
    {"category": "network",  "priority": "high",   "year": 2025, "verified": True},
    {"category": "network",  "priority": "low",    "year": 2024, "verified": True},
    # Accounts
    {"category": "accounts", "priority": "high",   "year": 2025, "verified": True},
    {"category": "accounts", "priority": "high",   "year": 2025, "verified": True},
    {"category": "accounts", "priority": "high",   "year": 2025, "verified": True},
    # Printing
    {"category": "printing", "priority": "medium", "year": 2025, "verified": True},
    {"category": "printing", "priority": "low",    "year": 2025, "verified": True},
    {"category": "printing", "priority": "low",    "year": 2025, "verified": True},
]

ids = [f"doc-{i:03d}" for i in range(len(documents))]

# ---------------------------------------------------------------------------
# Initialise collection
# ---------------------------------------------------------------------------

client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("it_helpdesk")

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

print(f"Collection ready — {collection.count()} documents loaded.\n")
print("=" * 60)

# ---------------------------------------------------------------------------
# Helper — print results neatly
# ---------------------------------------------------------------------------

def print_results(label, results, show_distances=False):
    """Print get() or query() results in a readable format."""
    print(f"\n>>> {label}")

    # query() results are nested one level deeper than get()
    is_query = isinstance(results["ids"][0], list)

    ids_list      = results["ids"][0]       if is_query else results["ids"]
    docs_list     = results["documents"][0] if is_query else results["documents"]
    metas_list    = results["metadatas"][0] if is_query else results["metadatas"]
    dists_list    = results.get("distances", [[]])[0] if is_query else []

    if not ids_list:
        print("  (no results)")
        return

    for i, (doc_id, doc, meta) in enumerate(zip(ids_list, docs_list, metas_list)):
        dist_str = f"  distance={dists_list[i]:.4f}" if show_distances and dists_list else ""
        print(f"  [{doc_id}] [{meta['category']}] [{meta['priority']}]{dist_str}")
        print(f"    {doc[:90]}{'...' if len(doc) > 90 else ''}")


# ---------------------------------------------------------------------------
# EXERCISE 1 — Metadata filtering with where
# ---------------------------------------------------------------------------
# TODO: Use collection.get() with a where filter to fetch all documents
# where category == "vpn"
#
# Expected: 3 documents (doc-000, doc-001, doc-002)

print("\n--- EXERCISE 1: Basic metadata filter ---")

# Write your code here:
results = collection.get(
    where={"category": "vpn"}
)
print_results("All VPN documents:\n", results)


# ---------------------------------------------------------------------------
# EXERCISE 2 — Combining filters with $and and $or
# ---------------------------------------------------------------------------
# TODO: Use collection.get() with a combined where filter to find documents
# where priority == "high" AND year == 2025 AND verified == True
#
# Expected: 7 documents covering vpn, email, network, accounts categories

print("\n--- EXERCISE 2: Combined metadata filters ---")

# Write your code here:
results = collection.get(
    where={
        "$and": [
            {"priority": "high"},
            {"year": 2025},
            {"verified": True}
        ]
    }
)
print_results("High priority, 2025, verified:\n", results)


# ---------------------------------------------------------------------------
# EXERCISE 3 — Full text search with where_document
# ---------------------------------------------------------------------------
# TODO A: Use collection.get() with where_document to find all documents
#         that contain the word "student"
#
# TODO B: Extend it — find documents that contain "student" but do NOT
#         contain "password" (use $and with $contains and $not_contains)

print("\n--- EXERCISE 3: Full text search ---")

# Write your code here:

# TODO A: Documents containing "student"
results_a = collection.get(
    where_document={"$contains": "student"}
)
print_results("Documents containing 'student':\n", results_a)

# TODO B: Documents containing "student" but NOT "password"
results_b = collection.get(
    where_document={
        "$and": [
            {"$contains": "student"},
            {"$not_contains": "password"}
        ]
    }
)
print_results("Documents containing 'student' but not 'password':\n", results_b)


# ---------------------------------------------------------------------------
# EXERCISE 4 — Combining semantic query with metadata and text filters
# ---------------------------------------------------------------------------
# TODO: Use collection.query() to find documents semantically related to
#       "how do I print documents on campus"
#       BUT restrict results to category == "printing" using where,
#       AND only include documents that contain the word "page" using where_document
#
# Observe: how do the distances change compared to an unfiltered query?

print("\n--- EXERCISE 4: Semantic query + metadata filter + text filter ---")

# Filtered query
filtered_results = collection.query(
    query_texts=["how do I print documents on campus"],
    n_results=5,
    where={"category": "printing"},
    where_document={"$contains": "page"}
)
print_results("Semantic query + filters:\n", filtered_results, show_distances=True)

# Unfiltered query for comparison
unfiltered_results = collection.query(
    query_texts=["how do I print documents on campus"],
    n_results=5
)
print_results("Unfiltered comparison:\n", unfiltered_results, show_distances=True)