import argparse
import csv
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BANK_TRANSACTIONS_CSV, OPENAI_API_KEY, get_env, require_env
from app.services.llm_service import generate
from app.services.rag_service import retrieve

# Optional: use OpenAI for classification (returns valid JSON so some items can be marked deductible).
def _openai_available() -> bool:
    return bool(OPENAI_API_KEY)

DEFAULT_CSV = BANK_TRANSACTIONS_CSV
BATCH_SIZE = 15

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a South African tax deduction analyst. You will be given:
    1. A batch of bank transactions (date, description, amount, type).
    2. Relevant excerpts from the SA Income Tax Act No. 58 of 1962 (za-act-1962-58-publication-document).
    3. You are supposed to help a south african freelancer and business owner to understand if their transactions are deductible.

    Your job is to be **helpfully liberal but still grounded in the Act**:
    - Identify plausible deductions and potential exemptions where the Act supports them.
    - Where the Act is not crystal clear but suggests a strong possibility, use "maybe" and explain.
    - Avoid saying everything is "not deductible" unless the Act clearly excludes it or provides no support.
    - Avoid saying everything is "deductible" unless the Act clearly allows it or provides strong support.
    - You are not allowed to say anything that is not supported by the Act.
    - Do not classify transactions that are not deductible as "maybe" or "deductible".
    - If the transaction is not deductible, explain why is it not deductible.
    - Always note that only business expenses are deductible.
    - Personal expenses are not deductible (e.g. movies, restaurants, groceries, general online shopping). Only a few transactions will typically be deductible; mark as true or maybe where the Act or context supports it.

    Use this guidance:
    - Set "deductible": true when the excerpts contain a specific section that clearly allows
      this type of expense (e.g. a cited section and wording that permits the deduction).
    - Set "deductible": maybe when:
        * the excerpts mention a relevant section or principle, AND
        * whether it applies depends on facts you cannot see (e.g. proportion of business use,
          whether the taxpayer is a freelancer, whether it is incurred in the production of income).
      In this case, briefly explain the conditions under which it would be deductible.
    - Set "deductible": false when:
        * the excerpts clearly disallow this type of expense, OR
        * there is no reasonable basis in the excerpts to treat it as incurred in the production of income.

    ALWAYS:
    - Base your answers primarily on the Act excerpts, but you may use reasonable tax intuition
      to classify expenses as likely business vs personal (e.g. "Accountant fee" vs "Movie tickets").
    - Prefer "maybe" over "false" when there is any plausible path to deduction under the Act
      (e.g. business-related bank fees, professional fees, medical, donations) even if key facts are missing.
    - Only a few items should be "true"; use "maybe" liberally for anything that could qualify under s11(a), s18, or similar, so the user can review and keep only the ones they want to claim.

    For EACH transaction output:
    - "deductible": true / false / maybe (using the criteria above).
    - "category": short label, e.g. "Medical (s18)", "Home office (s11(a))",
      "Not deductible - clearly personal", "Maybe - mixed use (vehicle expenses)".
    - "section": the Act section from the excerpts that applies, or "None" if you are using
      a general principle or cannot tie it to a specific section.
    - "reason": 1-3 sentences explaining why you chose true / false / maybe.
      If not deductible, explain briefly why (e.g. appears personal, no link to income, or Act excludes it).

    Return ONLY valid JSON -- an array of objects with keys:
    ["transaction_index", "date", "description", "amount", "deductible", "category", "section", "reason"]

    Do NOT wrap in markdown fences. Do NOT add extra text.
""")


def load_transactions(csv_path: Path) -> list[dict]:
    """Load and normalise the bank transactions CSV."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({
                "index": i,
                "date": row.get("date", ""),
                "description": " | ".join(filter(None, [
                    row.get("description_line_1", ""),
                    row.get("description_line_2", ""),
                    row.get("description_line_3", ""),
                ])),
                "amount": row.get("amount", ""),
                "type": row.get("transaction_type", ""),
            })
    return rows


def _format_batch(transactions: list[dict]) -> str:
    lines = []
    for t in transactions:
        lines.append(
            f"[{t['index']}] {t['date']}  {t['description']}  "
            f"R{t['amount']}  ({t['type']})"
        )
    return "\n".join(lines)


def _extract_json(raw: str) -> list[dict] | None:
    """Parse JSON array from model reply. Returns None if empty or invalid."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    text = text.strip()
    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        text = json_match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _is_spend(t: dict) -> bool:
    """Return True if this transaction is money spent (potentially deductible).

    We treat:
    - Negative amounts as spend.
    - Positive amounts with types like salary, eft_credit, cash_deposit, interest
      as income (not spend).
    """
    amount = _safe_float(t.get("amount"))
    ttype = (t.get("type") or "").strip().lower()
    if amount < 0:
        return True
    income_types = {
        "salary",
        "eft_credit",
        "cash_deposit",
        "interest",
    }
    if ttype in income_types:
        return False
    # Default: non-negative amount but not clearly an income type -> treat as not spend
    return False


def scan_batch(transactions: list[dict]) -> list[dict]:
    """Classify one batch of transactions using RAG context and local LLM.

    Only money spent (outflows) are sent to the model for deduction analysis.
    Income / inflows (e.g. salary, deposits) are marked as not deductible and
    are not sent to the model.
    """
    spend_tx = [t for t in transactions if _is_spend(t)]
    income_tx = [t for t in transactions if not _is_spend(t)]

    results: list[dict] = []

    # Pre-classify income / inflows: not deductible by definition (they are not expenses).
    for t in income_tx:
        results.append(
            {
                "transaction_index": t["index"],
                "date": t["date"],
                "description": t["description"],
                "amount": t["amount"],
                "deductible": False,
                "category": "Income item (e.g. salary / deposit)",
                "section": "None",
                "reason": "This is income received, not an expense; it cannot be claimed as a deduction.",
            }
        )

    if not spend_tx:
        return results

    descriptions = " ".join(t["description"] for t in spend_tx)
    search_query = (
        "South African tax deductions for: medical aid, insurance premiums, "
        "bank fees, business expenses, donations, retirement fund contributions. "
        + descriptions[:500]
    )
    context_chunks = retrieve(search_query, n_results=10)
    context_block = "\n\n---\n\n".join(
        f"[LAW EXCERPT {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    batch_text = _format_batch(spend_tx)
    user_content = (
        f"RELEVANT LAW EXCERPTS:\n{context_block}\n\n"
        f"TRANSACTIONS TO CLASSIFY:\n{batch_text}"
    )

    reply: str
    if _openai_available():
        from openai import OpenAI
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=get_env("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # API may return {"transactions": [...]} or a bare array
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                reply = raw
            elif isinstance(obj, dict) and "transactions" in obj:
                reply = json.dumps(obj["transactions"])
            else:
                reply = raw
        except (json.JSONDecodeError, TypeError):
            reply = raw
    else:
        reply = generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.1,
        )

    parsed = _extract_json(reply)
    if parsed is not None:
        # Model classifications are trusted for spend transactions.
        results.extend(parsed)
        return results

    # Model returned empty or invalid JSON; mark spend items as undetermined.
    print("  Warning: model output was not valid JSON; marking spend transactions as undetermined (manual review needed).")
    for t in spend_tx:
        results.append(
            {
                "transaction_index": t["index"],
                "date": t["date"],
                "description": t["description"],
                "amount": t["amount"],
                "deductible": "undetermined",
                "category": "Undetermined - model output invalid",
                "section": "None",
                "reason": "The LLM did not return valid JSON for this batch; classification for this expense requires manual review.",
            }
        )
    return results


def scan_all(csv_path: Path) -> list[dict]:
    """Scan every transaction in the CSV and return classification results."""
    transactions = load_transactions(csv_path)
    print(f"Loaded {len(transactions)} transactions from {csv_path.name}")
    if _openai_available():
        print("Using OpenAI for classification (OPENAI_API_KEY set).")
    else:
        print("Using local LLM for classification (set OPENAI_API_KEY for reliable JSON and more deductibles).")

    all_results: list[dict] = []

    for i in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[i : i + BATCH_SIZE]
        print(f"  Scanning transactions {i}..{i + len(batch) - 1} ...")
        results = scan_batch(batch)
        all_results.extend(results)

    return all_results


def _normalise_deductible(value) -> str | bool:
    if value is True or value == "true" or value == "True":
        return True
    if value is False or value == "false" or value == "False":
        return False
    return value


def print_report(results: list[dict]) -> None:
    """Pretty-print the deduction scan results."""
    for r in results:
        r["deductible"] = _normalise_deductible(r.get("deductible"))
    deductible = [r for r in results if r.get("deductible") is True]
    maybe = [r for r in results if r.get("deductible") == "maybe"]
    undetermined = [r for r in results if r.get("deductible") == "undetermined"]
    not_ded = [
        r
        for r in results
        if r.get("deductible") is False
        and r.get("deductible") not in ("maybe", "undetermined")
    ]

    def safe_float(x):
        try:
            return abs(float(x))
        except (TypeError, ValueError):
            return 0.0

    total_deductible = sum(safe_float(r.get("amount")) for r in deductible)
    total_maybe = sum(safe_float(r.get("amount")) for r in maybe)

    print("\n" + "=" * 70)
    print("  SA TAX DEDUCTION SCAN REPORT")
    print("=" * 70)

    print(f"\nTotal transactions scanned: {len(results)}")
    print(f"Definitely deductible:     {len(deductible)}")
    print(f"Possibly deductible:       {len(maybe)}")
    print(f"Not deductible:            {len(not_ded)}")
    print(f"Undetermined (LLM invalid): {len(undetermined)}")
    print(f"\nEstimated deductible amount:  R {total_deductible:,.2f}")
    print(f"Possible additional amount:  R {total_maybe:,.2f}")

    if deductible:
        print("\n--- DEDUCTIBLE ITEMS ---")
        for r in deductible:
            print(f"  [{r.get('date', '?')}] {r.get('description', '?')}")
            print(f"    Amount: R {r.get('amount', '?')}  |  "
                  f"Category: {r.get('category', '?')}  |  "
                  f"Section: {r.get('section', '?')}")
            print(f"    Reason: {r.get('reason', '?')}")

    if maybe:
        print("\n--- POSSIBLY DEDUCTIBLE (needs more info) ---")
        for r in maybe:
            print(f"  [{r.get('date', '?')}] {r.get('description', '?')}")
            print(f"    Amount: R {r.get('amount', '?')}  |  "
                  f"Category: {r.get('category', '?')}  |  "
                  f"Section: {r.get('section', '?')}")
            print(f"    Reason: {r.get('reason', '?')}")

    if undetermined:
        print("\n--- UNDETERMINED (LLM output invalid) ---")
        for r in undetermined:
            print(f"  [{r.get('date', '?')}] {r.get('description', '?')}")
            print(f"    Amount: R {r.get('amount', '?')}  |  "
                  f"Category: {r.get('category', '?')}  |  "
                  f"Section: {r.get('section', '?')}")
            print(f"    Reason: {r.get('reason', '?')}")

    print("\n" + "-" * 70)
    print("DISCLAIMER: This analysis is for informational purposes only and")
    print("does not constitute professional tax advice. Consult a registered")
    print("South African tax practitioner for your specific circumstances.")
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan bank transactions for SA tax deductions. "
        "Uses OpenAI if OPENAI_API_KEY is set (recommended for valid JSON); otherwise uses local LLM (HF_MODEL)."
    )
    parser.add_argument(
        "-f", "--file", type=Path, default=DEFAULT_CSV,
        help="Path to the bank transactions CSV file.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Save full results as JSON to this path.",
    )
    args = parser.parse_args()

    results = scan_all(args.file)
    print_report(results)

    if args.output:
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
