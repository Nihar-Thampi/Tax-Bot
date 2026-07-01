import csv
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from env_config import get_env, require_env

OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4.1-mini")

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    return _openai_client

OUTPUT_CSV = Path(__file__).with_name("bank_transactions_2025.csv")
MAX_RETRIES = 3

MONTHS = [
    ("January",   "2025-01-01", "2025-01-31"),
    ("February",  "2025-02-01", "2025-02-28"),
    ("March",     "2025-03-01", "2025-03-31"),
    ("April",     "2025-04-01", "2025-04-30"),
    ("May",       "2025-05-01", "2025-05-31"),
    ("June",      "2025-06-01", "2025-06-30"),
    ("July",      "2025-07-01", "2025-07-31"),
    ("August",    "2025-08-01", "2025-08-31"),
    ("September", "2025-09-01", "2025-09-30"),
    ("October",   "2025-10-01", "2025-10-31"),
    ("November",  "2025-11-01", "2025-11-30"),
    ("December",  "2025-12-01", "2025-12-31"),
]

CSV_COLUMNS = [
    "date",
    "description_line_1",
    "description_line_2",
    "description_line_3",
    "amount",
    "balance",
    "transaction_type",
]

PROMPT_TEMPLATE = """\
Generate realistic South African bank transactions for the month of {month} \
({start_date} to {end_date}).

Requirements:
- Include between 25 and 40 transactions spread across the month.
- Each transaction MUST have exactly 3 separate description lines:
    * description_line_1: The payee or merchant name.
    * description_line_2: A short note about what the transaction was for.
    * description_line_3: Reference number or additional detail.
- "amount" is a float. Use negative values for debits and positive for credits.
- "balance" is the running balance after that transaction. Start the month with a \
balance of {opening_balance:.2f} ZAR.
- "transaction_type" is one of: "debit_order", "card_purchase", "eft_credit", \
"eft_debit", "cash_withdrawal", "cash_deposit", "salary", "interest", "bank_fee", \
"transfer".
- "date" must be in YYYY-MM-DD format and fall within the specified range.

You MUST return ONLY a valid JSON object with:
- a single top-level key \"transactions\" (an array)
- each element of \"transactions\" having exactly these keys:
  - \"date\" (YYYY-MM-DD string)
  - \"description_line_1\" (string)
  - \"description_line_2\" (string)
  - \"description_line_3\" (string)
  - \"amount\" (number)
  - \"balance\" (number)
  - \"transaction_type\" (one of the allowed strings above)

- Do NOT add comments in the JSON.
- Do NOT add any extra keys.
- Do NOT wrap the JSON in markdown code fences or add any extra text.
- Provide at least 3-4 tax deductable transactions for freelancers and business owners in South Africa per month.
"""


def extract_json(raw_text: str) -> list[dict]:
    """Parse JSON from the model reply, supporting both arrays and {'transactions': [...]}."""
    text = (raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty reply from model", "", 0)

    def _postprocess(data: object) -> list[dict]:
        if isinstance(data, dict) and "transactions" in data:
            tx = data["transactions"]
            if not isinstance(tx, list):
                raise ValueError('"transactions" must be a list.')
            return tx
        if isinstance(data, list):
            return data
        raise ValueError("Model reply JSON is neither a list nor an object with 'transactions'.")

    # First try direct JSON (used with response_format='json_object').
    try:
        return _postprocess(json.loads(text))
    except json.JSONDecodeError:
        # Fall back to repair heuristic below.
        pass

    # Heuristic: strip fences, grab between first '{' and last '}', fix trailing commas.
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
    else:
        # Fallback to first array if no object found.
        json_match = re.search(r"\[[\s\S]*\]", text)
        if not json_match:
            raise json.JSONDecodeError("Could not locate JSON object or array in model reply", text, 0)
        candidate = json_match.group(0)

    # Remove trailing commas before ']' or '}' (common LLM mistake).
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    data = json.loads(candidate)
    return _postprocess(data)


def generate_month(month: str, start: str, end: str,
                   opening_balance: float) -> list[dict]:
    """Use local LLM to generate one month's transactions and return parsed rows."""
    prompt = PROMPT_TEMPLATE.format(
        month=month,
        start_date=start,
        end_date=end,
        opening_balance=opening_balance,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate realistic South African bank transaction data "
                            "in strict JSON format only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            print(f"\nRAW OpenAI reply for {month} (first 800 chars):\n{raw[:800]}\n")
            rows = extract_json(raw)
            print(f"  -> received {len(rows)} transactions")
            return rows
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  JSON parse error on attempt {attempt}: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)

    raise RuntimeError(f"All retries exhausted for {month}")


def main() -> None:
    all_rows: list[dict] = []
    opening_balance = 15_420.75

    for month_name, start, end in MONTHS:
        print(f"Requesting {month_name} 2025 ...")
        rows = generate_month(month_name, start, end, opening_balance)
        all_rows.extend(rows)

        if rows:
            last_balance = rows[-1].get("balance", opening_balance)
            try:
                opening_balance = float(last_balance)
            except (TypeError, ValueError):
                pass

        time.sleep(0.5)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            filtered = {k: row.get(k, "") for k in CSV_COLUMNS}
            writer.writerow(filtered)

    print(f"\nDone -- {len(all_rows)} transactions saved to {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
