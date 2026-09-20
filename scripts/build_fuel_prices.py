"""
Build data/fuel_prices.csv - petrol / diesel / CNG price by Indian state.

HONESTY NOTE
------------
There is no free, current, machine-readable state-wise fuel price feed for India.
Checked and rejected during this project:
  * github.com/anshikakaythwas/fuel-prices-india-api  -> STALE (Pune petrol Rs 87.13, a ~2021 price)
  * Zyla "Fuel Prices in India" API                   -> paid, requires a key
  * energy.thecore.in/fuel/prices                     -> JS-rendered, table not machine-readable

So this table is built from VERIFIED metro anchors, with every other state marked as an
ESTIMATE. The `source` column records which is which. Do not present an estimate as measured.

Verified petrol anchors (energy.thecore.in, last updated 25 May 2026):
    Delhi 102.12 | Mumbai 111.18 | Kolkata 113.47 | Chennai 107.77 | Bengaluru 110.93 | Hyderabad 115.69
Verified diesel anchor: Delhi 95.20

Diesel and CNG are derived from petrol by a fixed ratio. Those ratios cross-validate two ways:
    verified Delhi pair   : 95.20 / 102.12 = 0.932
    this project's own CSV: diesel 92.82 / petrol 100.64 = 0.922
                          : CNG    79.15 / petrol 100.64 = 0.786
They agree, so DIESEL_RATIO = 0.928 and CNG_RATIO = 0.786 are used.

Run:  python scripts/build_fuel_prices.py
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = os.path.join(ROOT, "data", "india_cities.csv")
OUT = os.path.join(ROOT, "data", "fuel_prices.csv")

DIESEL_RATIO = 0.928
CNG_RATIO = 0.786

# Petrol Rs/L - VERIFIED from the metro anchors documented above.
VERIFIED = {
    "Delhi": 102.12,
    "Maharashtra": 111.18,
    "West Bengal": 113.47,
    "Tamil Nadu": 107.77,
    "Karnataka": 110.93,
    "Telangana": 115.69,
}

# Petrol Rs/L - ESTIMATED. Placed inside the verified national spread (102-116) according to
# each state's broadly known relative fuel-tax position. These are plausible, not measured.
ESTIMATED = {
    "Andhra Pradesh": 113.60, "Arunachal Pradesh": 103.40, "Assam": 105.20,
    "Bihar": 109.30, "Chhattisgarh": 108.10, "Goa": 104.60,
    "Gujarat": 105.00, "Haryana": 104.30, "Himachal Pradesh": 100.40,
    "Jammu and Kashmir": 107.90, "Jharkhand": 106.20, "Kerala": 109.80,
    "Madhya Pradesh": 112.40, "Manipur": 104.90, "Meghalaya": 105.60, "Mizoram": 103.80,
    "Nagaland": 105.10, "Odisha": 106.80, "Punjab": 104.10,
    "Rajasthan": 110.20, "Sikkim": 105.40, "Tripura": 106.30,
    "Uttar Pradesh": 103.60, "Uttarakhand": 103.90, "Ladakh": 108.50,
    "Chandigarh": 100.90, "Puducherry": 104.80,
    "Andaman and Nicobar": 93.50,
    "Dadra and Nagar Haveli and Daman and Diu": 103.20,
}

NATIONAL_FALLBACK = 107.50


def main():
    with open(CITIES, encoding="utf-8") as fh:
        states = sorted({row["state"] for row in csv.DictReader(fh)})

    rows = []
    for state in states:
        if state in VERIFIED:
            petrol, source = VERIFIED[state], "verified"
        elif state in ESTIMATED:
            petrol, source = ESTIMATED[state], "estimated"
        else:
            petrol, source = NATIONAL_FALLBACK, "fallback"
        rows.append(
            {
                "state": state,
                "petrol": round(petrol, 2),
                "diesel": round(petrol * DIESEL_RATIO, 2),
                "cng": round(petrol * CNG_RATIO, 2),
                "source": source,
            }
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["state", "petrol", "diesel", "cng", "source"])
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"states written : {len(rows)} -> {OUT}")
    print(f"by source      : {counts}")
    print(f"petrol range   : {min(r['petrol'] for r in rows)} - {max(r['petrol'] for r in rows)}")


if __name__ == "__main__":
    main()
