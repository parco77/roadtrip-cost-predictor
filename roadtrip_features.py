"""
Feature builders shared by the training scripts and the API.

WHY A SHARED MODULE
-------------------
Every model in this project is trained in scripts/ and served from app.py. If those two build
their feature vectors separately, they will drift - a column reordered in one place and not the
other produces predictions that are wrong but not obviously wrong. So each model has exactly one
function here that returns (names, values) in a fixed order, and both sides call it.

`scripts/` is excluded from the Docker image, so this module lives at the project root where the
API can import it.

Every builder returns a plain list of floats. No pandas, no sklearn - so app.py stays light and
the feature logic is readable on its own.
"""
import math

# ---------------------------------------------------------------------------- geometry

EARTH_RADIUS_KM = 6371.0088


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle (straight-line) distance in km between two points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- 1. road distance model
#
# Replaces the hardcoded `WINDING_FACTOR = 1.2355`. The winding factor is not a constant: a
# coastal route bends, a Deccan highway runs straight, a hill road switchbacks. It depends on
# WHERE the route is, so the model is given the endpoints and the shape of the line between them,
# not just its length.

DISTANCE_FEATURES = [
    "haversine_km",      # the dominant term - road distance is mostly proportional to it
    "log_haversine",     # lets the factor vary smoothly with trip length
    "start_lat", "start_lon", "dest_lat", "dest_lon",
    "mid_lat", "mid_lon",   # roughly "which part of India is this route in"
    "abs_dlat", "abs_dlon",
    "ns_share",          # 0 = purely east-west route, 1 = purely north-south
]


def distance_features(start_lat, start_lon, dest_lat, dest_lon):
    hav = haversine(start_lat, start_lon, dest_lat, dest_lon)
    abs_dlat = abs(dest_lat - start_lat)
    abs_dlon = abs(dest_lon - start_lon)
    span = abs_dlat + abs_dlon
    return [
        hav,
        math.log1p(hav),
        start_lat, start_lon, dest_lat, dest_lon,
        (start_lat + dest_lat) / 2.0, (start_lon + dest_lon) / 2.0,
        abs_dlat, abs_dlon,
        abs_dlat / span if span > 0 else 0.0,
    ]


# ------------------------------------------------------------------- 2. mileage model
#
# Replaces making the user look up their own fuel efficiency. The original CSV gives a different
# mileage range for each (vehicle, fuel) pair - nine cells in total. Additive one-hots alone can
# only fit 1 + 2 + 2 = 5 parameters, which cannot reproduce nine independent cell means, so the
# four vehicle x fuel interaction terms are included as well. With them the design matrix spans
# all nine cells exactly. This is the same "a linear model cannot multiply unless you hand it the
# product" lesson as the Week 6 fuel-bill interaction.

MILEAGE_FEATURES = [
    "vehicle_Sedan", "vehicle_SUV",        # Hatchback is the baseline
    "fuel_Diesel", "fuel_CNG",             # Petrol is the baseline
    "Sedan_x_Diesel", "Sedan_x_CNG", "SUV_x_Diesel", "SUV_x_CNG",
]


def mileage_features(vehicle, fuel):
    sedan = 1.0 if vehicle == "Sedan" else 0.0
    suv = 1.0 if vehicle == "SUV" else 0.0
    diesel = 1.0 if fuel == "Diesel" else 0.0
    cng = 1.0 if fuel == "CNG" else 0.0
    return [sedan, suv, diesel, cng,
            sedan * diesel, sedan * cng, suv * diesel, suv * cng]


# -------------------------------------------------------------------- 3. litres model
#
# Replaces `TRAFFIC_MULT = {"Low": 0.944, "Medium": 1.045, "High": 1.165}`.
#
# The quantity being predicted is  litres = (distance / mileage) * traffic_multiplier  - a
# PRODUCT of a ratio and a per-class constant. A linear model can only add, so it is handed:
#   * the ratio itself (distance / mileage), and
#   * that ratio multiplied by each traffic one-hot.
# The fitted weights on those three columns then recover the three multipliers directly.

LITRES_FEATURES = [
    "distance_km", "mileage",
    "km_per_litre_ratio",              # distance / mileage - litres before traffic
    "traffic_Medium", "traffic_High",  # Low is the baseline
    "ratio_x_Medium", "ratio_x_High",  # the interaction terms that make the product expressible
]


def litres_features(distance_km, mileage, traffic_level):
    ratio = distance_km / mileage if mileage else 0.0
    medium = 1.0 if traffic_level == "Medium" else 0.0
    high = 1.0 if traffic_level == "High" else 0.0
    return [distance_km, mileage, ratio, medium, high, ratio * medium, ratio * high]


# ---------------------------------------------------------------------- 4. toll model
#
# Replaces `TOLL_RATE = 1.301`. Toll is genuinely close to a per-km rate, so this model is
# almost as simple as the constant it replaces - but it is FITTED, it reports its own error bar,
# and it is evaluated like every other model instead of being asserted.

TOLL_FEATURES = ["distance_km", "log_distance"]


def toll_features(distance_km):
    return [distance_km, math.log1p(distance_km)]


# ------------------------------------------------------------------- 5. parking model
#
# Replaces `DEFAULT_PARKING = 70.0`. Included for completeness and expected to FAIL: in the
# generator, parking is drawn uniformly from {0, 40, 60, 80, 120, 150} independently of every
# other column, so nothing predicts it and the best possible model is the mean. Reported as a
# negative result rather than tuned until it looks respectable.

PARKING_FEATURES = ["distance_km", "vehicle_Sedan", "vehicle_SUV", "passengers"]


def parking_features(distance_km, vehicle, passengers):
    return [distance_km,
            1.0 if vehicle == "Sedan" else 0.0,
            1.0 if vehicle == "SUV" else 0.0,
            float(passengers)]


# ------------------------------------------------------------------ 6. traffic model
#
# Departure hour is CYCLICAL: 23:00 and 00:00 are one hour apart, not twenty-three. Feeding the
# raw integer would make the model treat them as maximally distant, so hour and month go in as
# sin/cos pairs. `is_peak` gives a linear model the rush-hour block directly, since sin/cos alone
# cannot express "high between 8-11 and again between 17-21".

TRAFFIC_FEATURES = ["hour_sin", "hour_cos", "is_peak", "month_sin", "month_cos"]


def traffic_features(departure_hour, month):
    return [
        math.sin(2 * math.pi * departure_hour / 24),
        math.cos(2 * math.pi * departure_hour / 24),
        1.0 if (8 <= departure_hour <= 11 or 17 <= departure_hour <= 21) else 0.0,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
    ]


# --------------------------------------------------------------- 7. cost regressor
#
# Unchanged from the Week 6 model that this project already ships, so the chained pipeline is a
# fair swap: same target, same features, same closed-form linear fit. The only difference is
# where `toll_cost`, `parking_cost` and `fuel_consumption_litres` come from - sub-model
# predictions instead of arithmetic constants.

COST_FEATURES = [
    "distance_km", "mileage", "fuel_price", "toll_cost", "parking_cost",
    "fuel_consumption_litres",
    "fuel_bill",                             # litres x price - the Week 6 interaction term
    "vehicle_type_SUV", "vehicle_type_Sedan",   # Hatchback is the baseline
]


def cost_features(distance_km, mileage, fuel_price, toll_cost, parking_cost, litres, vehicle):
    return [distance_km, mileage, fuel_price, toll_cost, parking_cost, litres,
            litres * fuel_price,
            1.0 if vehicle == "SUV" else 0.0,
            1.0 if vehicle == "Sedan" else 0.0]


# ------------------------------------------------------- 8. end-to-end (honesty) model
#
# The benchmark model: fed ONLY what a traveller could actually know, with nothing derived by
# arithmetic first. No distance, no toll, no litres. Its error against the cost regressor above
# measures how much of that model's R2 was arithmetic being handed back to it.
#
# The last two columns are the Week 6 lesson applied again. The true cost contains
# (distance / mileage) * fuel_price, so a purely additive model is structurally incapable of
# fitting it. Both variants are trained and compared in scripts/train_end_to_end.py.

END_TO_END_BASE = [
    "haversine_km", "log_haversine",
    "start_lat", "start_lon", "dest_lat", "dest_lon", "mid_lat", "mid_lon",
    "abs_dlat", "abs_dlon",
    "mileage", "fuel_price", "passengers",
    "hour_sin", "hour_cos", "is_peak", "month_sin", "month_cos",
    "vehicle_Sedan", "vehicle_SUV", "fuel_Diesel", "fuel_CNG",
]
END_TO_END_INTERACTIONS = ["hav_over_mileage", "hav_over_mileage_x_price"]
END_TO_END_FEATURES = END_TO_END_BASE + END_TO_END_INTERACTIONS


def end_to_end_features(start_lat, start_lon, dest_lat, dest_lon, mileage, fuel_price,
                        passengers, departure_hour, month, vehicle, fuel,
                        interactions=True):
    hav = haversine(start_lat, start_lon, dest_lat, dest_lon)
    abs_dlat = abs(dest_lat - start_lat)
    abs_dlon = abs(dest_lon - start_lon)
    row = [
        hav, math.log1p(hav),
        start_lat, start_lon, dest_lat, dest_lon,
        (start_lat + dest_lat) / 2.0, (start_lon + dest_lon) / 2.0,
        abs_dlat, abs_dlon,
        mileage, fuel_price, float(passengers),
        math.sin(2 * math.pi * departure_hour / 24),
        math.cos(2 * math.pi * departure_hour / 24),
        1.0 if (8 <= departure_hour <= 11 or 17 <= departure_hour <= 21) else 0.0,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        1.0 if vehicle == "Sedan" else 0.0,
        1.0 if vehicle == "SUV" else 0.0,
        1.0 if fuel == "Diesel" else 0.0,
        1.0 if fuel == "CNG" else 0.0,
    ]
    if interactions:
        ratio = hav / mileage if mileage else 0.0
        row += [ratio, ratio * fuel_price]
    return row


# ------------------------------------------------------------------------- city lookup

def load_city_index(path):
    """city name -> (state, lat, lon). Plain csv so app.py need not import pandas for this."""
    import csv
    index = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index[row["city"]] = (row["state"], float(row["lat"]), float(row["lon"]))
    return index
