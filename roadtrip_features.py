"""
Feature builders shared by the training scripts and the API.

WHY A SHARED MODULE
-------------------
Every model in this project is trained in scripts/ and served from app.py. If those two build
their feature vectors separately, they will drift - a column reordered in one place and not the
other produces predictions that are wrong but not obviously wrong. So each model has exactly one
function here that returns values in a fixed order, and both sides call it.

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
# Powers the distance lookup when a city pair is not in the shipped reference table.
#
# Road distance is not the straight-line distance times some fixed number. A coastal route
# bends, a Deccan highway runs straight, a hill road switchbacks - Surat and Bhavnagar sit 94 km
# apart across the Gulf of Khambhat and 339 km apart by road. The ratio depends on WHERE the
# route is, so the model is given the endpoints and the shape of the line between them, not just
# its length.
#
# It predicts the RATIO and multiplies by haversine afterwards, rather than predicting kilometres
# directly. That keeps the target scale-free, so the model spends its capacity on the part that
# is genuinely unknown - the shape of the route - instead of re-learning "longer straight line,
# longer road".

DISTANCE_FEATURES = [
    "haversine_km",      # the dominant term - road distance is mostly proportional to it
    "log_haversine",     # lets the ratio vary smoothly with trip length
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
# Gives each vehicle its own fuel efficiency, which is what makes pricing all three at once
# mean anything: a hatchback and an SUV on the same route do not burn the same fuel, and the
# difference between them is mostly mileage, not wear.
#
# The data gives a different mileage range for each (vehicle, fuel) pair - nine cells in total.
# Additive one-hots alone can only fit 1 + 2 + 2 = 5 parameters, which cannot reproduce nine
# independent cell means, so the four vehicle x fuel interaction terms are included as well.
# With them the design matrix spans all nine cells exactly. Same "a linear model cannot multiply
# unless you hand it the product" idea as the fuel-bill term in the cost model below.

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
# Fuel burnt over the trip. The dominant term is the ratio distance / mileage, which is why it
# is handed over directly rather than left for the model to discover from its two parts - a
# linear model cannot divide.
#
# Real fuel burn is not exactly that ratio: the same car on the same road burns more in dense
# traffic than in clear. Nothing the user tells us says which they will meet, so that variation
# is unobserved. The fitted slope on the ratio therefore comes out slightly above 1.0 - it
# settles at the average conditions in the data rather than at the textbook figure - and the
# spread around it is irreducible error that scales with trip length. Both are measured in
# scripts/train_pipeline.py rather than assumed.

LITRES_FEATURES = ["distance_km", "mileage", "km_per_litre_ratio"]


def litres_features(distance_km, mileage):
    ratio = distance_km / mileage if mileage else 0.0
    return [distance_km, mileage, ratio]


# ---------------------------------------------------------------------- 4. toll model
#
# Toll is close to a per-km rate, so this model is nearly as simple as a rate would be - but it
# is FITTED, it reports its own error bar, and it is evaluated like every other model instead of
# being asserted. The log term lets the effective rate drift with trip length.

TOLL_FEATURES = ["distance_km", "log_distance"]


def toll_features(distance_km):
    return [distance_km, math.log1p(distance_km)]


# ----------------------------------------------------------------- 5. parking (no model)
#
# Parking is a user input, and these features exist to show why. Six model families are fitted
# against them in scripts/train_pipeline.py and every one scores test R2 <= 0 - worse than
# predicting the mean - because parking in this data carries no relationship to distance,
# vehicle or party size at all. A quantity nothing predicts should be asked for, not guessed,
# and the negative result is reported rather than tuned away.

PARKING_FEATURES = ["distance_km", "vehicle_Sedan", "vehicle_SUV", "passengers"]


def parking_features(distance_km, vehicle, passengers):
    return [distance_km,
            1.0 if vehicle == "Sedan" else 0.0,
            1.0 if vehicle == "SUV" else 0.0,
            float(passengers)]


# --------------------------------------------------------------- 6. cost regressor
#
# The final stage. Consumes the sub-model outputs above plus what the user typed, and is trained
# on PREDICTED components rather than true ones so that training and serving see the same kind
# of input (see scripts/train_pipeline.py).
#
# Two groups of terms are here because a purely additive model cannot express them:
#
#   fuel_bill = litres x price. The cost contains that product; given only the two factors, an
#   additive model would have to approximate it.
#
#   distance_x_SUV / distance_x_Sedan. Running costs differ between vehicles PER KILOMETRE, not
#   by a flat amount per trip. With the plain one-hots alone the fitted SUV premium is a
#   constant - about Rs 130 whether the trip is 200 km or 1200 km - when the real gap runs from
#   roughly Rs 50 to Rs 320 across that range. Since this model prices all three vehicles side
#   by side, that gap IS the output, so the interaction is load-bearing rather than decorative.

COST_FEATURES = [
    "distance_km", "mileage", "fuel_price", "toll_cost", "parking_cost",
    "fuel_consumption_litres",
    "fuel_bill",                                 # litres x price
    "vehicle_type_SUV", "vehicle_type_Sedan",    # Hatchback is the baseline
    "distance_x_SUV", "distance_x_Sedan",        # per-km running-cost difference
]


def cost_features(distance_km, mileage, fuel_price, toll_cost, parking_cost, litres, vehicle):
    suv = 1.0 if vehicle == "SUV" else 0.0
    sedan = 1.0 if vehicle == "Sedan" else 0.0
    return [distance_km, mileage, fuel_price, toll_cost, parking_cost, litres,
            litres * fuel_price,
            suv, sedan,
            distance_km * suv, distance_km * sedan]


# ------------------------------------------------------------------------- city lookup

def load_city_index(path):
    """city name -> (state, lat, lon). Plain csv so app.py need not import pandas for this."""
    import csv
    index = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index[row["city"]] = (row["state"], float(row["lat"]), float(row["lon"]))
    return index
