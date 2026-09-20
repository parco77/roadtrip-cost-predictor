"""
API tests for the Road Trip Cost Predictor.

Run:  pytest -q

These deliberately cover the things that broke during development rather than trivially
exercising every endpoint: the derived-feature arithmetic, the error paths, and the honesty
warnings — which are easy to silently drop and hard to notice missing.

Network note: /api/predict prefers live OSRM routing and falls back to a calibrated haversine.
Tests assert on behaviour that holds either way, so they pass offline.
"""
import math

import pytest
from fastapi.testclient import TestClient

import app as api

client = TestClient(api.app)

# From the recovered cost formula — see README "Results" and notebook Week 4.
TRAFFIC_MULTIPLIER = {"Low": 0.944, "Medium": 1.045, "High": 1.165}
MAINTENANCE = {"Hatchback": 0.5694, "Sedan": 0.6943, "SUV": 0.8366}
TOLL_RATE = 1.301


def predict(**overrides):
    body = {"start_city": "Jaipur", "destination_city": "Agra", "mileage": 15.5}
    body.update(overrides)
    return client.post("/api/predict", json=body)


# --------------------------------------------------------------------------- health
def test_health_reports_the_in_domain_city_count():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["cities"] == 537


# --------------------------------------------------------------------------- city search
def test_city_search_prefix_matches_rank_first():
    results = client.get("/api/cities", params={"q": "jai"}).json()["results"]
    assert results[0]["city"] == "Jaipur"


def test_city_search_matches_on_state_too():
    results = client.get("/api/cities", params={"q": "kerala", "limit": 5}).json()["results"]
    assert results and all(c["state"] == "Kerala" for c in results)


def test_city_search_empty_query_returns_largest_cities():
    results = client.get("/api/cities", params={"q": "", "limit": 3}).json()["results"]
    assert [c["city"] for c in results] == ["Mumbai", "Delhi", "Bengaluru"]


def test_city_search_respects_limit():
    assert len(client.get("/api/cities", params={"q": "a", "limit": 4}).json()["results"]) == 4


# --------------------------------------------------------------------------- errors
def test_same_city_is_rejected():
    r = predict(destination_city="Jaipur")
    assert r.status_code == 400


def test_unknown_city_is_404_not_a_silent_guess():
    r = predict(destination_city="Atlantis")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_city_outside_the_537_domain_is_rejected():
    """Small towns exist in india_cities.csv but the model never saw them."""
    r = predict(destination_city="Chikodi")
    assert r.status_code == 404


@pytest.mark.parametrize("field,value", [
    ("mileage", 999),          # above the trained range
    ("mileage", 1),            # below it
    ("departure_hour", 24),    # hours are 0-23
    ("month", 13),
    ("passengers", 0),
    ("vehicle_type", "Truck"),  # not one of the three classes
    ("fuel_type", "Hydrogen"),
])
def test_out_of_range_fields_are_422(field, value):
    assert predict(**{field: value}).status_code == 422


# --------------------------------------------------------------------------- derived values
def test_litres_are_derived_not_requested():
    """The strongest feature must never be asked for — it is computed from distance,
    mileage and traffic. This is the whole reason the product is usable."""
    body = predict(traffic_level="Medium").json()
    distance = body["route"]["distance_km"]
    expected = (distance / 15.5) * TRAFFIC_MULTIPLIER["Medium"]
    assert body["derived"]["fuel_consumption_litres"] == pytest.approx(expected, rel=1e-3)


def test_derived_explanation_string_is_renderable():
    """The UI prints this verbatim, so it must contain the actual arithmetic."""
    body = predict(traffic_level="High").json()
    text = body["derived"]["fuel_consumption_explained"]
    assert "km/l" in text and "1.165" in text and "high traffic" in text


def test_toll_tracks_the_recovered_rate_without_being_hardcoded_to_it():
    """Toll used to be `distance * 1.301`. It is now sub-model 4, so the assertion changed.

    The old test demanded the answer match the constant to 0.1%. That is exactly what a
    FITTED model should not be required to do: it carries an intercept and a log-distance
    term, so its implied per-km rate is close to 1.301 but deliberately not identical. What
    must still hold is that it lands in the right neighbourhood - a model that had drifted
    off the recovered rate would be broken.
    """
    body = predict().json()
    distance = body["route"]["distance_km"]
    toll = body["derived"]["toll_cost"]
    implied_rate = toll / distance
    assert implied_rate == pytest.approx(TOLL_RATE, rel=0.05), (
        f"implied {implied_rate:.4f} Rs/km vs recovered {TOLL_RATE}")


def test_toll_is_a_model_and_not_a_single_multiplication():
    """Proof the constant is really gone: a fitted model's implied rate moves with distance."""
    short = predict(start_city="Palwal", destination_city="New Delhi").json()
    long_ = predict(start_city="Jaipur", destination_city="Kolkata").json()
    rates = []
    for body in (short, long_):
        rates.append(body["derived"]["toll_cost"] / body["route"]["distance_km"])
    assert rates[0] != pytest.approx(rates[1], rel=1e-6), (
        "identical implied rates would mean toll is still distance x a constant")
    for r in rates:
        assert 0.9 * TOLL_RATE < r < 1.15 * TOLL_RATE


def test_traffic_is_inferred_when_omitted_and_respected_when_given():
    inferred = predict().json()["derived"]
    assert inferred["traffic_inferred"] is True

    chosen = predict(traffic_level="Low").json()["derived"]
    assert chosen["traffic_inferred"] is False
    assert chosen["traffic_level"] == "Low"


def test_night_departure_is_cheaper_than_rush_hour():
    """Rush-hour traffic burns more fuel, so the classifier should make 3am cheaper than 9am."""
    night = predict(departure_hour=3).json()["prediction"]["total_trip_cost"]
    rush = predict(departure_hour=9).json()["prediction"]["total_trip_cost"]
    assert night < rush


# --------------------------------------------------------------------------- cost breakdown
def test_breakdown_sums_to_the_total():
    body = predict().json()["prediction"]
    assert sum(body["breakdown"].values()) == pytest.approx(body["total_trip_cost"], abs=1.0)


def test_wear_and_tear_orders_correctly_by_vehicle_and_stays_plausible():
    """Wear must rise Hatchback < Sedan < SUV, and land in the right neighbourhood.

    Not an exact-match assertion, deliberately. The API derives this component by subtraction
    (total - fuel - toll - parking), so it absorbs whatever error the regression made; and the
    training data carries +/-20% noise on the per-km rate. Demanding it equal distance x rate
    would be asserting something the design never promised. What IS guaranteed is the ordering
    and a sane magnitude.
    """
    wear = {}
    for vehicle in ("Hatchback", "Sedan", "SUV"):
        body = predict(vehicle_type=vehicle, mileage=15.5, traffic_level="Medium").json()
        wear[vehicle] = body["prediction"]["breakdown"]["wear_and_tear"]
        distance = body["route"]["distance_km"]

    assert wear["Hatchback"] < wear["Sedan"] < wear["SUV"]
    for vehicle, value in wear.items():
        expected = distance * MAINTENANCE[vehicle]
        assert 0.5 * expected < value < 1.5 * expected, f"{vehicle}: {value} vs ~{expected:.0f}"


def test_vehicle_type_actually_changes_the_prediction():
    """Regression test. vehicle_type was collected by the UI but missing from the regression
    features, so all three vehicles returned an identical total on the same route."""
    totals = {
        v: predict(vehicle_type=v, mileage=15.5, traffic_level="Medium")
        .json()["prediction"]["total_trip_cost"]
        for v in ("Hatchback", "Sedan", "SUV")
    }
    assert totals["Hatchback"] < totals["Sedan"] < totals["SUV"]
    assert totals["SUV"] - totals["Hatchback"] > 50, "vehicle choice should move the needle"


def test_cost_band_is_one_of_the_four_classes():
    band = predict().json()["prediction"]["cost_band"]
    assert band in {"Excellent", "Good", "Average", "Poor"}


# --------------------------------------------------------------------------- honesty warnings
def test_long_route_warns_that_it_is_extrapolating():
    body = predict(start_city="Srinagar", destination_city="Chennai").json()
    assert body["route"]["distance_km"] > 1500
    assert any("outside the model's training range" in w for w in body["warnings"])


def test_estimated_fuel_price_is_disclosed():
    """29 of 35 states have estimated prices; the user must be told which."""
    body = predict(start_city="Srinagar", destination_city="Chennai").json()
    assert body["inputs"]["fuel_price_source"] != "verified"
    assert any("estimate, not a measured price" in w for w in body["warnings"])


def test_verified_fuel_price_produces_no_price_warning():
    body = predict(start_city="Delhi", destination_city="Jaipur").json()
    assert body["inputs"]["fuel_price_source"] == "verified"
    assert not any("fuel price" in w.lower() for w in body["warnings"])


def test_in_range_route_with_verified_price_has_no_warnings_at_all():
    body = predict(start_city="Delhi", destination_city="Jaipur").json()
    assert body["warnings"] == [] or all("Routing service" in w for w in body["warnings"])


# --------------------------------------------------------------------------- fuel price input
def test_city_payload_carries_the_state_fuel_prices():
    """The form prefills from this, so it must arrive with the city — not via a second request."""
    city = client.get("/api/cities", params={"q": "Jaipur", "limit": 1}).json()["results"][0]
    prices = city["fuel_prices"]
    assert set(prices) == {"petrol", "diesel", "cng", "source"}
    assert prices["petrol"] > prices["diesel"] > prices["cng"]


def test_user_fuel_price_overrides_the_table():
    body = predict(fuel_price=120.0).json()
    assert body["inputs"]["fuel_price"] == 120.0
    assert body["inputs"]["fuel_price_source"] == "user"


def test_user_fuel_price_changes_the_estimate():
    cheap = predict(fuel_price=90.0).json()["prediction"]["total_trip_cost"]
    dear = predict(fuel_price=120.0).json()["prediction"]["total_trip_cost"]
    assert dear > cheap


def test_user_fuel_price_suppresses_the_estimate_warning():
    """A price the user typed is not an estimate — warning about it would be wrong."""
    estimated = predict().json()
    assert any("estimate" in w for w in estimated["warnings"])

    supplied = predict(fuel_price=105.0).json()
    assert not any("fuel price" in w.lower() for w in supplied["warnings"])


@pytest.mark.parametrize("bad", [5, 25, 500, 1000])
def test_absurd_fuel_prices_are_rejected(bad):
    assert predict(fuel_price=bad).status_code == 422


def test_omitting_fuel_price_still_uses_the_table():
    body = predict().json()
    assert body["inputs"]["fuel_price_source"] in {"verified", "estimated", "fallback"}


def test_departure_sweep_honours_the_user_fuel_price():
    base = client.post("/api/departure-sweep", json={
        "start_city": "Jaipur", "destination_city": "Agra", "mileage": 15.5}).json()
    dear = client.post("/api/departure-sweep", json={
        "start_city": "Jaipur", "destination_city": "Agra", "mileage": 15.5,
        "fuel_price": 140.0}).json()
    assert dear["cheapest"]["total_trip_cost"] > base["cheapest"]["total_trip_cost"]


# --------------------------------------------------------------------------- departure sweep
def sweep(**overrides):
    body = {"start_city": "Jaipur", "destination_city": "Agra",
            "mileage": 15.5, "vehicle_type": "Sedan", "month": 6}
    body.update(overrides)
    return client.post("/api/departure-sweep", json=body)


def test_sweep_covers_every_hour_exactly_once():
    hours = sweep().json()["hours"]
    assert [h["hour"] for h in hours] == list(range(24))


def test_sweep_uses_expected_cost_not_argmax():
    """Weighting by the classifier's probabilities must give 24 distinct values. Using the
    single most likely traffic level collapsed everything onto 3, throwing away the model's
    confidence — that regression is the reason this test exists."""
    costs = {h["total_trip_cost"] for h in sweep().json()["hours"]}
    assert len(costs) > 20, f"expected a smooth curve, got {len(costs)} distinct values"


def test_sweep_probabilities_are_a_distribution():
    for h in sweep().json()["hours"]:
        probs = h["traffic_probabilities"]
        assert set(probs) == {"Low", "Medium", "High"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-2)
        assert h["traffic_confidence"] == pytest.approx(max(probs.values()), abs=1e-2)


def test_night_is_cheaper_than_rush_hour_across_the_sweep():
    hours = {h["hour"]: h["total_trip_cost"] for h in sweep().json()["hours"]}
    assert hours[3] < hours[9], "3am should beat the morning peak"
    assert hours[3] < hours[18], "3am should beat the evening peak"


def test_sweep_identifies_cheapest_and_dearest_consistently():
    body = sweep().json()
    costs = [h["total_trip_cost"] for h in body["hours"]]
    assert body["cheapest"]["total_trip_cost"] == min(costs)
    assert body["dearest"]["total_trip_cost"] == max(costs)
    assert body["max_saving"] == pytest.approx(max(costs) - min(costs), abs=0.01)


def test_sweep_rejects_the_same_city():
    assert sweep(destination_city="Jaipur").status_code == 400


def test_sweep_and_predict_agree_on_distance():
    """Both endpoints resolve the route the same way, so the cached distance must match."""
    a = sweep().json()["route"]["distance_km"]
    b = predict(vehicle_type="Sedan", month=6).json()["route"]["distance_km"]
    assert a == b


# --------------------------------------------------------------------------- metrics
def test_metrics_expose_baselines_alongside_accuracies():
    """An accuracy without its baseline is not a result. The UI relies on both."""
    m = client.get("/api/metrics").json()
    assert m["traffic_accuracy"] > m["traffic_baseline"]
    assert {"regression", "band", "traffic"} <= set(m["notes"])


def test_metrics_regression_numbers_are_sane():
    reg = client.get("/api/metrics").json()["regression"]
    assert 0.99 < reg["r2"] <= 1.0
    assert 0 < reg["mae"] < 100


def test_loss_curve_is_downsampled_and_monotonically_improving():
    body = client.get("/api/loss-curve").json()
    curve = body["curve"]
    assert 100 < len(curve) < 400, "60k raw points must not go over the wire"
    assert curve[0]["loss"] > curve[-1]["loss"], "loss should fall"
    assert body["r2_sklearn"] == pytest.approx(body["r2_gradient_descent"], abs=1e-4)


# --------------------------------------------------------------------------- distance
def test_haversine_matches_a_known_distance():
    delhi, jaipur = (28.65195, 77.23149), (26.91962, 75.78781)
    assert api.haversine_km(*delhi, *jaipur) == pytest.approx(239, abs=5)


def test_distance_source_is_declared():
    source = predict().json()["route"]["distance_source"]
    # "haversine" was the old constant-based fallback; it is now "distance_model".
    assert source in {"osrm", "distance_model"}


def test_ambiguous_city_name_resolves_to_the_largest():
    """India has several Udaipurs; the API must pick Rajasthan's, not Tripura's."""
    body = predict(start_city="Udaipur", destination_city="Ahmedabad").json()
    assert body["route"]["from"]["state"] == "Rajasthan"
    assert body["route"]["distance_km"] < 400


# ==========================================================================================
# The chained pipeline - the six sub-models that replaced app.py's hardcoded constants.
#
# These are the tests that would have caught the original problem: not "does the endpoint
# return 200" but "is a model actually doing this work, and does it report honestly when it
# cannot?"
# ==========================================================================================
def test_mileage_is_filled_by_the_model_when_the_user_omits_it():
    """Sub-model 2. The old API defaulted every vehicle to 15.5 km/l, which is wrong for an
    SUV and wrong for a hatchback in opposite directions."""
    body = client.post("/api/predict", json={
        "start_city": "Jaipur", "destination_city": "Agra"}).json()
    assert body["inputs"]["mileage_source"] == "mileage_model"
    assert 3 < body["inputs"]["mileage"] < 60


def test_mileage_model_separates_the_vehicle_and_fuel_combinations():
    """A single default would make these identical. A model must not."""
    got = {}
    for vehicle in ("Hatchback", "Sedan", "SUV"):
        body = client.post("/api/predict", json={
            "start_city": "Jaipur", "destination_city": "Agra",
            "vehicle_type": vehicle, "fuel_type": "Petrol"}).json()
        got[vehicle] = body["inputs"]["mileage"]
    assert got["SUV"] < got["Sedan"] < got["Hatchback"], got

    diesel = client.post("/api/predict", json={
        "start_city": "Jaipur", "destination_city": "Agra",
        "vehicle_type": "Sedan", "fuel_type": "Diesel"}).json()["inputs"]["mileage"]
    assert diesel > got["Sedan"], "diesel should be more efficient than petrol"


def test_supplied_mileage_always_wins():
    body = predict(mileage=21.0).json()
    assert body["inputs"]["mileage"] == 21.0
    assert body["inputs"]["mileage_source"] == "user"


def test_traffic_multipliers_are_read_out_of_the_fitted_model():
    """The multipliers are no longer written down in app.py - they are recovered from the
    litres model at startup. Because that model fits the relationship at R2 = 1.000000, the
    recovered values must equal the ones Week 4 derived from the data."""
    assert api.TRAFFIC_MULT == pytest.approx(TRAFFIC_MULTIPLIER, abs=1e-3)


def test_litres_come_from_the_model_and_still_match_the_recovered_formula():
    """Sub-model 3. Passing through a model must not change the answer, because the model
    reproduces the generating relationship exactly."""
    body = predict(traffic_level="High", mileage=15.5).json()
    distance = body["route"]["distance_km"]
    expected = (distance / 15.5) * TRAFFIC_MULTIPLIER["High"]
    assert body["derived"]["fuel_consumption_litres"] == pytest.approx(expected, rel=1e-3)


def test_parking_default_is_the_fitted_mean_not_the_old_constant():
    """Sub-model 5 is the negative result: nothing predicts parking, so the mean is served.
    The old hardcoded 70.0 was not even the mean of the training data."""
    assert api.DEFAULT_PARKING != 70.0
    assert 60 < api.DEFAULT_PARKING < 90
    body = predict().json()
    assert body["prediction"]["breakdown"]["parking"] == api.DEFAULT_PARKING


def test_offline_the_distance_model_takes_over_from_osrm(monkeypatch):
    """The fallback path is the one that used to be `haversine x 1.2355`. It must now be the
    fitted model, and it must still produce a sane distance with the network gone."""
    def boom(*args, **kwargs):
        raise OSError("no network")

    api.road_distance_km.cache_clear()
    monkeypatch.setattr(api.urllib.request, "urlopen", boom)
    try:
        body = predict().json()
        assert body["route"]["distance_source"] == "distance_model"
        # Jaipur -> Agra is 238 km by road. The model should be in the right neighbourhood,
        # and notably closer than the old constant's 274 km.
        assert 200 < body["route"]["distance_km"] < 300
    finally:
        api.road_distance_km.cache_clear()


def test_distance_model_beats_the_constant_on_the_gulf_of_khambhat():
    """The route that proves a single winding factor cannot work.

    Surat and Bhavnagar face each other across the Gulf of Khambhat: 94 km straight line,
    339 km of road around the head of the gulf. The old constant predicts 116 km - short by a
    factor of three. The model has seen this region in training and must do better.
    """
    surat = api.resolve_city("Surat")
    bhavnagar = api.resolve_city("Bhavnagar")
    straight = api.haversine_km(float(surat.lat), float(surat.lon),
                                float(bhavnagar.lat), float(bhavnagar.lon))
    constant = straight * 1.2355
    modelled = api.model_distance_km(float(surat.lat), float(surat.lon),
                                     float(bhavnagar.lat), float(bhavnagar.lon))
    truth = 339.4
    assert abs(modelled - truth) < abs(constant - truth), (
        f"model {modelled:.0f} km vs constant {constant:.0f} km, truth {truth} km")


def test_metrics_lead_with_the_served_number_not_the_flattering_one():
    """The regression test for the project's own honesty.

    /api/metrics used to report R2 = 0.999707 as if it described a prediction. It was measured
    with the true toll, parking and litres handed in as features - which a user never supplies.
    The endpoint must now report what the chain actually achieves, and say so.
    """
    m = client.get("/api/metrics").json()
    served, fed = m["served"], m["fed_true_components"]
    assert served["r2"] < fed["r2"], "the honest number must be the smaller one"
    assert served["mae"] > fed["mae"]
    assert m["honesty_gap"]["r2"] > 0
    assert m["honesty_gap"]["mae_rupees"] > 0
    assert "served" in m["notes"]["served_vs_fed"]


def test_metrics_disclose_the_parking_negative_result():
    """A model family that failed must be reported, not quietly dropped."""
    parking = client.get("/api/metrics").json()["sub_models"]["parking"]
    assert "NEGATIVE RESULT" in parking["note"]


def test_metrics_show_the_distance_model_beating_its_constant():
    d = client.get("/api/metrics").json()["sub_models"]["distance"]
    assert d["mae_km"] < d["constant_baseline_mae_km"], (
        "a sub-model that cannot beat the constant it replaced has not earned its place")
    assert "real OSRM" in d["trained_on"]


def test_metrics_error_budget_explains_the_remaining_error():
    """The chained model's RMSE should be accounted for by the noise in the data, not left
    unexplained. If these stop agreeing, something in the chain has regressed."""
    budget = client.get("/api/metrics").json()["sub_models"]["error_budget"]
    assert budget["quadrature_sum"] > 0
    ratio = budget["chained_rmse"] / budget["quadrature_sum"]
    assert 0.7 < ratio < 1.6, (
        f"chained RMSE {budget['chained_rmse']:.0f} vs irreducible "
        f"{budget['quadrature_sum']:.0f} - ratio {ratio:.2f}")


def test_the_error_bar_beside_a_prediction_describes_that_prediction():
    """Regression test for the project's worst piece of self-flattery.

    `typical_error` used to be Rs 39.66 - the MAE of the cost regressor when it was handed the
    true toll, parking and litres. A user supplies none of those, so that error bar was
    unachievable by any real prediction. It must now be the chained pipeline's own MAE.
    """
    body = predict().json()["prediction"]
    m = client.get("/api/metrics").json()
    assert body["typical_error"] == pytest.approx(m["served"]["mae"], abs=0.01)
    assert body["typical_error"] > m["fed_true_components"]["mae"]
