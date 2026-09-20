"""
API tests for the Road Trip Cost Predictor.

Run:  pytest -q

These deliberately cover the things that broke during development rather than trivially
exercising every endpoint: the derived-feature arithmetic, the error paths, and the honesty
warnings - which are easy to silently drop and hard to notice missing.

Network note: /api/distance prefers live OSRM routing and falls back to the fitted distance
model. Tests assert on behaviour that holds either way, so they pass offline.
"""
import pytest
from fastapi.testclient import TestClient

import app as api

client = TestClient(api.app)

# Per-km running costs recovered from the data - see README "Results".
RUNNING_COST = {"Hatchback": 0.5694, "Sedan": 0.6943, "SUV": 0.8366}
TOLL_RATE = 1.301

VEHICLES = ["Hatchback", "Sedan", "SUV"]
FUELS = ["Petrol", "Diesel", "CNG"]


def predict(**overrides):
    """A valid request, with fields overridable per test.

    Every price is supplied so the default request produces NO warnings; a test that wants the
    national-average warning drops one deliberately. Keeping the baseline warning-free is what
    makes `assert not warnings` a meaningful assertion elsewhere.
    """
    body = {"distance_km": 500, "petrol_price": 106.0, "diesel_price": 98.0,
            "cng_price": 84.0, "parking_cost": 70, "passengers": 4}
    body.update(overrides)
    return client.post("/api/predict", json=body)


def rows(body, fuel="Petrol"):
    return body["results"][fuel]


def row_for(body, vehicle, fuel="Petrol"):
    return next(r for r in rows(body, fuel) if r["vehicle_type"] == vehicle)


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


def test_city_payload_carries_the_state_fuel_prices():
    """So the lookup page can offer a starting point without another round trip."""
    city = client.get("/api/cities", params={"q": "Jaipur"}).json()["results"][0]
    prices = city["fuel_prices"]
    assert {"petrol", "diesel", "cng", "source"} <= prices.keys()
    assert 50 < prices["petrol"] < 200


# --------------------------------------------------------------------------- distance
def test_distance_between_two_cities():
    body = client.get("/api/distance", params={"from": "Jaipur", "to": "Agra"}).json()
    assert 200 < body["distance_km"] < 300          # 238 km by road
    assert body["straight_line_km"] < body["distance_km"]
    assert body["source"] in {"measured", "distance_model"}
    assert body["in_trained_range"] is True


def test_distance_rejects_the_same_city():
    r = client.get("/api/distance", params={"from": "Jaipur", "to": "Jaipur"})
    assert r.status_code == 400


def test_unknown_city_is_404_not_a_silent_guess():
    r = client.get("/api/distance", params={"from": "Atlantis", "to": "Agra"})
    assert r.status_code == 404


def test_city_outside_the_domain_is_rejected():
    """Population < 100k is out of domain; the API must say so rather than extrapolate.

    Luckeesarai is in india_cities.csv at 99,979 - just under the cut. Picking a town that is
    absent from the file entirely would pass for the wrong reason.
    """
    assert "Luckeesarai" in set(api.CITIES.city), "pick another city just under the cut"
    r = client.get("/api/distance", params={"from": "Luckeesarai", "to": "Agra"})
    assert r.status_code == 404


def test_ambiguous_city_name_resolves_to_the_largest():
    """India has several Udaipurs; the API must pick Rajasthan's, not Tripura's."""
    body = client.get("/api/distance",
                      params={"from": "Udaipur", "to": "Ahmedabad"}).json()
    assert body["from"]["state"] == "Rajasthan"
    assert body["distance_km"] < 400


def test_winding_ratio_is_not_a_constant():
    """The reason a distance MODEL exists rather than one multiplier.

    A straight Deccan highway and a route around a gulf cannot share a ratio. If these two
    came back equal, something had collapsed to a constant.
    """
    straight = client.get("/api/distance",
                          params={"from": "Surat", "to": "Bhavnagar"}).json()
    plain = client.get("/api/distance", params={"from": "Delhi", "to": "Agra"}).json()
    assert straight["winding_ratio"] > 2.5      # around the Gulf of Khambhat
    assert plain["winding_ratio"] < 1.5


def test_haversine_matches_a_known_distance():
    delhi, jaipur = (28.65195, 77.23149), (26.91962, 75.78781)
    assert api.rf.haversine(*delhi, *jaipur) == pytest.approx(239, abs=5)


def test_offline_the_distance_model_takes_over_from_osrm(monkeypatch):
    """The fallback must be the fitted model, and must still give a sane answer offline."""
    def boom(*args, **kwargs):
        raise OSError("no network")

    api.road_distance_km.cache_clear()
    monkeypatch.setattr(api.urllib.request, "urlopen", boom)
    try:
        body = client.get("/api/distance", params={"from": "Jaipur", "to": "Agra"}).json()
        assert body["source"] == "distance_model"
        assert 200 < body["distance_km"] < 300
    finally:
        api.road_distance_km.cache_clear()


def test_distance_model_beats_a_single_ratio_on_the_gulf_of_khambhat():
    """The route that proves one winding factor cannot serve the whole country.

    Surat and Bhavnagar face each other across the Gulf of Khambhat: 94 km straight line,
    339 km of road around the head of the gulf. A national-average ratio predicts 116 km -
    short by a factor of three. The model has seen this region in training and must do better.
    """
    surat = api.resolve_city("Surat")
    bhavnagar = api.resolve_city("Bhavnagar")
    straight = api.rf.haversine(float(surat.lat), float(surat.lon),
                                float(bhavnagar.lat), float(bhavnagar.lon))
    flat_ratio = straight * 1.2355
    modelled = api.model_distance_km(float(surat.lat), float(surat.lon),
                                     float(bhavnagar.lat), float(bhavnagar.lon))
    truth = 339.4
    assert abs(modelled - truth) < abs(flat_ratio - truth), (
        f"model {modelled:.0f} km vs flat ratio {flat_ratio:.0f} km, truth {truth} km")


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("field,value", [
    ("distance_km", 0),         # would divide by zero on cost_per_km
    ("distance_km", 5),         # below anything the toll model has seen
    ("distance_km", 20000),     # extrapolates into six figures without erroring
    ("distance_km", -100),
    ("petrol_price", 5),
    ("petrol_price", 1000),
    ("parking_cost", -1),
    ("passengers", 0),
    ("passengers", 99),
])
def test_out_of_range_fields_are_422(field, value):
    assert predict(**{field: value}).status_code == 422


def test_distance_is_required():
    assert client.post("/api/predict", json={"petrol_price": 106.0}).status_code == 422


@pytest.mark.parametrize("stale", ["departure_hour", "month", "traffic_level",
                                   "vehicle_type", "start_city", "fuel_type"])
def test_fields_this_api_no_longer_reads_are_rejected(stale):
    """A stale client must get a 422, not a confident answer computed without its input.

    Silently ignoring an unknown field is the worst outcome: the caller believes it asked for
    an SUV at 9am and gets back a number that honoured neither.
    """
    assert predict(**{stale: "x"}).status_code == 422


# --------------------------------------------------------------------- the three vehicles
def test_every_vehicle_and_fuel_combination_comes_back():
    body = predict().json()
    assert set(body["results"]) == set(FUELS)
    for fuel in FUELS:
        assert [r["vehicle_type"] for r in rows(body, fuel)] == VEHICLES


def test_each_vehicle_gets_its_own_mileage():
    """The whole point of the three-vehicle output. One shared figure would be a lie, and it
    would also flatten the comparison to nothing but running cost."""
    body = predict().json()
    mileages = [r["mileage"] for r in rows(body)]
    assert len(set(mileages)) == 3
    assert mileages[0] > mileages[1] > mileages[2], "hatchback > sedan > SUV"
    assert all(r["mileage_source"] == "mileage_model" for r in rows(body))


def test_the_vehicle_gap_scales_with_distance():
    """Running costs differ PER KILOMETRE, so the gap between a hatchback and an SUV must grow
    with the trip. A flat offset here means the distance x vehicle interaction has been lost
    from COST_FEATURES, which is invisible in any single estimate."""
    short = predict(distance_km=200).json()
    long_ = predict(distance_km=1200).json()
    gap_short = (row_for(short, "SUV")["total_trip_cost"]
                 - row_for(short, "Hatchback")["total_trip_cost"])
    gap_long = (row_for(long_, "SUV")["total_trip_cost"]
                - row_for(long_, "Hatchback")["total_trip_cost"])
    assert gap_long > gap_short * 4, (
        f"gap barely moved: Rs {gap_short:.0f} at 200 km vs Rs {gap_long:.0f} at 1200 km")


def test_vehicles_are_ordered_by_cost():
    body = predict().json()
    totals = [r["total_trip_cost"] for r in rows(body)]
    assert totals[0] < totals[1] < totals[2], "hatchback cheapest, SUV dearest"


def test_running_cost_orders_correctly_by_vehicle_and_stays_plausible():
    body = predict(distance_km=500).json()
    costs = {r["vehicle_type"]: r["breakdown"]["running_cost"] for r in rows(body)}
    assert costs["Hatchback"] < costs["Sedan"] < costs["SUV"]
    for vehicle, per_km in RUNNING_COST.items():
        expected = per_km * 500
        assert costs[vehicle] == pytest.approx(expected, rel=0.6), (
            f"{vehicle} running cost {costs[vehicle]:.0f} is nowhere near {expected:.0f}")


def test_a_mileage_override_applies_to_only_that_vehicle():
    body = predict(mileage={"SUV": 9.0}).json()
    assert row_for(body, "SUV")["mileage"] == 9.0
    assert row_for(body, "SUV")["mileage_source"] == "user"
    assert row_for(body, "Hatchback")["mileage_source"] == "mileage_model"


def test_a_worse_mileage_costs_more():
    thirsty = row_for(predict(mileage={"SUV": 8.0}).json(), "SUV")["total_trip_cost"]
    frugal = row_for(predict(mileage={"SUV": 18.0}).json(), "SUV")["total_trip_cost"]
    assert thirsty > frugal


def test_mileage_model_separates_all_nine_vehicle_fuel_cells():
    """Additive one-hots alone cannot reach nine independent cells - that is what the
    interaction terms in MILEAGE_FEATURES are for. If any two cells collapse, they are gone."""
    seen = {(v, f): api.default_mileage(v, f) for v in VEHICLES for f in FUELS}
    assert len(set(seen.values())) == 9, seen


# --------------------------------------------------------------------------- arithmetic
def test_breakdown_sums_to_the_total():
    for row in rows(predict().json()):
        assert sum(row["breakdown"].values()) == pytest.approx(row["total_trip_cost"], abs=1.0)


def test_fuel_component_is_litres_times_price():
    row = row_for(predict(petrol_price=106.0).json(), "Sedan")
    assert row["breakdown"]["fuel"] == pytest.approx(
        row["fuel_consumption_litres"] * 106.0, abs=1.0)


def test_the_explanation_string_reproduces_its_own_numbers():
    """The line printed under each result must be checkable by a reader with a calculator.

    It shows the textbook litres and the predicted litres as separate quantities precisely
    because the model has an intercept - printing a single multiplier would look right and
    not quite reconcile.
    """
    row = row_for(predict(distance_km=500).json(), "Sedan")
    text = row["fuel_consumption_explained"]
    on_paper = 500 / row["mileage"]
    assert f"{on_paper:.2f} L on paper" in text
    assert f"predicted {row['fuel_consumption_litres']} L" in text


def test_litres_are_derived_not_requested():
    """A traveller cannot know their fuel burn before setting off, so it must never be an
    input - it is the strongest feature in the cost model and handing it over inflates R2."""
    assert "fuel_consumption_litres" not in api.PredictRequest.model_fields
    assert predict(fuel_consumption_litres=30).status_code == 422


def test_toll_is_derived_not_requested():
    assert "toll_cost" not in api.PredictRequest.model_fields
    for row in rows(predict().json()):
        assert row["toll_cost"] > 0


def test_toll_tracks_the_recovered_rate_without_being_hardcoded_to_it():
    toll = row_for(predict(distance_km=500).json(), "Sedan")["toll_cost"]
    assert toll == pytest.approx(500 * TOLL_RATE, rel=0.25)


def test_toll_is_a_model_and_not_a_single_multiplication():
    """If toll were distance x rate, toll/distance would be identical at every distance."""
    per_km = [
        row_for(predict(distance_km=d).json(), "Sedan")["toll_cost"] / d
        for d in (100, 500, 1200)
    ]
    assert len(set(round(p, 4) for p in per_km)) > 1


def test_toll_is_identical_across_vehicles():
    """Tolls are flat across vehicle classes in this data - a hatchback pays what an SUV pays.
    A toll that varied by vehicle would mean the wrong feature reached the model."""
    tolls = {r["toll_cost"] for r in rows(predict().json())}
    assert len(tolls) == 1


def test_parking_is_passed_through_untouched():
    """It is a user input precisely because no model beat the mean on it."""
    for row in rows(predict(parking_cost=250).json()):
        assert row["breakdown"]["parking"] == 250


def test_cost_per_passenger_divides_the_total():
    body = predict(passengers=5).json()
    row = row_for(body, "Sedan")
    assert row["cost_per_passenger"] == pytest.approx(row["total_trip_cost"] / 5, abs=1.0)


def test_cost_band_is_one_of_the_four_classes():
    for row in rows(predict().json()):
        assert row["cost_band"] in {"Excellent", "Good", "Average", "Poor"}


def test_cost_band_is_not_stuck_on_one_value():
    """A band that never moves is the signature of a classifier receiving a column of zeros
    where a category should be - it fails silently, with no exception."""
    bands = set()
    for distance in (100, 400, 900, 1400):
        bands |= {r["cost_band"] for r in rows(predict(distance_km=distance).json())}
    assert len(bands) > 1


# --------------------------------------------------------------------------- fuel prices
def test_a_supplied_price_changes_the_estimate():
    cheap = row_for(predict(petrol_price=80.0).json(), "Sedan")["total_trip_cost"]
    dear = row_for(predict(petrol_price=140.0).json(), "Sedan")["total_trip_cost"]
    assert dear > cheap


def test_price_only_moves_its_own_fuel():
    base = predict().json()
    bumped = predict(petrol_price=150.0).json()
    assert (row_for(bumped, "Sedan", "Petrol")["total_trip_cost"]
            > row_for(base, "Sedan", "Petrol")["total_trip_cost"])
    assert (row_for(bumped, "Sedan", "Diesel")["total_trip_cost"]
            == row_for(base, "Sedan", "Diesel")["total_trip_cost"])


def test_an_omitted_price_falls_back_to_the_national_average_and_says_so():
    body = predict(cng_price=None).json()
    assert body["inputs"]["fuel_price_sources"]["CNG"] == "national_average"
    assert body["inputs"]["fuel_prices"]["CNG"] == api.NATIONAL_PRICE["CNG"]
    assert any("CNG" in w for w in body["warnings"])


def test_a_supplied_price_is_labelled_as_the_users_and_raises_no_warning():
    body = predict().json()
    assert set(body["inputs"]["fuel_price_sources"].values()) == {"user"}
    assert not body["warnings"]


# --------------------------------------------------------------------------- honesty
def test_long_route_warns_that_it_is_extrapolating():
    body = predict(distance_km=2500).json()
    assert any("extrapolating" in w for w in body["warnings"])


def test_in_range_route_with_supplied_prices_has_no_warnings_at_all():
    assert predict(distance_km=500).json()["warnings"] == []


def test_the_error_bar_beside_a_prediction_describes_that_prediction():
    """It must be the chained pipeline's own held-out MAE, not the far smaller figure the
    regressor scores when handed the true toll and litres."""
    quoted = predict().json()["prediction"]["typical_error"]
    served = api.PIPE["metrics"]["cost_chained"]["mae"]
    fed = api.PIPE["metrics"]["cost_true_components"]["mae"]
    assert quoted == pytest.approx(served, abs=0.01)
    assert quoted > fed * 3, "the flattering number has crept back into the UI"


def test_cheapest_and_dearest_are_consistent_with_the_grid():
    body = predict().json()
    every = [r["total_trip_cost"] for fuel in FUELS for r in rows(body, fuel)]
    assert body["cheapest"]["total_trip_cost"] == min(every)
    assert body["dearest"]["total_trip_cost"] == max(every)
    assert body["max_saving"] == pytest.approx(max(every) - min(every), abs=0.01)


# --------------------------------------------------------------------------- metrics
def test_metrics_lead_with_the_served_number_not_the_flattering_one():
    """The regression test for the project's own honesty.

    `served` is what a user receives. `fed_true_components` is the same regressor handed the
    litres and toll - a question nobody can ask. Both are reported; only one is the headline.
    """
    m = client.get("/api/metrics").json()
    assert m["served"]["r2"] < m["fed_true_components"]["r2"]
    assert m["honesty_gap"]["r2"] > 0
    assert m["served"]["mae"] > m["fed_true_components"]["mae"]


def test_metrics_regression_numbers_are_sane():
    m = client.get("/api/metrics").json()
    assert 0.9 < m["served"]["r2"] < 1.0
    assert m["served"]["rmse"] > m["served"]["mae"] > 0


def test_metrics_disclose_the_parking_negative_result():
    parking = client.get("/api/metrics").json()["sub_models"]["parking"]
    assert parking["best_r2"] <= 0, "a model that failed must be reported as having failed"
    assert "NEGATIVE RESULT" in parking["note"]


def test_metrics_show_the_distance_model_beating_its_baseline():
    d = client.get("/api/metrics").json()["sub_models"]["distance"]
    assert d["mae_km"] < d["fixed_ratio_baseline_mae_km"]


def test_metrics_report_the_litres_slope_above_one():
    """Real driving burns more than distance / mileage. If this ever came back at exactly 1.0
    the model would have stopped learning the real-world penalty."""
    litres = client.get("/api/metrics").json()["sub_models"]["litres"]
    assert 1.0 < litres["fitted_ratio_slope"] < 1.2


def test_metrics_error_budget_explains_the_remaining_error():
    """The chain should sit near the floor this data allows, not above it - otherwise the
    error is the model's weakness rather than the data's unpredictability."""
    eb = client.get("/api/metrics").json()["sub_models"]["error_budget"]
    assert eb["parking_sd"] == 0, "parking is typed, so it contributes no error"
    assert eb["distance_sd_km"] == 0, "distance is typed, so it contributes no error"
    ratio = eb["chained_rmse"] / eb["quadrature_sum"]
    assert 0.7 < ratio < 1.6, f"budget and achieved RMSE disagree (ratio {ratio:.2f})"


def test_metrics_disclose_the_training_data_convention_gap():
    """The cost model learned on a synthetic distance convention and is served real measured
    kilometres. That gap has to be stated, not discovered."""
    gap = client.get("/api/metrics").json()["distance_convention_gap"]
    assert abs(gap["mean_signed_pct"]) < 2, "systematic bias should be small"
    assert gap["mean_abs_pct"] > 2, "per-route spread is the part that does not cancel"


# --------------------------------------------------------------------------- mileage endpoint
def test_default_mileage_endpoint_returns_all_three_vehicles():
    body = client.get("/api/default-mileage", params={"fuel": "Diesel"}).json()
    assert set(body["mileage"]) == set(VEHICLES)
    assert body["mileage"]["Hatchback"] > body["mileage"]["SUV"]


def test_default_mileage_rejects_an_unknown_fuel():
    assert client.get("/api/default-mileage", params={"fuel": "Kerosene"}).status_code == 422


# --------------------------------------------------------------------------- loss curve
def test_loss_curve_is_downsampled_and_monotonically_improving():
    body = client.get("/api/loss-curve").json()
    curve = body["curve"]
    assert 100 < len(curve) < 400, "60k raw points must not go over the wire"
    assert curve[0]["loss"] > curve[-1]["loss"], "loss should fall"
    assert body["r2_sklearn"] == pytest.approx(body["r2_gradient_descent"], abs=1e-4)


def test_loss_curve_features_match_the_served_feature_list():
    """The exhibit must describe the model that actually ships, not a stale column order."""
    assert client.get("/api/loss-curve").json()["features"] == api.rf.COST_FEATURES
