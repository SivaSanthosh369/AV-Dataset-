import os
import math
import json
import time
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
from tqdm import tqdm
from shapely.geometry import LineString


# =========================
# CONFIG
# =========================

PLACE = "Kochi, Kerala, India"

OUTPUT_CSV = "straight_road_routes_v0.1.csv"
OUTPUT_METADATA_JSON = "street_view_metadata_v0.1.json"

MIN_LENGTH_M = 80
MAX_LENGTH_M = 300
MAX_SINUOSITY = 1.02
MAX_HEADING_CHANGE_DEG = 8

# Optional Street View check 
ENABLE_STREET_VIEW_CHECK = True

# Put your Google API key here only if ENABLE_STREET_VIEW_CHECK = True
GOOGLE_API_KEY = "AIzaSyBLmSnu-iiMgLONuti69n6noY30MlZRgs4"

STREET_VIEW_SAMPLE_INTERVAL_M = 25
MIN_STREET_VIEW_COVERAGE = 0.70


# =========================
# GEOMETRY HELPERS
# =========================

def clean_value(value):
    """
    OSM values can be lists, strings, floats, or NaN.
    This makes them CSV-friendly.
    """
    if isinstance(value, list):
        return ";".join(str(v) for v in value)

    if pd.isna(value):
        return ""

    return str(value)


def bearing(p1, p2):
    """
    Bearing in degrees using projected x/y coordinates.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360) % 360


def angle_diff(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def max_heading_change(line):
    coords = list(line.coords)

    if len(coords) < 3:
        return 0.0

    bearings = []

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        if p1 == p2:
            continue

        bearings.append(bearing(p1, p2))

    if len(bearings) < 2:
        return 0.0

    changes = [
        angle_diff(bearings[i], bearings[i + 1])
        for i in range(len(bearings) - 1)
    ]

    return max(changes) if changes else 0.0


def straight_line_distance(line):
    coords = list(line.coords)
    start = coords[0]
    end = coords[-1]
    return LineString([start, end]).length


def sample_points_along_line(line, interval_m=25):
    points = []
    distance = 0

    while distance <= line.length:
        points.append(line.interpolate(distance))
        distance += interval_m

    points.append(line.interpolate(line.length))
    return points


def get_start_end_latlng(line_wgs84):
    coords = list(line_wgs84.coords)

    start_lng, start_lat = coords[0]
    end_lng, end_lat = coords[-1]

    return start_lat, start_lng, end_lat, end_lng


# =========================
# STREET VIEW METADATA
# =========================

def get_street_view_metadata(lat, lng, api_key):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    params = {
        "location": f"{lat},{lng}",
        "key": api_key
    }

    response = requests.get(url, params=params, timeout=10)
    return response.json()


def calculate_street_view_coverage(route_id, line_projected, crs_projected, api_key):
    """
    Samples points along the projected line, converts them to WGS84,
    checks Street View metadata, and calculates coverage.
    """
    sampled_points = sample_points_along_line(
        line_projected,
        interval_m=STREET_VIEW_SAMPLE_INTERVAL_M
    )

    points_gdf = gpd.GeoDataFrame(
        geometry=sampled_points,
        crs=crs_projected
    ).to_crs(epsg=4326)

    metadata_records = []
    ok_count = 0

    for sample_index, point in enumerate(points_gdf.geometry):
        lng = point.x
        lat = point.y

        try:
            metadata = get_street_view_metadata(lat, lng, api_key)
        except Exception as e:
            metadata = {
                "status": "ERROR",
                "error": str(e)
            }

        status = metadata.get("status", "UNKNOWN")

        if status == "OK":
            ok_count += 1

        record = {
            "route_id": route_id,
            "sample_index": sample_index,
            "lat": lat,
            "lng": lng,
            "status": status,
            "pano_id": metadata.get("pano_id", ""),
            "date": metadata.get("date", ""),
            "raw_metadata": metadata
        }

        metadata_records.append(record)

        
        time.sleep(0.05)

    coverage = ok_count / len(sampled_points) if sampled_points else 0

    return coverage, metadata_records


# =========================
# MAIN PIPELINE
# =================