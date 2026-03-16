# src/procesamiento/capa1/config_dicts.py

import re

# ==========================================
# CONSTANTES GLOBALES Y REGLAS FÍSICAS
# ==========================================
MAX_TRIP_DURATION_MIN = 360.0   # 6 horas máximo

# Warnings de importes extremos
EXTREME_TOTAL_AMOUNT = 500.0
EXTREME_FARE_AMOUNT = 400.0
EXTREME_BASE_FARE = 400.0
EXTREME_DRIVER_PAY = 400.0

# ==========================================
# YELLOW TAXIS (TPEP)
# ==========================================
YELLOW_EXPECTED_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", 
    "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", 
    "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", 
    "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee", "cbd_congestion_fee"
]

YELLOW_ALLOWED_VENDOR_ID = {1, 2, 6, 7}
YELLOW_ALLOWED_RATECODE_ID = {1, 2, 3, 4, 5, 6} # Quitamos el 99, lo trataremos como nulo
YELLOW_ALLOWED_STORE_FLAG = {"Y", "N"}
YELLOW_ALLOWED_PAYMENT_TYPE = {0, 1, 2, 3, 4} # 5 y 6 (Unknown/Voided) los trataremos como nulos

# ==========================================
# GREEN TAXIS (LPEP)
# ==========================================
GREEN_EXPECTED_COLUMNS = [
    "VendorID", "lpep_pickup_datetime", "lpep_dropoff_datetime", "store_and_fwd_flag", 
    "RatecodeID", "PULocationID", "DOLocationID", "passenger_count", "trip_distance", 
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", 
    "total_amount", "payment_type", "trip_type", "congestion_surcharge", "cbd_congestion_fee"
]

GREEN_OPTIONAL_COLUMNS = {"cbd_congestion_fee"}

GREEN_ALLOWED_VENDOR_ID = {1, 2, 6}
GREEN_ALLOWED_RATECODE_ID = {1, 2, 3, 4, 5, 6} # Quitamos el 99, lo trataremos como nulo
GREEN_ALLOWED_TRIP_TYPE = {1, 2}
GREEN_ALLOWED_PAYMENT_TYPE = {0, 1, 2, 3, 4} # 5 y 6 (Unknown/Voided) los trataremos como nulos
GREEN_ALLOWED_STORE_FLAG = {"Y", "N"}

# ==========================================
# HVFHV (UBER / LYFT / ETC)
# ==========================================
FHVHV_EXPECTED_COLUMNS = [
    "hvfhs_license_num", "dispatching_base_num", "originating_base_num", "request_datetime", 
    "on_scene_datetime", "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID", 
    "trip_miles", "trip_time", "base_passenger_fare", "tolls", "bcf", "sales_tax", 
    "congestion_surcharge", "airport_fee", "tips", "driver_pay", "shared_request_flag", 
    "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag", "cbd_congestion_fee"
]

FHVHV_KNOWN_LICENSES = {"HV0002", "HV0003", "HV0004", "HV0005"}
FHVHV_HVFHS_PATTERN = re.compile(r"^HV\d{4}$")
FHVHV_BASE_NUM_PATTERN = re.compile(r"^[A-Z]\d{5}$")  # típico TLC: B00013
FHVHV_YN_FLAGS = {"shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag"}

FHVHV_ALLOWED_YN = {"Y", "N"}

FHVHV_FLOAT_COLS = [
    "trip_miles",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "cbd_congestion_fee",
]