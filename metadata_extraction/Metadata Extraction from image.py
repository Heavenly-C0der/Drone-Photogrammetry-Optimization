# -*- coding: utf-8 -*-
"""
Extract GPS, camera intrinsics, and flight orientation (yaw/pitch/roll) from Autel EVO II Pro RTK images.
Outputs a CSV suitable for SfM / Bundle Adjustment initialization.
"""

import os
import pandas as pd
import utm
import exiftool
from tqdm import tqdm

# --- CONFIG ---
IMAGE_DIR = r"D:\SB Project\odm_data_helenenschacht-main\odm_data_helenenschacht-main\images"
OUTPUT_CSV = r"D:\SB Project\image_metadata.csv"
EXIFTOOL_PATH = r"C:\Program Files\exiftool-13.39_64\exiftool.exe"

# --- CAMERA CONSTANTS ---
CAMERA_MODEL = "Autel EVO II Pro RTK (Sony 1-inch CMOS)"
SENSOR_WIDTH_MM = 13.2
SENSOR_HEIGHT_MM = 8.8
DEFAULT_FOCAL_MM = 8.8
DEFAULT_IMAGE_WIDTH = 5472
DEFAULT_IMAGE_HEIGHT = 3648

# --- Helper: convert GPS DMS to decimal ---
def dms_to_deg(dms):
    try:
        d, m, s = [float(x.num) / float(x.den) for x in dms]
        return d + (m / 60.0) + (s / 3600.0)
    except:
        return None

# --- Extract metadata for a single image using ExifTool ---
def extract_metadata(et, img_path):
    metadata_json = et.execute_json("-json", img_path)
    data = metadata_json[0]

    row = {}
    row["filename"] = os.path.basename(img_path)
    row["camera_model"] = CAMERA_MODEL
    row["sensor_width_mm"] = SENSOR_WIDTH_MM
    row["sensor_height_mm"] = SENSOR_HEIGHT_MM

    # --- GPS ---
    try:
        lat = data.get("EXIF:GPSLatitude")
        lon = data.get("EXIF:GPSLongitude")
        alt = data.get("EXIF:GPSAltitude")
        if lat is not None and lon is not None:
            row["latitude"] = float(lat)
            row["longitude"] = float(lon)
            row["altitude"] = float(alt) if alt is not None else None
            easting, northing, zone_num, zone_letter = utm.from_latlon(row["latitude"], row["longitude"])
            row["easting"] = easting
            row["northing"] = northing
            row["utm_zone"] = f"{zone_num}{zone_letter}"
        else:
            row["latitude"] = row["longitude"] = row["altitude"] = None
            row["easting"] = row["northing"] = row["utm_zone"] = None
    except:
        row["latitude"] = row["longitude"] = row["altitude"] = None
        row["easting"] = row["northing"] = row["utm_zone"] = None

    # --- Camera intrinsics ---
    try:
        focal_mm = float(data.get("EXIF:FocalLength", DEFAULT_FOCAL_MM))
        img_width = int(data.get("EXIF:ExifImageWidth", DEFAULT_IMAGE_WIDTH))
        img_height = int(data.get("EXIF:ExifImageLength", DEFAULT_IMAGE_HEIGHT))
        fx = (focal_mm * img_width) / SENSOR_WIDTH_MM
        fy = (focal_mm * img_height) / SENSOR_HEIGHT_MM
        cx = img_width / 2.0
        cy = img_height / 2.0
    except:
        focal_mm = DEFAULT_FOCAL_MM
        img_width, img_height, fx, fy, cx, cy = DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, None, None, None, None

    row.update({
        "focal_length_mm": focal_mm,
        "image_width_px": img_width,
        "image_height_px": img_height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    })

    # --- Orientation: flight yaw/pitch/roll (in degrees) ---
    row["yaw_deg"] = float(data.get("XMP:FlightYawDegree", 0.0))
    row["pitch_deg"] = float(data.get("XMP:FlightPitchDegree", 0.0))
    row["roll_deg"] = float(data.get("XMP:FlightRollDegree", 0.0))

    # --- Camera ID ---
    row["camera_id"] = 1

    return row

# --- Main ---
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    rows = []

    with exiftool.ExifTool(executable=EXIFTOOL_PATH) as et:
        for fname in tqdm(image_files, desc="Extracting metadata"):
            img_path = os.path.join(IMAGE_DIR, fname)
            row = extract_metadata(et, img_path)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Order columns
    df = df[[
        "camera_id", "filename", "latitude", "longitude", "altitude",
        "easting", "northing", "utm_zone", "yaw_deg", "pitch_deg", "roll_deg",
        "camera_model", "sensor_width_mm", "sensor_height_mm",
        "focal_length_mm", "image_width_px", "image_height_px",
        "fx", "fy", "cx", "cy"
    ]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Metadata CSV saved: {OUTPUT_CSV} ({len(df)} images)")

if __name__ == "__main__":
    main()
