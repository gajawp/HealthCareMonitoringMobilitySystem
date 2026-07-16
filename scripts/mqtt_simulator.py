"""
MQTT Simulator: Publishes real chair session data through AWS IoT Core,
writes per-rep metrics to DynamoDB, and archives raw session JSON to S3.

Usage:
    python3 mqtt_simulator.py --patient-id P1001
"""
import argparse
import json
import time
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

import boto3
from awsiot import mqtt_connection_builder
from awscrt import mqtt

from app import config

# Paths to real data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPS_CSV = PROJECT_ROOT / "data" / "mobility" / "processed" / "session_20260314_131518_reps.csv"
RAW_JSON = PROJECT_ROOT / "data" / "mobility" / "raw" / "session_20260314_131518.json"


def load_reps():
    """Load per-repetition metrics from CSV."""
    reps = []
    with open(REPS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reps.append(row)
    return reps


def publish_to_iot_core(patient_id, reps):
    """Publish each rep as an MQTT message to AWS IoT Core."""
    print(f"\n[IoT Core] Connecting to {config.IOT_ENDPOINT}...")
    connection = mqtt_connection_builder.mtls_from_path(
        endpoint=config.IOT_ENDPOINT,
        cert_filepath=config.CERT_PATH,
        pri_key_filepath=config.KEY_PATH,
        ca_filepath=config.ROOT_CA_PATH,
        client_id=f"mobility-simulator-{patient_id}",
        clean_session=False,
        keep_alive_secs=30,
    )
    connect_future = connection.connect()
    connect_future.result(timeout=10)
    print("[IoT Core] Connected!")

    topic = f"{config.IOT_TOPIC_PREFIX}/{patient_id}"
    latencies = []

    for rep in reps:
        payload = {
            "patient_id": patient_id,
            "device_id": "Chair_001",
            "type": "rep_metric",
            "session_id": rep["session_id"],
            "rep_id": int(rep["rep_id"]),
            "leg": rep["leg"],
            "duration_s": float(rep["duration_s"]),
            "lift_duration_s": float(rep["lift_duration_s"]),
            "hold_duration_s": float(rep["hold_duration_s"]),
            "lower_duration_s": float(rep["lower_duration_s"]),
            "shin_lift_angle_deg": float(rep["shin_lift_angle_deg"]),
            "peak_angular_velocity": float(rep["peak_angular_velocity"]),
            "jerk_score": float(rep["jerk_score"]),
            "tof_symmetry": float(rep["tof_symmetry"]),
            "flagged": rep["flagged"] == "True",
            "flag_hold": rep["flag_hold"] == "True",
            "flag_angle": rep["flag_angle"] == "True",
            "flag_jerk": rep["flag_jerk"] == "True",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        start = time.time()
        pub_future, _ = connection.publish(
            topic=topic,
            payload=json.dumps(payload),
            qos=mqtt.QoS.AT_LEAST_ONCE,
        )
        pub_future.result(timeout=5)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        print(f"  [MQTT] Rep {rep['rep_id']} ({rep['leg']}): "
              f"published in {latency_ms:.1f} ms")
        time.sleep(0.1)  # Small delay between publishes

    connection.disconnect().result(timeout=5)
    print(f"[IoT Core] Disconnected. Published {len(reps)} reps.")
    return latencies


def write_to_dynamodb(patient_id, reps):
    """Write per-rep metrics to DynamoDB."""
    print(f"\n[DynamoDB] Writing {len(reps)} reps to {config.DYNAMODB_TABLE}...")
    dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
    table = dynamodb.Table(config.DYNAMODB_TABLE)

    latencies = []
    with table.batch_writer() as batch:
        for rep in reps:
            session_rep_key = f"{rep['session_id']}#rep{rep['rep_id']}"
            item = {
                "patient_id": patient_id,
                "session_rep_key": session_rep_key,
                "session_id": rep["session_id"],
                "rep_id": int(rep["rep_id"]),
                "leg": rep["leg"],
                "duration_s": Decimal(rep["duration_s"]),
                "lift_duration_s": Decimal(rep["lift_duration_s"]),
                "hold_duration_s": Decimal(rep["hold_duration_s"]),
                "lower_duration_s": Decimal(rep["lower_duration_s"]),
                "shin_lift_angle_deg": Decimal(rep["shin_lift_angle_deg"]),
                "peak_angular_velocity": Decimal(rep["peak_angular_velocity"]),
                "jerk_score": Decimal(rep["jerk_score"]),
                "tof_symmetry": Decimal(rep["tof_symmetry"]),
                "flagged": rep["flagged"] == "True",
                "flag_hold": rep["flag_hold"] == "True",
                "flag_angle": rep["flag_angle"] == "True",
                "flag_jerk": rep["flag_jerk"] == "True",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "device_id": "Chair_001",
            }
            start = time.time()
            batch.put_item(Item=item)
            latencies.append((time.time() - start) * 1000)

    print(f"[DynamoDB] Batch write complete.")
    return latencies


def archive_to_s3(patient_id, session_id):
    """Archive raw session JSON to S3 research data lake."""
    print(f"\n[S3] Archiving raw session to {config.S3_BUCKET}...")
    s3 = boto3.client("s3", region_name=config.AWS_REGION)

    date_str = session_id[:8]  # "20260314"
    s3_key = f"raw/{patient_id}/{date_str}/{session_id}.json"

    start = time.time()
    s3.upload_file(RAW_JSON, config.S3_BUCKET, s3_key)
    latency_ms = (time.time() - start) * 1000

    print(f"[S3] Uploaded to s3://{config.S3_BUCKET}/{s3_key} in {latency_ms:.1f} ms")
    return s3_key, latency_ms


def query_dynamodb(patient_id):
    """Query DynamoDB to verify data and measure query latency."""
    print(f"\n[DynamoDB] Querying reps for patient {patient_id}...")
    dynamodb = boto3.client("dynamodb", region_name=config.AWS_REGION)

    start = time.time()
    response = dynamodb.query(
        TableName=config.DYNAMODB_TABLE,
        KeyConditionExpression="patient_id = :pid",
        ExpressionAttributeValues={":pid": {"S": patient_id}},
    )
    latency_ms = (time.time() - start) * 1000
    count = response["Count"]
    print(f"[DynamoDB] Query returned {count} items in {latency_ms:.1f} ms")
    return count, latency_ms


def main():
    parser = argparse.ArgumentParser(description="MQTT Simulator for Chair Data")
    parser.add_argument("--patient-id", default="P1001", help="Patient ID")
    parser.add_argument("--skip-mqtt", action="store_true", help="Skip MQTT publish")
    args = parser.parse_args()

    reps = load_reps()
    session_id = reps[0]["session_id"]
    print(f"Loaded {len(reps)} reps from session {session_id}")

    results = {"patient_id": args.patient_id, "session_id": session_id,
               "total_reps": len(reps)}

    # 1. Publish to IoT Core via MQTT
    if not args.skip_mqtt:
        mqtt_latencies = publish_to_iot_core(args.patient_id, reps)
        results["mqtt_avg_latency_ms"] = sum(mqtt_latencies) / len(mqtt_latencies)
        results["mqtt_max_latency_ms"] = max(mqtt_latencies)
        results["mqtt_min_latency_ms"] = min(mqtt_latencies)
    else:
        print("\n[MQTT] Skipped.")

    # 2. Write to DynamoDB
    dynamo_latencies = write_to_dynamodb(args.patient_id, reps)
    results["dynamodb_write_latency_ms"] = sum(dynamo_latencies) / len(dynamo_latencies)

    # 3. Archive to S3
    s3_key, s3_latency = archive_to_s3(args.patient_id, session_id)
    results["s3_upload_latency_ms"] = s3_latency
    results["s3_key"] = s3_key

    # 4. Query DynamoDB to verify
    count, query_latency = query_dynamodb(args.patient_id)
    results["dynamodb_query_latency_ms"] = query_latency
    results["dynamodb_items_returned"] = count

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)

    # Save results for paper
    results_path = os.path.join(config.BASE_DIR, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
