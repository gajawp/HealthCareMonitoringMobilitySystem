"""Centralized configuration for the Healthcare Mobility Monitoring System."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# AWS General
AWS_REGION = "us-west-2"
AWS_ACCOUNT_ID = "107970446096"

# AWS IoT Core
IOT_ENDPOINT = "alhzwqwdkq0mw-ats.iot.us-west-2.amazonaws.com"
IOT_TOPIC_PREFIX = "mobility/sessions"
CERT_DIR = os.path.join(BASE_DIR, "certs")
CERT_PATH = os.path.join(CERT_DIR, "device.pem.crt")
KEY_PATH = os.path.join(CERT_DIR, "private.pem.key")
ROOT_CA_PATH = os.path.join(CERT_DIR, "AmazonRootCA1.pem")

# DynamoDB
DYNAMODB_TABLE = "MobilitySessionReps"

# S3
S3_BUCKET = "mobility-research-datalake-107970446096"

# Data source toggle: "aws" or "mock"
DATA_SOURCE = os.environ.get("DATA_SOURCE", "mock")
