import json
import os

def infer_schema():
    os.makedirs("/opt/airflow/reports", exist_ok=True)

    # Schema was already saved by Node 4
    schema = json.load(open("/opt/airflow/reports/schema.json"))
    print("Schema loaded from reports/schema.json")
    print(json.dumps(schema, indent=2))
    print("Node 5 complete.")