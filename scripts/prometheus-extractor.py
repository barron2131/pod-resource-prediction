import json
import requests
import argparse
import csv
from typing import Dict


PROMETHEUS_ENDPOINT = "http://localhost:9090/api/v1/query_range"
PROMETHEUS_QUERY = 'sum(rate({}{{pod="{}"}}[1m])) by (pod)'


def get_data(metric: str, pod: str, start: int, end: int, step: int = 1) -> Dict:
    query = PROMETHEUS_QUERY.format(metric, pod)

    try:
        response = requests.get(PROMETHEUS_ENDPOINT, {
            "query": query, "start": start, "end": end, "step": step
        })
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise err

    response = json.loads(response.text)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus API metrics extractor")
    parser.add_argument("--metric", required=True, type=str,
                        help="Prometheus counter metric.")
    parser.add_argument("--pod", required=True, type=str,
                        help="Pod name (use `kubectl get raw` to see the name of every pod in your service).")
    parser.add_argument("--start", required=True, type=int,
                        help="Start timestamp, inclusive")
    parser.add_argument("--end", required=True, type=int,
                        help=" End timestamp, inclusive.")
    parser.add_argument("--step", default=60, type=int,
                        help="Query resolution step width (in seconds).")
    parser.add_argument("--output", required=True, type=str,
                        help="Output filename.")
    args = parser.parse_args()

    query_result = get_data(args.metric, args.pod, args.start, args.end, args.step)
    values = query_result["data"]["result"][0]["values"]

    with open(args.output, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for [timestamp, value] in values:
            writer.writerow({"timestamp": timestamp, "value": float(value)})
