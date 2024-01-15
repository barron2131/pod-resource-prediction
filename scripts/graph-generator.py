import json
import requests
import argparse
import csv
from typing import Dict


PROMETHEUS_ENDPOINT = "http://localhost:9090/api/v1/query_range"
PROMETHEUS_QUERY = "sum(rate(istio_requests_total[1m])) by (source_workload, destination_workload)"


def get_data(start: int, end: int, step: int = 1) -> Dict:
    try:
        response = requests.get(PROMETHEUS_ENDPOINT, {
            "query": PROMETHEUS_QUERY, "start": start, "end": end, "step": step
        })
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise err

    response = json.loads(response.text)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus graph generator")
    parser.add_argument("--start", required=True, type=int,
                        help="Start timestamp, inclusive")
    parser.add_argument("--end", required=True, type=int,
                        help=" End timestamp, inclusive.")
    parser.add_argument("--step", default=60, type=int,
                        help="Query resolution step width (in seconds).")
    parser.add_argument("--output", required=True, type=str,
                        help="Output filename.")
    args = parser.parse_args()

    query_result = get_data(args.start, args.end, args.step)["data"]["result"]
    with open(args.output, "w", newline="") as csvfile:
        fieldnames = ["from", "to", "timestamp", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in query_result:
            source = result["metric"]["source_workload"]
            dest = result["metric"]["destination_workload"]

            if source == "unknown" or dest == "unknown" or source == "loadgenerator" or source == "loadgenerator":
                continue

            for [timestamp, value] in result["values"]:
                writer.writerow({
                    "from": source, "to": dest,
                    "timestamp": timestamp, "value": float(value)
                })
