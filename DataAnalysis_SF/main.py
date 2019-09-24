import os
import time
from google.cloud import bigquery
from google.cloud.bigquery.client import Client


def main():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'EiS_Thesis.json'
    bq_client = Client()

    # Perform a query.
    QUERY = ('SELECT * FROM `bigquery-public-data.san_francisco.bikeshare_status`;')

    print("Estimated Size: ", estimate_query_size(bq_client, QUERY), " GB")
    query_job = bq_client.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish

    start_time = time.time()

    with open('SF_BikeShare.csv', "w") as f:
        print("time, station_id, bikes_available, docks_available", sep=',', file=f)
        i = 0
        for row in rows:
            if i % 100000 == 0:
                print(i, time.time() - start_time)
            print(row.time, row.station_id, row.bikes_available, row.docks_available, sep=',', file=f)
            i = i + 1

    end_time = time.time() - start_time
    print("Time required: ", end_time)


def estimate_query_size(client, query):
    """
    Estimate gigabytes scanned by query.
    Does not consider if there is a cached query table.
    See https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    """
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = client.query(query, job_config=my_job_config)
    return my_job.total_bytes_processed / (1024 * 10**6)


if __name__ == "__main__":
    main()
