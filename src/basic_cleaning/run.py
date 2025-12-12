#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    df = pd.read_csv(artifact_local_path)

    # Fill missing values in 'reviews_per_month' with 0
    logger.info("Filling missing values in 'reviews_per_month'")
    df['reviews_per_month'].fillna(0, inplace=True)

    # Convert 'last_review' to datetime
    logger.info("Converting 'last_review' to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Filter the DataFrame to keep only rows where the price is between min_price and max_price
    logger.info(f"Filtering price between {args.min_price} and {args.max_price}")
    df = df.query(f'{args.min_price} <= price <= {args.max_price}').copy()

    # Save the cleaned DataFrame
    output_file = "clean_sample.csv"
    df.to_csv(output_file, index=False)

    # Upload the artifact to W&B
    logger.info(f"Uploading artifact {args.output_artifact}")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    # Clean up the local file
    os.remove(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for filtering",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for filtering",
        required=True
    )

    args = parser.parse_args()

    go(args)
