import argparse
import sys

from hw3.spam_baseline.train import train_from_url


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spam baseline CLI")
    parser.add_argument("--dataset-url", required=True, help="Raw CSV URL for dataset")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for model and metrics")
    args = parser.parse_args(argv)

    pipeline, metrics = train_from_url(args.dataset_url, args.output_dir)
    print("Metrics:")
    print(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
