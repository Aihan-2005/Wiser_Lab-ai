import argparse

from persian_sea.config import load_config
from persian_sea.segmentation.runner import segment_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    segment_all(cfg)


if __name__ == "__main__":
    main()
