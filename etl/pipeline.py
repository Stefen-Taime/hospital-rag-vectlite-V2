"""Complete ETL pipeline: extract -> transform -> load."""

import logging
import time

from etl.extract import extract
from etl.load import load
from etl.transform import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run() -> None:
    start = time.perf_counter()

    logger.info("=== ETL Pipeline START ===")

    logger.info("[1/3] Extract")
    df = extract()

    logger.info("[2/3] Transform")
    documents = transform(df)

    logger.info("[3/3] Load")
    load(documents)

    elapsed = time.perf_counter() - start
    logger.info("=== ETL Pipeline END — %.1fs ===", elapsed)


if __name__ == "__main__":
    run()
