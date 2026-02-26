import argparse
import sys
import logging
import resource
from utils.experiments import Experiments

logger = logging.getLogger(__name__)


def log_resources() -> None:
    """Log the resources used in the experiment.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)

    logger.info(f"User CPU time: {usage.ru_utime:.2f}s")
    logger.info(f"System CPU time: {usage.ru_stime:.2f}s")
    logger.info(f"Max RSS: {usage.ru_maxrss / 1024:.2f} MB")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running experiments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Initializes and runs experiment pipeline."
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="Schirrmeister2017",
        help="Name of the dataset used in the experiment."
    )

    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Single subject number to run."
    )

    parser.add_argument(
        "--subject-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Run experiments for a range of subjects (inclusive)."
    )

    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Run inversion experiment."
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning."
    )

    parser.add_argument(
        "--react",
        action="store_true",
        help="Enable ReAct."
    )

    return parser.parse_args()


def main() -> None:
    """Initializes the experiment object. Reads the experiment parameters
    from the terminal from which the script is called.
    """
    args = parse_args()

    exp = Experiments(dataset=args.dataset)

    if args.subject_range:
        start, end = args.subject_range

        if start > end:
            raise ValueError("Subject range START must be <= END.")

        subjects = list(range(start, end + 1))
        print("Running experiments for subjects:", subjects)

        for subject in subjects:
            exp.experiment(
                inverse=args.inverse,
                subject=subject,
                tune=args.tune,
                ReAct=args.react
            )

    else:
        exp.experiment(
            inverse=args.inverse,
            subject=args.subject,
            tune=args.tune,
            ReAct=args.react
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        main()
        log_resources()
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
