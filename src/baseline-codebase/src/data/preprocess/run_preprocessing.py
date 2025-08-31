import argparse

from data.preprocess.fomo1 import convert_and_preprocess_task1
from data.preprocess.fomo2 import convert_and_preprocess_task2
from data.preprocess.fomo3 import convert_and_preprocess_task3


def preprocess_task(
    taskid: int,
    source_path: str,
    output_path: str,
    num_workers: int = None,
):
    """
    Preprocess the specified task using parallel processing.

    Args:
        taskid: Task ID to preprocess (1, 2, or 3)
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved (optional)
        num_workers: Number of parallel workers to use (default: CPU count - 1)
    """
    assert taskid in [1, 2, 3], f"Task {taskid} not supported"

    print(f"Starting preprocessing for Task {taskid}")

    if taskid == 1:
        convert_and_preprocess_task1(
            source_path=source_path, output_path=output_path, num_workers=num_workers
        )
    elif taskid == 2:
        convert_and_preprocess_task2(
            source_path=source_path, output_path=output_path, num_workers=num_workers
        )
    elif taskid == 3:
        convert_and_preprocess_task3(
            source_path=source_path, output_path=output_path, num_workers=num_workers
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallelize preprocessing for FOMO25 Challenge tasks"
    )
    parser.add_argument(
        "--taskid", type=int, required=True, help="Task ID to preprocess (1, 2, or 3)"
    )
    parser.add_argument(
        "--source_path", type=str, required=True, help="Path to the source data directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/preprocessed",
        help="Path to save preprocessed data (default: data/preprocessed)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use for preprocessing. Default is CPU count - 1",
    )
    args = parser.parse_args()

    preprocess_task(args.taskid, args.source_path, args.output_path, args.num_workers)
