from config import NETWORK_MOUNT, DATASET_SPLITS, JSON_LABELS_PREFIX
from pathlib import Path
import ijson

from collections import defaultdict


class DatasetAnalyzer:
    """
    Docstring for DatasetAnalyzer
    """

    def get_json_labels_path(self) -> str:
        """
        Docstring for get_labels_path

        :return: Description
        :rtype: str
        """
        dataset_root = Path(NETWORK_MOUNT)
        json_labels_path = (
            dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels"
        )
        return json_labels_path

    def read_labels(
        self, split_name, progressbar_callback=None, status_callback=None
    ) -> defaultdict[int]:
        """
        Docstring for read_labels

        :param self: Description
        :param split_name: Description
        :param progressbar_callback: Description
        :param status_callback: Description
        :return: Description
        :rtype: defaultdict[int, Any]
        """
        json_labels_path = self.get_json_labels_path()
        filename = JSON_LABELS_PREFIX + split_name + ".json"
        count_dict = defaultdict(int)
        label_file = json_labels_path / filename
        progress_counter = 0
        split_count = DATASET_SPLITS[split_name]["count"]

        with open(label_file, "r") as f:
            objects = ijson.items(f, "item")
            for object in objects:
                progress_counter += 1
                for label in object["labels"]:
                    count_dict[label["category"]] += 1
                if progressbar_callback:
                    progressbar_callback.progress(progress_counter / split_count)
                if status_callback:
                    status_callback.text(
                        f"{split_name} split processing {progress_counter / split_count * 100:.2f}% complete"
                    )
        return f"Split {split_name} has distribution {count_dict}"
