"""
File containing DatasetAnalyzer class
"""

from config import NETWORK_MOUNT, DATASET_SPLITS, BDD_LABELS_PREFIX, CATEGORIES
from pathlib import Path
import ijson
from analysis.scene_statistics import SceneStatistics
from analysis.category_statistics import CategoryStatistics

from collections import defaultdict


class DatasetAnalyzer:
    """
    A class for analyzing the BDD100K dataset labels by computing and saving statistics.
    """

    def get_bdd_labels_path(self) -> Path:
        """
        Returns the root directory of the BDD100K dataset labels.

        :return: Absolute path to the labels directory.
        :rtype: Path
        """
        dataset_root = Path(NETWORK_MOUNT)
        json_labels_path = (
            dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels"
        )
        return json_labels_path

    def compute_statistics(
        self, split_name, progressbar_callback=None, status_callback=None
    ) -> defaultdict[int]:
        """
        Processes the JSON labels of the BDD Dataset and computes statistics

        :param split_name: Name of the dataset split, i.e. train or val
        :param progressbar_callback: A callback for the progressbar displayed on the hello page
        :param status_callback: A callback for the processing status displayed on the hello page
        :return: Description
        :rtype: defaultdict[int, Any]
        """
        # Build the file path
        bdd_labels_path = self.get_bdd_labels_path()
        filename = BDD_LABELS_PREFIX + split_name + ".json"
        label_file = bdd_labels_path / filename
        # Initialize variables for calculating progress
        progress_counter = 0
        split_count = DATASET_SPLITS[split_name]["count"]
        full_split_name = DATASET_SPLITS[split_name]["full_name"]

        # Initialize objects for computing statistics
        scene_statistics = SceneStatistics(split_name)
        category_statistics = CategoryStatistics(split_name)

        # Open the file and stream the objects to avoid
        # loading the large .json file into memory all at once
        with open(label_file, "r") as f:
            objects = ijson.items(f, "item")
            for object in objects:
                image_name = object["name"]
                progress_counter += 1
                for attribute, value in object["attributes"].items():
                    scene_statistics.update_stats_counter(attribute, value)
                    if attribute == "weather":
                        weather = value
                    if attribute == "timeofday":
                        timeofday = value
                for label in object["labels"]:
                    if label["category"] in CATEGORIES.keys():
                        # Update counts
                        category_statistics.increment_category_total_count(
                            label["category"]
                        )
                        scene_statistics.update_category_distribution(
                            weather, timeofday, label["category"]
                        )
                        if label["attributes"]["occluded"]:
                            category_statistics.increment_category_occluded_count(
                                label["category"]
                            )
                        if label["attributes"]["truncated"]:
                            category_statistics.increment_category_truncated_count(
                                label["category"]
                            )
                        # Check if anomaly
                        x1 = label["box2d"]["x1"]
                        x2 = label["box2d"]["x2"]
                        y1 = label["box2d"]["y1"]
                        y2 = label["box2d"]["y2"]

                        # Update area
                        area = (y2 - y1) * (x2 - x1)
                        aspect_ratio = (x2 - x1) / (y2 - y1)
                        category_statistics.add_category_total_area(
                            label["category"], area
                        )
                        category_statistics.update_category_max_area(
                            label["category"], area
                        )
                        category_statistics.update_category_min_area(
                            label["category"], area
                        )
                        category_statistics.categorize_area(label["category"], area)
                        category_statistics.update_category_anomaly_count(
                            label["category"],
                            area,
                            aspect_ratio,
                            image_name,
                            x1,
                            x2,
                            y1,
                            y2,
                        )
                        category_statistics.insert_record(
                            label["category"],
                            area,
                            label["attributes"]["occluded"],
                            label["attributes"]["truncated"],
                        )

                # After processing each image, update progressbar and status callbacks
                if progressbar_callback:
                    progressbar_callback.progress(progress_counter / split_count)
                if status_callback:
                    status_callback.text(
                        f"{full_split_name} split: {progress_counter / split_count * 100:.2f}% processed"
                    )
        scene_statistics.save_csvs()
        scene_statistics.save_category_distribution_csv()
        category_statistics.save_stats_csv()
        category_statistics.save_records_csv()
        category_statistics.save_anomalies()
