"""
File containing CategoryStatistics class
"""

from collections import defaultdict
import pandas as pd
from config import CSV_DIR, CATEGORIES
from pathlib import Path


class CategoryStatistics:
    """
    A class for computing statistics for the 10 different categories in our dataset
    """

    def __init__(self, split_name: str):
        """
        Initializes instances of the class for a particular split

        :param split_name: The name of the split, i.e., train or val
        :type split_name: str
        """
        self.split = split_name
        self.stats = defaultdict(lambda: [0, 0, 0, 0, 0, float("inf"), 0, 0, 0, 0])
        self.stats_columns = [
            "total_count",
            "occluded",
            "truncated",
            "total_area",
            "max_area",
            "min_area",
            "anomalies",
            "small",
            "medium",
            "large",
        ]
        self.anomalies = []
        self.records = {
            "category": [],
            "area": [],
            "occluded": [],
            "truncated": [],
            "size_group": [],
        }

    def increment_category_total_count(self, category: str) -> None:
        """
        Increments the total count of the category by 1

        :param category: One of the 10 categories
        :type category: str
        """
        self.stats[category][0] += 1

    def increment_category_occluded_count(self, category: str) -> None:
        """
        Increments the occluded count of the category by 1

        :param category: One of the 10 categories
        :type category: str
        """
        self.stats[category][1] += 1

    def increment_category_truncated_count(self, category: str) -> None:
        """
        Increments the truncated count of the category by 1

        :param category: One of the 10 categories
        :type category: str
        """
        self.stats[category][2] += 1

    def add_category_total_area(self, category: str, area: float) -> None:
        """
        Updates the total area of the category. Useless stat, actually.

        :param self: Description
        :param category: One of the 10 categories
        :type category: str
        :param area: Area of the bounding box of the argument label
        :type area: float
        """
        self.stats[category][3] += area

    def update_category_max_area(self, category: str, area: float) -> None:
        """
        Updates the record for the max area of the category

        :param category: One of the 10 categories
        :type category: str
        :param area: The area of the bounding box of the argument label
        :type area: float
        """
        self.stats[category][4] = max(self.stats[category][4], area)

    def update_category_min_area(self, category: str, area: float) -> None:
        """
        Updates the record for the min area of the category

        :param category: One of the 10 categories
        :type category: str
        :param area: The area of the bounding box of the argument label
        :type area: float
        """
        self.stats[category][5] = min(self.stats[category][5], area)

    def update_category_anomaly_count(
        self,
        category: str,
        area: float,
        aspect_ratio: float,
        image_name: str,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
    ) -> None:
        """
        Checks if the label is an anomaly and updates the anomaly count.
        Also saves the anomaly as a record.

        :param category: One of the 10 categories
        :type category: str
        :param area: The area of the bounding box of the argument label
        :type area: float
        :param aspect_ratio: The aspect_ratio of the bounding box of the argument label
        :type aspect_ratio: float
        :param image_name: The name of the image where the anomaly was found
        :type image_name: str
        :param x1: The x-coordinate of the lower left point of the bounding box of the argument label
        :type aspect_ratio: float
        :param y1: The y-coordinate of the lower left point of the bounding box of the argument label
        :type aspect_ratio: float
        :param x2: The x-coordinate of the upper right point of the bounding box of the argument label
        :type aspect_ratio: float
        :param y2: The y-coordinate of the upper right point of the bounding box of the argument label
        :type aspect_ratio: float
        """
        if area <= 16**2 or area >= (1024 * 576):
            self.stats[category][6] += 1
            self.anomalies.append(
                (
                    image_name,
                    category,
                    0,
                    aspect_ratio,
                    area,
                    x1,
                    y1,
                    x2,
                    y2,
                    self.split,
                )
            )
        if aspect_ratio <= 0.1 or aspect_ratio >= 10:
            self.stats[category][6] += 1
            self.anomalies.append(
                (
                    image_name,
                    category,
                    1,
                    aspect_ratio,
                    area,
                    x1,
                    y1,
                    x2,
                    y2,
                    self.split,
                )
            )

    def categorize_area(self, category: str, area: float) -> None:
        """
        Categorizes a bounding box area as small, medium, or large by COCO standards

        :param category: One of the 10 categories
        :type category: str
        :param area: The area of the bounding box of the category
        :type area: float
        """
        if area < 32**2:
            self.stats[category][7] += 1
        elif area >= 32**2 and area <= 96**2:
            self.stats[category][8] += 1
        else:
            self.stats[category][9] += 1

    def insert_record(
        self, category: str, area: float, occluded: bool, truncated: bool
    ) -> None:
        """
        Create and insert a row into the record dict

        :param self: Description
        :param category: One of the 10 categories
        :type category: str
        :param area: The area of the bounding box of the category
        :type area: float
        :param occluded: Flag denoting whether the label is occluded
        :type occluded: bool
        :param truncated: Flag denoting whether the label is truncated
        :type truncated: bool
        """
        self.records["category"].append(CATEGORIES[category])
        self.records["area"].append(area)
        self.records["occluded"].append(occluded)
        self.records["truncated"].append(truncated)
        if area < 32**2:
            size_group = 0
        elif area >= 32**2 and area <= 96**2:
            size_group = 1
        else:
            size_group = 2
        self.records["size_group"].append(size_group)

    def save_stats_csv(self) -> None:
        """
        Saves the stats dict as a csv file
        """
        stats_df = pd.DataFrame.from_dict(
            self.stats, orient="index", columns=self.stats_columns
        ).reset_index(names="class")
        output_dir = Path(CSV_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"category_stats_{self.split}.csv"
        stats_df.to_csv(output_path, index=False)

    def save_records_csv(self) -> None:
        """
        Saves the records dict as a csv file
        """
        records_df = pd.DataFrame(self.records)
        output_dir = Path(CSV_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"records_{self.split}.csv"
        records_df.to_csv(output_path, index=False)

    def save_anomalies(self) -> None:
        """
        Saves the anomalies as a csv
        """
        df = pd.DataFrame(
            self.anomalies,
            columns=[
                "image_name",
                "category",
                "type",
                "aspect_ratio",
                "area",
                "x1",
                "y1",
                "x2",
                "y2",
                "split",
            ],
        )
        df.to_csv(Path(CSV_DIR) / f"anomalies_{self.split}.csv", index=False)
