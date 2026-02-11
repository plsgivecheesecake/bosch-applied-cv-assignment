"""
File containing the scene statistics class
"""

from collections import defaultdict
import pandas as pd
from pathlib import Path
from config import CSV_DIR, CATEGORIES


class SceneStatistics:
    """
    A class for computing the scene level statistics of the BDD100K dataset
    """

    def __init__(self, split_name: str) -> None:
        """
        Initializes an instance of the class for a particular split

        :param self: Description
        :param split_name: One of the two splits
        :type split_name: str
        """
        self.split = split_name
        self.stats = defaultdict(lambda: defaultdict(int))
        self.category_distribution = defaultdict(lambda: defaultdict(int))

    def update_stats_counter(self, key: str, value: str) -> None:
        """
        Update one of the three stats: weather, scene, timeofday by 1

        :param key: attribute name
        :type key: str
        :param value: class name
        :type value: str
        """
        self.stats[key][value] += 1

    def update_category_distribution(
        self, weather: str, timeofday: str, category: str
    ) -> None:
        """
        Update the category distribution by weather and time of day

        :param weather: The weather condition of the scene
        :type weather: str
        :param timeofday: The time of day of the image setting
        :type timeofday: str
        :param category: One of the 10 classes
        :type category: str
        """
        self.category_distribution[(timeofday, weather)][category] += 1

    def save_csvs(self) -> None:
        """
        Saves csv files of statistics to the output directory
        """
        output_dir = Path(CSV_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        for stat_name, counts in self.stats.items():
            output_path = output_dir / f"{stat_name}_{self.split}.csv"
            stat_df = pd.DataFrame(counts.items(), columns=["attribute", "count"])
            stat_df.to_csv(output_path, index=False)

        return f"csvs were written to: {CSV_DIR}"

    def save_category_distribution_csv(self) -> None:
        """
        Saves the distribution of categories by weather and time of day to the output directory
        """
        output_dir = Path(CSV_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timeofday_list = sorted(["daytime", "dawn/dusk", "night", "undefined"])
        weather_list = sorted(
            [
                "clear",
                "rainy",
                "undefined",
                "snowy",
                "overcast",
                "partly cloudy",
                "foggy",
            ]
        )
        rows = []

        for time_of_day in timeofday_list:
            for weather in weather_list:
                for category in CATEGORIES.keys():
                    rows.append(
                        (
                            time_of_day,
                            weather,
                            category,
                            self.category_distribution[(time_of_day, weather)][
                                category
                            ],
                        )
                    )
        category_distribution_df = pd.DataFrame(
            rows, columns=["weather", "time", "class", "value"]
        )
        output_path = output_dir / f"categories_by_scene_params_{self.split}.csv"
        category_distribution_df.to_csv(output_path, index=False)

        return f"csvs were written to: {CSV_DIR}"
