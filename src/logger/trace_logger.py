import logging
from collections import defaultdict
from itertools import zip_longest
from logging import Logger
from typing import Tuple

import torch


class TraceLogger:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._stats = defaultdict(lambda: [])
        self._initialize_placeholders()

    def _initialize_placeholders(self):
        """Initialize placeholders for stats."""
        self._stats["eval_score_mean"].append((0, 0.0))
        self._stats["eval_score_running_mean"].append((0, 0.0))
        self._stats["eval_won_battles_mean"].append((0, 0.0))
        self._stats["eval_score_std"].append((0, float("inf")))
        self._stats["eval_score_var"].append((0, float("inf")))

    def log_stat(self, stat: str, value: Tuple[int, float], episode: int) -> None:
        self._stats[stat].append((episode, value))

    def display_recent_stats(self) -> None:
        if not self._stats or all(len(values) == 0 for values in self._stats.values()):
            self._logger.info("No stats to display.")
            return

        # Create the message about displaying stats
        display_message = "Displaying Recent Stats:"

        # Determine the maximum length of stat names for formatting
        max_stat_len = max((len(stat) for stat in self._stats.keys()), default=0)

        # Prepare the header and separator lines
        header = f"| {'Stat'.ljust(max_stat_len)} | Episode   | Value      |"
        separator = "-" * len(header)

        # Start with the display message, header and separator
        lines = [display_message, separator, header, separator]

        # Collect the lines for each stat, ensuring each stat is listed only once
        for stat in sorted(self._stats.keys()):
            latest_episode, latest_value = (
                self._stats[stat][-1] if self._stats[stat] else (0, 0.0)
            )
            if isinstance(latest_value, torch.Tensor):
                latest_value = latest_value.item()  # Convert tensors to numbers
            value_str = (
                f"{latest_value:>10.4f}"
                if isinstance(latest_value, (float, int))
                else str(latest_value)
            )
            lines.append(
                f"| {stat.ljust(max_stat_len)} | {str(latest_episode).rjust(8)} | {value_str:>10} |"
            )

        # Add the closing separator
        lines.append(separator)

        # Combine all lines into a single string and log it
        output = "\n".join(lines)
        self._logger.info(output)
