from collections import defaultdict
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
        self._stats["timesteps_passed"].append((0, 0))

        self._stats["eval_score_mean"].append((0, 0.0))
        self._stats["eval_score_running_mean"].append((0, 0.0))
        self._stats["eval_won_battles_mean"].append((0, 0.0))

        self._stats["eval_most_won_battles"].append((0, 0.0))
        self._stats["eval_mean_higest_score"].append((0, 0.0))

        self._stats["eval_score_std"].append((0, 0.0))
        self._stats["eval_score_var"].append((0, 0.0))

    def log_stat(self, stat: str, value: Tuple[int, float], episode: int) -> None:
        self._stats[stat].append((episode, value))

    def display_recent_stats(self) -> None:
        if not self._stats or all(len(values) == 0 for values in self._stats.values()):
            self._logger.info("No stats to display.")
            return

        display_message = "Displaying Recent Stats:"
        max_stat_len = max((len(stat) for stat in self._stats.keys()), default=0)
        header = f"| {'Stat'.ljust(max_stat_len)} | Episode   | Value      |"
        separator = "-" * len(header)
        lines = [display_message, separator, header, separator]

        for stat in sorted(self._stats.keys()):
            latest_episode, latest_value = (
                self._stats[stat][-1] if self._stats[stat] else (0, 0.0)
            )
            if isinstance(latest_value, torch.Tensor):
                latest_value = latest_value.item()

            if stat == "timesteps_passed":
                value_str = f"{latest_value:>10d}"  # Format as integer
            else:
                value_str = (
                    f"{latest_value:>10.4f}"
                    if isinstance(latest_value, (float, int))
                    else str(latest_value)
                )

            lines.append(
                f"| {stat.ljust(max_stat_len)} | {str(latest_episode).rjust(8)} | {value_str:>11} |"
            )

        lines.append(separator)
        output = "\n".join(lines)
        self._logger.info(output)
