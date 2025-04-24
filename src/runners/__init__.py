REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_differ import EpisodeRunner
REGISTRY["episode_differ"] = EpisodeRunner

from .episode_runner_inspire import EpisodeRunner
REGISTRY["episode_inspire"] = EpisodeRunner