REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_differ import EpisodeRunner
REGISTRY["episode_differ"] = EpisodeRunner

from .episode_runner_inspire import EpisodeRunner
REGISTRY["episode_inspire"] = EpisodeRunner

from .parallel_runner_ices import ParallelRunner
REGISTRY["parallel_ices"] = ParallelRunner

from .episode_runner_ices import EpisodeRunner
REGISTRY["episode_ices"] = EpisodeRunner