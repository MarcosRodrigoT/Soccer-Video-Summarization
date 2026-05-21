"""Configuration management via YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ShotSegmentationConfig:
    """Matlab-equivalent shot segmentation parameters.

    Histogram-based abrupt + gradual transition detection. See
    soccer_summarization/shot_segmentation.py for the algorithm.
    """
    num_processes: Optional[int] = None  # None = use all CPUs

    # Abrupt transition detection
    abrupt_threshold: float = 0.3
    abrupt_window_size: int = 15
    abrupt_neighborhood_factor: float = 0.5

    # Gradual transition detection
    gradual_filter_window: int = 15
    gradual_min_peak_height: float = 0.03
    gradual_min_peak_prominence: float = 0.02
    gradual_local_threshold_factor: float = 1.2
    gradual_local_window_ratio: int = 3
    gradual_min_duration: int = 15
    gradual_min_distance_from_abrupt: int = 25
    gradual_min_contrast_factor: float = 1.0


@dataclass
class ObjectDetectionConfig:
    model_path: str = "data/models/best.pt"
    confidence_threshold: float = 0.5
    # Model classes: ball=0, goalkeeper=1, player=2, referee=3
    # People = {1,2,3} (all except ball)
    frame_step: int = 5  # process every Nth frame (5 = 5fps at 25fps video)


@dataclass
class PlayClassifierConfig:
    play_threshold: int = 5  # min players for "play"
    smoothing_window: int = 50  # rolling average window in frames
    min_shot_duration: float = 4.0  # seconds


@dataclass
class OCRScoreboardConfig:
    num_sample_frames: int = 200  # frames for variability map
    std_percentile: float = 8.0  # bottom percentile of variability → static overlay pixels
    min_region_area: int = 300  # min pixels for a candidate overlay region
    closing_kernel: tuple = (51, 21)  # morphological closing kernel (w, h)
    score_sample_interval: float = 3.0  # seconds between OCR samples
    score_pattern: str = r"\d+\s*[-:]\s*\d+"
    majority_window: int = 5  # frames for temporal majority voting


@dataclass
class HighlightAssemblerConfig:
    min_shot_duration: float = 4.0  # seconds
    merge_gap_threshold: float = 0.5  # seconds - merge segments closer than this
    goal_context_before: float = 30.0  # seconds before goal to include
    goal_context_after: float = 15.0  # seconds after goal to include


@dataclass
class EvaluationConfig:
    tolerance_window: float = 30.0  # seconds for goal detection matching
    goal_gap_threshold: float = 30.0  # seconds gap to separate GT goal events


@dataclass
class PipelineConfig:
    shot_segmentation: ShotSegmentationConfig = field(default_factory=ShotSegmentationConfig)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    play_classifier: PlayClassifierConfig = field(default_factory=PlayClassifierConfig)
    ocr_scoreboard: OCRScoreboardConfig = field(default_factory=OCRScoreboardConfig)
    highlight_assembler: HighlightAssemblerConfig = field(default_factory=HighlightAssemblerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Global paths
    data_dir: str = "/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos"
    results_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file, merging with defaults."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Map nested sections to sub-configs
        section_map = {
            "shot_segmentation": (config.shot_segmentation, ShotSegmentationConfig),
            "object_detection": (config.object_detection, ObjectDetectionConfig),
            "play_classifier": (config.play_classifier, PlayClassifierConfig),
            "ocr_scoreboard": (config.ocr_scoreboard, OCRScoreboardConfig),
            "highlight_assembler": (config.highlight_assembler, HighlightAssemblerConfig),
            "evaluation": (config.evaluation, EvaluationConfig),
        }

        for section_name, (sub_config, _) in section_map.items():
            if section_name in data:
                for key, value in data[section_name].items():
                    if hasattr(sub_config, key):
                        setattr(sub_config, key, value)

        # Global settings
        for key in ("data_dir", "results_dir"):
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict

        data = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
