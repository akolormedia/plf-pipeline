from dataclasses import dataclass, field


@dataclass
class ShotBoundary:
    i: float
    o: float
    confidence: float = 1.0


@dataclass
class ShotFrame:
    shot_index: int
    i: float
    o: float
    frame_paths: list[str] = field(default_factory=list)


@dataclass
class VisionOutput:
    shot_index: int
    shot_type: str = ""
    cam: str = ""
    mood: str = ""
    act: str = ""
    arc: str = ""
    location: str = ""
    elements: list[dict] = field(default_factory=list)
    description: str = ""


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker_id: str = ""
    confidence: float = 1.0


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str = "es"


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker_id: str
    speaker_label: str = ""


@dataclass
class AudioEvent:
    start: float
    end: float
    source: str
    diegetic: str = "D"
    level: int = 3
    mood: str = ""
    event_type: str = "ambient"


@dataclass
class AudioClassification:
    music_events: list[AudioEvent] = field(default_factory=list)
    ambient_events: list[AudioEvent] = field(default_factory=list)


@dataclass
class UserContext:
    description: str = ""
    characters: list[dict] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    language: str = "es"
    genre: str = "drama"
    fps: int = 24
    aspect_ratio: str = "16:9"


@dataclass
class PipelineResult:
    raw_plf_text: str = ""
    shots: list[ShotBoundary] = field(default_factory=list)
    vision: list[VisionOutput] = field(default_factory=list)
    transcript: Transcript = field(default_factory=Transcript)
    audio: AudioClassification = field(default_factory=AudioClassification)
    errors: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.shots:
            return self.shots[-1].o
        return 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.raw_plf_text) > 0
