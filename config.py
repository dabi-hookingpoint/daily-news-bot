from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Audience(BaseModel):
    누구: str
    이미_아는_것: str = ""


class Topic(BaseModel):
    이름: str
    색상: str = "#5F7476"
    데스크지침: str = ""


class Config(BaseModel):
    독자: Audience
    중요도_기준: list[str] = Field(min_length=1)
    버릴_것: list[str] = Field(default_factory=list)
    토픽: list[Topic] = Field(default_factory=list)


def load_config(path: str | Path = "audience.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(data)
