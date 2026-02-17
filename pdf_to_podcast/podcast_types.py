# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic import BaseModel
from typing import Optional, Dict, Literal, List

DEFAULT_SPEAKER_1_NAME = "Alice"
DEFAULT_SPEAKER_2_NAME = "Will"


class SavedPodcast(BaseModel):
    job_id: str
    filename: str
    created_at: str
    size: int
    transcription_params: Optional[Dict] = {}


class SavedPodcastWithAudio(SavedPodcast):
    audio_data: str


class DialogueEntry(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]


class Conversation(BaseModel):
    scratchpad: str
    dialogue: List[DialogueEntry]


class SegmentPoint(BaseModel):
    description: str


class SegmentTopic(BaseModel):
    title: str
    points: List[SegmentPoint]


class PodcastSegment(BaseModel):
    section: str
    topics: List[SegmentTopic]
    duration: int
    references: List[str]


class PodcastOutline(BaseModel):
    title: str
    segments: List[PodcastSegment]


class PodcastMetadata(BaseModel):
    """Typed metadata contract for podcast pipeline items.

    Every pipeline step reads and writes through this model to ensure
    consistent metadata across the entire flow. Stored under
    item.metadata["user"]["podcast"].
    """

    pdf_name: str
    pipeline_stage: Optional[str] = None
    monologue: bool = False
    focus: Optional[str] = None
    with_references: bool = False
    speaker_1_name: str = DEFAULT_SPEAKER_1_NAME
    speaker_2_name: str = DEFAULT_SPEAKER_2_NAME
    duration: int = 10
    references: Optional[List[str]] = None
    summary_item_id: Optional[str] = None
    outline_item_id: Optional[str] = None
    pdf_id: Optional[str] = None
    segment_idx: Optional[int] = None
    total_segments: Optional[int] = None
    segment_topic: Optional[str] = None
    topics: Optional[List[str]] = None

    def validate_stage(self, expected: str) -> None:
        """Validate this metadata belongs to the expected pipeline stage."""
        if self.pipeline_stage is not None and self.pipeline_stage != expected:
            raise ValueError(
                f"Expected item at pipeline stage '{expected}', "
                f"got '{self.pipeline_stage}'. Wrong item may have been passed."
            )

    def to_item_metadata(self, **extra_user_fields) -> dict:
        """Serialize to Dataloop item metadata format: {\"user\": {\"podcast\": {...}, ...extra}}"""
        user_meta = {"podcast": self.model_dump()}
        user_meta.update(extra_user_fields)
        return {"user": user_meta}
