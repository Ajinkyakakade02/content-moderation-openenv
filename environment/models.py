from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    REEL = "reel"

class ModerationAction(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    REMOVE = "REMOVE"

class ViolationCategory(str, Enum):
    NUDITY = "nudity"
    SEXUAL = "sexual"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    HARASSMENT = "harassment"
    DANGEROUS = "dangerous"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    NONE = "none"

class Observation(BaseModel):
    """OpenEnv Observation model"""
    text: str = Field(..., max_length=1000, description="Content text")
    user_reputation: float = Field(ge=0, le=1, description="User trust score")
    report_count: int = Field(ge=0, le=100, description="Number of reports")
    policy_summary: str = Field(..., description="Current policy guidelines")
    content_type: ContentType = Field(default=ContentType.TEXT)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v

class Action(BaseModel):
    """OpenEnv Action model"""
    action: ModerationAction = Field(..., description="Moderation decision")
    confidence: Optional[float] = Field(ge=0, le=1, default=0.5)
    reasoning: Optional[str] = Field(default=None, max_length=200)

class Reward(BaseModel):
    """OpenEnv Reward model"""
    value: float = Field(..., description="Reward value")
    components: Dict[str, float] = Field(default_factory=dict)
    explanation: str = Field(default="")

class ModerationResult(BaseModel):
    """Complete moderation result"""
    id: str
    action: ModerationAction
    confidence: float
    violations: List[ViolationCategory]
    reason: str
    policy_reference: Optional[str]
    processing_time_ms: int
    timestamp: datetime

class ContentItem(BaseModel):
    """Content to moderate"""
    id: str
    text: str
    user_reputation: float
    report_count: int
    true_label: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    metadata: Dict[str, Any] = Field(default_factory=dict)