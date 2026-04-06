from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Dict, List, Literal, Optional


class ResumeObservation(Observation):
    """
    Observation provided to the agent at each step of a resume screening episode.
    The agent starts with the job description and header, then reveals more info
    through investigation actions before making a final decision.
    """
    task_type: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level of the current task."
    )
    phase: Literal["initial", "investigation", "decision_made"] = Field(
        "initial", description="Current phase of the episode."
    )
    job_description: str = Field(
        "", description="The job description to evaluate the resume against."
    )
    visible_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Resume sections revealed so far (section_name -> content)."
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types the agent can take next."
    )
    clarification_response: Optional[str] = Field(
        None, description="Response to the last clarification question."
    )
    reference_response: Optional[str] = Field(
        None, description="Result of the last reference check."
    )
    verification_result: Optional[str] = Field(
        None, description="Result of the last credential verification."
    )
    steps_remaining: int = Field(
        0, description="Number of steps left in the episode budget."
    )
    feedback: Optional[str] = Field(
        None, description="Environment feedback or warnings."
    )


class ResumeAction(Action):
    """
    Multi-type action for resume investigation and decision-making.
    The agent chooses an action_type and provides the relevant fields.
    """
    action_type: Literal[
        "view_section",
        "ask_clarification",
        "check_reference",
        "verify_credential",
        "submit_decision"
    ] = Field(..., description="Type of action to take.")

    # For view_section
    section: Optional[str] = Field(
        None,
        description="Section to view: header, summary, experience, education, skills, projects, references"
    )

    # For ask_clarification
    question: Optional[str] = Field(
        None, description="Clarification question to ask the candidate."
    )

    # For check_reference
    reference_id: Optional[str] = Field(
        None, description="Reference to contact: ref1 or ref2."
    )

    # For submit_decision
    decision: Optional[Literal["accept", "reject"]] = Field(
        None, description="Final hiring decision."
    )
    fraud_flag: Optional[bool] = Field(
        None, description="Whether the resume is flagged as fraudulent."
    )
    confidence: Optional[float] = Field(
        None, description="Confidence in the decision (0.0 to 1.0)."
    )
    fraud_reasoning: Optional[str] = Field(
        None, description="Explanation if fraud is suspected."
    )


class ResumeState(State):
    """
    Internal state of the environment for tracking episode progress.
    """
    current_index: int = Field(0, description="Index of the current resume.")
    task_type: str = Field("easy", description="Difficulty level.")
    max_steps: int = Field(8, description="Maximum steps allowed.")
    sections_viewed: List[str] = Field(
        default_factory=list, description="Sections the agent has viewed."
    )
    clarifications_asked: int = Field(
        0, description="Number of clarification questions asked."
    )
    references_checked: int = Field(
        0, description="Number of references contacted."
    )
    verifications_done: int = Field(
        0, description="Number of credential verifications performed."
    )
    investigation_score: float = Field(
        0.0, description="Running partial reward from investigation steps."
    )


# Rebuild models for Pydantic v2 compatibility
ResumeObservation.model_rebuild()
ResumeAction.model_rebuild()
ResumeState.model_rebuild()
