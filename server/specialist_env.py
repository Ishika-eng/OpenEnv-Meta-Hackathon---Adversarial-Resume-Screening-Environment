"""
SpecialistEnvironment — Restricted Action Spaces
=================================================
Day 2 implementation: each specialist agent operates within a declaratively
defined whitelist of allowed actions and sections.  The validator enforces
these constraints at the environment level — invalid actions are REJECTED,
not silently ignored.

Architecture
------------
SpecialistConfig          — dataclass describing a role's full constraint set
SPECIALIST_CONFIGS        — registry mapping role names → configs
SpecialistActionValidator — validates actions + filters observations per role

The FleetResumeEnvironment imports SpecialistActionValidator and calls it on
every step before dispatching to the action handler.  This gives:

  • Hard action whitelist  — submit_decision is illegal for fraud_specialist
  • Hard section whitelist — fraud_specialist cannot view 'experience'
  • Role-filtered views    — each specialist only SEES their sections in obs
  • Violation tracking     — out-of-role attempts counted; penalised at terminal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# SpecialistConfig
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    """Declarative description of one specialist's constraints."""

    role: str

    # Actions this specialist may submit.
    allowed_actions: List[str]

    # Sections this specialist may REQUEST to view (view_section whitelist).
    allowed_sections: List[str]

    # Sections this specialist may SEE in their observation even if revealed
    # by a previous phase (observation filter — typically same as allowed_sections
    # but can be broader if overseer should see a summary).
    observable_sections: List[str]

    # Whether this role has access to investigative tools.
    can_check_reference: bool = False
    can_verify_credential: bool = False
    can_ask_clarification: bool = False

    # Human-readable summary of the role's purpose (shown in feedback messages).
    focus_description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPECIALIST_CONFIGS: Dict[str, SpecialistConfig] = {

    "fraud_specialist": SpecialistConfig(
        role="fraud_specialist",
        allowed_actions=[
            "check_reference",
            "verify_credential",
            "view_section",
            "submit_specialist_report",
        ],
        allowed_sections=["header", "references"],
        observable_sections=["header", "references"],
        can_check_reference=True,
        can_verify_credential=True,
        can_ask_clarification=False,
        focus_description=(
            "Detect credential fraud, fabricated employment, and reference inconsistencies. "
            "Use check_reference and verify_credential. View 'header' and 'references' only."
        ),
    ),

    "skills_specialist": SpecialistConfig(
        role="skills_specialist",
        allowed_actions=[
            "view_section",
            "ask_clarification",
            "submit_specialist_report",
        ],
        allowed_sections=["experience", "education", "skills", "projects"],
        observable_sections=["experience", "education", "skills", "projects"],
        can_check_reference=False,
        can_verify_credential=False,
        can_ask_clarification=True,
        focus_description=(
            "Assess technical fit against job requirements. "
            "View experience, education, skills, projects. Ask clarifying questions."
        ),
    ),

    "timeline_specialist": SpecialistConfig(
        role="timeline_specialist",
        allowed_actions=[
            "view_section",
            "ask_clarification",
            "submit_specialist_report",
        ],
        allowed_sections=["header", "summary", "experience"],
        observable_sections=["header", "summary", "experience"],
        can_check_reference=False,
        can_verify_credential=False,
        can_ask_clarification=True,
        focus_description=(
            "Check chronological consistency and employment gaps. "
            "View header, summary, experience. Ask about career transitions and gaps."
        ),
    ),

    "overseer": SpecialistConfig(
        role="overseer",
        allowed_actions=[
            "request_reinvestigation",
            "submit_final_decision",
        ],
        # Overseer synthesises reports — cannot view sections directly.
        allowed_sections=[],
        # Overseer sees a brief header summary only (for context), not full sections.
        observable_sections=["header"],
        can_check_reference=False,
        can_verify_credential=False,
        can_ask_clarification=False,
        focus_description=(
            "Synthesise the three specialist reports and issue the final hiring verdict. "
            "Use request_reinvestigation (once) if reports conflict. Then submit_final_decision."
        ),
    ),
}


# ---------------------------------------------------------------------------
# SpecialistActionValidator
# ---------------------------------------------------------------------------

class SpecialistActionValidator:
    """
    Validates a FleetAction against the active specialist's config.

    Usage (inside FleetResumeEnvironment.step):

        validator = SpecialistActionValidator(SPECIALIST_CONFIGS[current_phase])
        valid, reason = validator.validate(action)
        if not valid:
            return self._violation_obs(reason)   # zero reward + rejection message
        # ... proceed with action handler
    """

    def __init__(self, config: SpecialistConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

    def validate(self, action) -> Tuple[bool, str]:
        """
        Return (True, '') if the action is within this specialist's whitelist.
        Return (False, reason) if the action violates a constraint.
        """
        at = action.action_type

        # 1. Action-type whitelist
        if at not in self.config.allowed_actions:
            return False, (
                f"[VIOLATION] {self.config.role} cannot use '{at}'. "
                f"Allowed: {self.config.allowed_actions}. "
                f"This step costs you a step and earns 0 reward."
            )

        # 2. Section whitelist (only for view_section)
        if at == "view_section":
            section = (action.section or "").lower().strip()
            if section not in self.config.allowed_sections:
                allowed = self.config.allowed_sections or ["(none)"]
                return False, (
                    f"[VIOLATION] {self.config.role} cannot view section '{section}'. "
                    f"Allowed sections: {allowed}. "
                    f"This step costs you a step and earns 0 reward."
                )

        # 3. Tool-specific checks
        if at == "check_reference" and not self.config.can_check_reference:
            return False, (
                f"[VIOLATION] {self.config.role} does not have reference-check access."
            )

        if at == "verify_credential" and not self.config.can_verify_credential:
            return False, (
                f"[VIOLATION] {self.config.role} does not have credential-verification access."
            )

        if at == "ask_clarification" and not self.config.can_ask_clarification:
            return False, (
                f"[VIOLATION] {self.config.role} cannot ask clarification questions."
            )

        return True, ""

    # ------------------------------------------------------------------
    # Observation filtering
    # ------------------------------------------------------------------

    def filter_sections(self, all_viewed_sections: Dict[str, str]) -> Dict[str, str]:
        """
        Return only the sections this specialist is ALLOWED to observe.
        Sections viewed by other specialists in previous phases are hidden.
        """
        if not self.config.observable_sections:
            return {}
        return {
            k: v for k, v in all_viewed_sections.items()
            if k in self.config.observable_sections
        }

    # ------------------------------------------------------------------
    # Dynamic available-actions list
    # ------------------------------------------------------------------

    def available_actions(
        self,
        sections_viewed: List[str],
        references_checked: int,
        verifications_done: int,
        reinvestigation_used: bool = False,
    ) -> List[str]:
        """
        Build the list of currently executable actions for this specialist,
        respecting both the whitelist and per-episode state (e.g. don't offer
        check_reference if already checked).
        """
        result = []
        for at in self.config.allowed_actions:

            if at == "view_section":
                unseen = set(self.config.allowed_sections) - set(sections_viewed)
                if unseen:
                    result.append("view_section")

            elif at == "check_reference":
                if self.config.can_check_reference and references_checked == 0:
                    result.append("check_reference")

            elif at == "verify_credential":
                if self.config.can_verify_credential and verifications_done == 0:
                    result.append("verify_credential")

            elif at == "ask_clarification":
                if self.config.can_ask_clarification:
                    result.append("ask_clarification")

            elif at == "request_reinvestigation":
                if not reinvestigation_used:
                    result.append("request_reinvestigation")

            else:
                # submit_specialist_report, submit_final_decision — always available
                result.append(at)

        return result

    # ------------------------------------------------------------------
    # Role instructions (enriched with constraint reminders)
    # ------------------------------------------------------------------

    def role_instructions(self) -> str:
        """Return role instructions with explicit constraint reminders."""
        base = ROLE_INSTRUCTIONS_TEMPLATE.get(self.config.role, "")
        if self.config.allowed_sections:
            base += (
                f"\n\nCONSTRAINTS: You may ONLY view sections: {self.config.allowed_sections}. "
                f"Attempting to view other sections will be REJECTED and waste your step budget."
            )
        if self.config.allowed_actions:
            base += (
                f"\nAllowed actions: {self.config.allowed_actions}. "
                f"Any other action type will be REJECTED."
            )
        return base


# ---------------------------------------------------------------------------
# Role instructions templates
# ---------------------------------------------------------------------------

ROLE_INSTRUCTIONS_TEMPLATE: Dict[str, str] = {
    "fraud_specialist": (
        "You are the FRAUD SPECIALIST. Detect fraudulent or exaggerated claims.\n"
        "Your tools: check_reference (call ref1 or ref2), verify_credential.\n"
        "Your sections: 'header' (identity check) and 'references' (reference names).\n"
        "After investigating, submit your report: submit_specialist_report with "
        "findings (string), has_issues (bool), specialist_confidence (float 0-1)."
    ),
    "skills_specialist": (
        "You are the SKILLS SPECIALIST. Assess whether the candidate's skills match the job.\n"
        "Your sections: 'experience', 'education', 'skills', 'projects'.\n"
        "You may ask_clarification for technical details.\n"
        "After assessing, submit: submit_specialist_report with "
        "findings, has_issues (true if skills mismatch), specialist_confidence."
    ),
    "timeline_specialist": (
        "You are the TIMELINE SPECIALIST. Check chronological consistency and career gaps.\n"
        "Your sections: 'header', 'summary', 'experience'.\n"
        "You may ask_clarification about employment gaps or overlaps.\n"
        "After checking, submit: submit_specialist_report with "
        "findings, has_issues (true if timeline inconsistencies found), specialist_confidence."
    ),
    "overseer": (
        "You are the OVERSEER. Read the specialist_reports from all three agents.\n"
        "You CANNOT view sections or use investigative tools directly.\n"
        "You may use request_reinvestigation ONCE if reports conflict or are incomplete.\n"
        "Submit your verdict: submit_final_decision with "
        "decision (accept|reject), fraud_flag (bool), confidence (0-1), fraud_reasoning."
    ),
}


# ---------------------------------------------------------------------------
# Terminal violation penalty calculator
# ---------------------------------------------------------------------------

def compute_violation_penalty(violations: int) -> float:
    """
    Compute the terminal reward deduction for out-of-role violations.
    Penalty = 0.05 per violation, capped at 0.25 total.
    The caller must ensure final_reward never goes below 0.0.
    """
    return round(min(violations * 0.05, 0.25), 4)
