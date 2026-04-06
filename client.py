from typing import Dict, Any

from openenv.core.env_client import EnvClient
from models import ResumeObservation, ResumeAction, ResumeState


class ResumeEnv(EnvClient[ResumeObservation, ResumeAction, ResumeState]):
    """
    Client-side interface for interacting with the Resume Screening Environment.
    """

    def _step_payload(self, action: ResumeAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, response: Dict[str, Any]) -> ResumeObservation:
        data = response.copy()
        if "observation" in data and isinstance(data["observation"], dict):
            obs_data = data.pop("observation")
            data.update(obs_data)
        if data.get("reward") is None:
            data["reward"] = 0.0
        if data.get("done") is None:
            data["done"] = False
        if "visible_sections" not in data:
            data["visible_sections"] = {}
        if "available_actions" not in data:
            data["available_actions"] = []
        return ResumeObservation(**data)

    def _parse_state(self, response: Dict[str, Any]) -> ResumeState:
        return ResumeState(**response)

    def _parse_reset(self, response: Dict[str, Any]) -> ResumeObservation:
        return self._parse_result(response)
