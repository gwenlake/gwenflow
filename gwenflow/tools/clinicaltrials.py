from dataclasses import dataclass
from typing import Any, Optional

import requests
from pydantic import Field

from gwenflow.tools.tool import BaseTool

CLINICALTRIALS_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CLINICALTRIALS_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


@dataclass(kw_only=True)
class ClinicalTrialsBase(BaseTool):
    top_k_results: int = 5
    timeout: int = 30
    fields: Optional[list[str]] = None

    @staticmethod
    def _get(d: dict, *path: str, default: Any = None) -> Any:
        cur: Any = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur if cur is not None else default

    def _parse_study(self, study: dict[str, Any]) -> dict[str, Any]:
        protocol = study.get("protocolSection", {}) or {}
        identification = protocol.get("identificationModule", {}) or {}
        status = protocol.get("statusModule", {}) or {}
        description = protocol.get("descriptionModule", {}) or {}
        conditions = protocol.get("conditionsModule", {}) or {}
        design = protocol.get("designModule", {}) or {}
        sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}
        eligibility = protocol.get("eligibilityModule", {}) or {}

        nct_id = identification.get("nctId", "")

        return {
            "nct_id": nct_id,
            "title": identification.get("briefTitle", ""),
            "official_title": identification.get("officialTitle", ""),
            "status": status.get("overallStatus", ""),
            "phase": ", ".join(design.get("phases", []) or []),
            "study_type": design.get("studyType", ""),
            "conditions": conditions.get("conditions", []) or [],
            "summary": description.get("briefSummary", ""),
            "lead_sponsor": self._get(sponsor, "leadSponsor", "name", default=""),
            "start_date": self._get(status, "startDateStruct", "date", default=""),
            "completion_date": self._get(status, "completionDateStruct", "date", default=""),
            "enrollment": self._get(design, "enrollmentInfo", "count", default=None),
            "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
            "url": CLINICALTRIALS_STUDY_URL.format(nct_id=nct_id) if nct_id else "",
        }


@dataclass(kw_only=True)
class ClinicalTrialsTool(ClinicalTrialsBase):
    name: str = "ClinicalTrialsTool"
    description: str = (
        "Search ClinicalTrials.gov for registered clinical studies. "
        "Returns NCT id, title, status, phase, conditions, sponsor, dates, "
        "enrollment, eligibility, and a clickable study URL. "
        "Input should be a search query (e.g. 'metformin diabetes', "
        "'CAR-T lymphoma phase 3', or an NCT id)."
    )

    def _run(
        self,
        query: str = Field(description="Search query for ClinicalTrials.gov"),
        status: str = Field(
            default="",
            description=(
                "Optional study status filter, e.g. RECRUITING, COMPLETED, "
                "ACTIVE_NOT_RECRUITING, NOT_YET_RECRUITING. Leave empty for all."
            ),
        ),
    ):
        params: dict[str, Any] = {
            "query.term": query,
            "pageSize": self.top_k_results,
            "format": "json",
        }
        if isinstance(status, str) and status.strip():
            params["filter.overallStatus"] = status.strip()

        response = requests.get(CLINICALTRIALS_API_BASE, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", []) or []
        if not studies:
            raise ValueError(f"No ClinicalTrials.gov results found for query: {query!r}")

        return [self._parse_study(study) for study in studies]
