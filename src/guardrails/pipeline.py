"""Guardrail Pipeline - orchestrates all guardrails."""

import logging

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailPipelineResult, GuardrailResult, GuardrailSeverity
from .contextual import ContextualCoherenceGuardrail
from .entity import EntityVerificationGuardrail
from .factual_grounding import FactualGroundingGuardrail
from .numerical import NumericalConsistencyGuardrail
from .temporal import TemporalConsistencyGuardrail

logger = logging.getLogger(__name__)

# Default max retries for failed guardrails
DEFAULT_MAX_RETRIES = 2


class GuardrailPipeline:
    """
    Orchestrates all guardrails in a pipeline.

    Guardrails are run in order of severity:
    1. Critical guardrails (block on failure)
    2. High guardrails (block on failure)
    3. Medium guardrails (warn, allow with flag)
    4. Low guardrails (log only)
    """

    def __init__(self, guardrails: list[BaseGuardrail] = None):
        """
        Initialize the pipeline with guardrails.

        Args:
            guardrails: List of guardrails to run. If None, uses default set.
        """
        if guardrails is None:
            self.guardrails = self._get_default_guardrails()
        else:
            self.guardrails = guardrails

        # Sort by severity (critical first)
        severity_order = {
            GuardrailSeverity.CRITICAL: 0,
            GuardrailSeverity.HIGH: 1,
            GuardrailSeverity.MEDIUM: 2,
            GuardrailSeverity.LOW: 3,
        }
        self.guardrails.sort(key=lambda g: severity_order[g.severity])

        logger.info(
            f"Initialized guardrail pipeline with {len(self.guardrails)} guardrails: "
            f"{[g.name for g in self.guardrails]}"
        )

    def _get_default_guardrails(self) -> list[BaseGuardrail]:
        """Get the default set of guardrails."""
        return [
            FactualGroundingGuardrail(),
            NumericalConsistencyGuardrail(),
            EntityVerificationGuardrail(),
            TemporalConsistencyGuardrail(),
            ContextualCoherenceGuardrail(),
        ]

    def validate(
        self,
        output: str,
        context: CaseContext,
        fail_fast: bool = True,
        **kwargs,
    ) -> GuardrailPipelineResult:
        """
        Run all guardrails on the output.

        Args:
            output: The AI-generated output to validate
            context: The input context
            fail_fast: If True, stop on first critical failure
            **kwargs: Additional context (extracted_data, etc.)

        Returns:
            GuardrailPipelineResult with all validation results
        """
        all_results: list[GuardrailResult] = []
        blocking_guardrails: list[str] = []
        should_block = False

        for guardrail in self.guardrails:
            try:
                results = guardrail.validate(output, context, **kwargs)
                all_results.extend(results)

                # Check for blocking failures
                for result in results:
                    if result.should_block:
                        should_block = True
                        blocking_guardrails.append(guardrail.name)

                        logger.warning(
                            f"Guardrail {guardrail.name} BLOCKED output: {result.message}"
                        )

                        if fail_fast and result.severity == GuardrailSeverity.CRITICAL:
                            # Stop immediately on critical failure
                            return GuardrailPipelineResult(
                                all_passed=False,
                                should_block=True,
                                results=all_results,
                                retry_suggested=True,
                                blocking_guardrails=blocking_guardrails,
                            )

            except Exception as e:
                logger.error(f"Guardrail {guardrail.name} raised exception: {e}")
                # Create a failure result for the exception
                all_results.append(
                    GuardrailResult(
                        passed=False,
                        guardrail_name=guardrail.name,
                        severity=GuardrailSeverity.HIGH,
                        message=f"Guardrail execution error: {str(e)}",
                        details={"exception": str(e)},
                    )
                )
                should_block = True
                blocking_guardrails.append(guardrail.name)

        all_passed = all(r.passed for r in all_results)

        return GuardrailPipelineResult(
            all_passed=all_passed,
            should_block=should_block,
            results=all_results,
            retry_suggested=should_block and len(blocking_guardrails) <= 2,
            blocking_guardrails=blocking_guardrails,
        )

    def get_retry_prompt_addition(self, pipeline_result: GuardrailPipelineResult) -> str:
        """
        Generate additional prompt instructions based on guardrail failures.

        Used when retrying after a guardrail failure.
        """
        if pipeline_result.all_passed:
            return ""

        additions = [
            "\n\n**IMPORTANT VALIDATION REQUIREMENTS:**",
            "The previous response had validation errors. Please ensure:",
        ]

        for result in pipeline_result.results:
            if not result.passed:
                if result.guardrail_name == "factual_grounding":
                    additions.append(
                        "- ONLY use invoice numbers and amounts from the provided context"
                    )
                    if result.expected:
                        additions.append(f"- Valid invoices: {result.expected}")
                elif result.guardrail_name == "numerical_consistency":
                    additions.append("- Verify all calculations match (totals = sum of parts)")
                    if result.details.get("calculated_total"):
                        additions.append(f"- Correct total: {result.details['calculated_total']}")
                elif result.guardrail_name == "entity_verification":
                    additions.append("- Use exact customer code and company name from context")
                    if result.expected:
                        additions.append(f"- Expected: {result.expected}")

        return "\n".join(additions)


# Singleton instance for easy import
guardrail_pipeline = GuardrailPipeline()
