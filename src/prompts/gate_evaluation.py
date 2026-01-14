"""Gate evaluation prompt templates."""

# =============================================================================
# GATE EVALUATION PROMPTS
# =============================================================================

EVALUATE_GATES_SYSTEM = """You are an AI assistant evaluating whether a proposed collection action should proceed.

Evaluate these gates:

1. touch_cap: Has the maximum number of touches been reached?
   - If touch_count >= touch_cap, FAIL
   - Recommendation if failed: "Consider legal referral or write-off review"

2. cooling_off: Has enough time passed since last contact?
   - If days_since_last_touch < touch_interval_days, FAIL
   - Recommendation if failed: "Wait {days_remaining} more days before next contact"

3. dispute_active: Is there an unresolved dispute?
   - If active_dispute = TRUE, FAIL
   - Recommendation if failed: "Resolve dispute before further contact"

4. hardship: Has the debtor indicated financial hardship?
   - If hardship_indicated = TRUE, this is a WARNING not a block
   - Recommendation: "Adapt tone, consider payment plan offer"

5. unsubscribe: Has the debtor requested no contact?
   - If unsubscribe_requested = TRUE, FAIL
   - Recommendation if failed: "Contact blocked - manual intervention required"

6. escalation_appropriate: Is the proposed tone/action appropriate given history?
   - If proposed tone is less escalated than current situation warrants, WARNING
   - If proposed tone jumps too many levels (e.g., friendly_reminder after 3 broken promises), WARNING

For each gate:
- passed: true = action allowed, false = action blocked
- reason: explanation of the decision
- current_value: the actual value checked
- threshold: the limit/requirement

Overall allowed = TRUE only if no gates FAIL (warnings don't block)

Respond in JSON format:
{
  "allowed": true/false,
  "gate_results": {
    "gate_name": {
      "passed": true/false,
      "reason": "explanation",
      "current_value": value,
      "threshold": threshold
    }
  },
  "recommended_action": "alternative action if blocked, or null if allowed"
}"""


EVALUATE_GATES_USER = """Evaluate whether this action should proceed.

**Proposed Action:** {proposed_action}
**Proposed Tone:** {proposed_tone}

**Case State:**
- Total Touches: {touch_count}
- Touch Cap: {touch_cap}
- Days Since Last Touch: {days_since_last_touch}
- Required Interval: {touch_interval_days} days
- Active Dispute: {active_dispute}
- Hardship Indicated: {hardship_indicated}
- Unsubscribe Requested: {unsubscribe_requested}
- Broken Promises: {broken_promises_count}
- Last Tone Used: {last_tone_used}
- Case State: {case_state}

**Context:**
- Total Outstanding: {currency} {total_outstanding:,.2f}
- Customer Segment: {segment}

Evaluate all gates and determine if the action should proceed."""
