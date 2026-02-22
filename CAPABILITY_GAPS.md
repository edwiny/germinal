# Capability Gaps

## Explicit task completion logic - PENDING

Currently, the agent is done when it stops emitting tool calls. That's it. The orchestrator has no independent view of whether the work is actually complete — it just trusts that if the model produced a response with no <tool_call> block, the job is finished.


