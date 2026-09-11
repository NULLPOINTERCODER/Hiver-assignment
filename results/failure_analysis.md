# In-Depth Failure Analysis & Error Taxonomy

This document provides a systematic diagnostic audit of the error modes discovered during the evaluation of 200 hand-labelled Golden Set examples on the Apple Support customer support dataset.

---

## 1. Summary of Error Distribution
- **Total Evaluated Examples**: 200
- **Intent Misclassifications**: 80 / 200 (40.0%)
- **Escalation Routing Mismatches**: 111 / 200 (Conservative mode over-escalates low-confidence safe queries to prioritize customer safety)
- **False Auto-Handling Rate (Safety Critical)**: 8 / 35 (22.9%)
- **Conservative Over-Escalation Rate**: 103 / 165 (62.4%)

---

## 2. Top 5 Real Failure Modes

### Failure Mode 1: Multi-Entity Semantic Entanglement (Hardware vs Battery/Software)
- **Real Customer Example (`gold_007`)**:
  > *"The back of my iPhone is swelling and pushing the screen out from the frame!"*
- **Expected Intent**: `battery_power_issue`
- **Predicted Intent**: `hardware_repair_issue`
- **Expected Decision**: `escalate` | **Predicted Decision**: `escalate`
- **Why It Failed**: The customer query simultaneously mentions battery swelling and screen/frame enclosure deformation. The bag-of-words and dense vector representations gave disproportionate weight to `"screen"` and `"frame"`.
- **System Impact**: While intent taxonomy label diverged, the downstream Escalation Policy triggered correctly (`hardware_physical_damage` rule matched `"swollen"` and `"frame"`), protecting the user from physical safety hazards.
- **Proposed Improvement**: Implement hierarchical multi-label classification or entity-aware dependency parsing to recognize the root causal agent (battery swelling causing chassis separation).

---

### Failure Mode 2: Keyword Misleading & Lexical Ambiguity in UI Descriptions
- **Real Customer Example (`gold_024`)**:
  > *"Update requested... has been spinning on my screen for two days without downloading."*
- **Expected Intent**: `software_update_issue`
- **Predicted Intent**: `hardware_repair_issue`
- **Why It Failed**: The phrase `"spinning on my screen"` contains the high-salience token `"screen"`. The classifier failed to capture that `"Update requested..."` is an exact iOS OTA software update status string.
- **System Impact**: Retrieval searched for screen hardware fixes rather than OTA update cache clearing steps (`Settings > General > iPhone Storage`).
- **Proposed Improvement**: Add an explicit phrase-matching tokenizer for official Apple OS status strings (e.g., `"Update Requested"`, `"Preparing Update"`, `"Verifying Update"`).

---

### Failure Mode 3: Environmental & Contextual Query Drift
- **Real Customer Example (`gold_006`)**:
  > *"Phone dies at 30% when I'm outside in cold weather."*
- **Expected Intent**: `battery_power_issue`
- **Predicted Intent**: `general_inquiry_advice`
- **Why It Failed**: The embedding representation grouped `"outside in cold weather"` with seasonal and outdoor general inquiries rather than lithium-ion battery voltage depression characteristics in sub-zero temperatures.
- **System Impact**: The system routed the query to general advice, underestimating the diagnostic requirement for battery health verification.
- **Proposed Improvement**: Enrich the training dataset with domain-specific environmental and battery operating range telemetry examples.

---

### Failure Mode 4: False Auto-Handling on Multi-Turn Latent Hardware Defects
- **Real Customer Example (`gold_094`)**:
  > *"Vibration Taptic engine makes a loud buzzing grinding noise whenever I get a text."*
- **Expected Intent**: `hardware_repair_issue`
- **Expected Decision**: `escalate` | **Predicted Decision**: `auto`
- **Why It Failed**: The query did not contain overt destructive trigger keywords like `"shattered"` or `"water damage"`. The classifier identified the issue as audio/haptic settings and auto-suggested a sound reset rather than identifying mechanical actuator failure.
- **System Impact**: Customer received a software restart suggestion for a physical mechanical defect.
- **Proposed Improvement**: Expand physical hardware damage keyword rules to include mechanical acoustics (`"grinding"`, `"rattling"`, `"buzzing"`, `"loose parts"`).

---

### Failure Mode 5: Boundary Ambiguity Between Billing and Account Entitlement
- **Real Customer Example (`gold_068`)**:
  > *"I have an active subscription but the app is still telling me to upgrade to premium."*
- **Expected Intent**: `billing_subscription_issue`
- **Predicted Intent**: `app_store_app_issue`
- **Why It Failed**: This issue sits directly on the boundary between an App Store in-app purchase validation bug and a subscription billing entitlement sync.
- **System Impact**: Retrieved app crash guidance rather than in-app `"Restore Purchases"` and Apple ID subscription management guidance.
- **Proposed Improvement**: Define explicit composite boundary routing rules in `intents.yaml` mapping subscription sync issues to a unified `"Billing & App Subscriptions"` hybrid handler.
