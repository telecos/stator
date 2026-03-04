// chromium/content/stator_dual_engine.h — Differential-testing harness that
// runs the same script through both V8 and Stator and compares results.
//
// Overview
// --------
// When the kStatorDualEngine feature flag is active, Chrome executes each
// script evaluation twice in a renderer context:
//
//   1. Primary (V8)   — the canonical execution whose side-effects are applied
//                       to the real DOM.
//   2. Shadow (Stator)— an isolated execution on a snapshot of the same context
//                       whose side-effects are discarded.
//
// After both executions finish, RunDualEngine() serialises a subset of the
// global/DOM state from each engine and returns a DualEngineResult describing
// whether the states match.
//
// Discrepancies are surfaced via the DevTools Protocol event
// `Stator.dualEngineDiscrepancy`, enabling automated regression detection in
// Chrome canary/dev channels without shipping Stator to production users.
//
// Limitations
// -----------
//   - Only the top-level global object's enumerable own string properties are
//     compared; full DOM diffing is out of scope for this harness.
//   - Non-deterministic scripts (Date.now(), Math.random()) will always differ;
//     callers should filter expected discrepancies before surfacing them.
//   - Execution cost is approximately 2× per script evaluation; do not enable
//     in stable builds.

#ifndef CHROMIUM_CONTENT_STATOR_DUAL_ENGINE_H_
#define CHROMIUM_CONTENT_STATOR_DUAL_ENGINE_H_

#include <string>
#include <vector>

namespace content {

// A key-value snapshot of a JS global object's enumerable own string
// properties, captured after a script evaluation completes.
//
// Snapshot scope
// --------------
// Only enumerable, own, non-function string properties of the global object
// are recorded.  Deeply nested objects are serialised to their JSON
// representations; however the built-in JSON parser used here does NOT support:
//   - escaped quote characters inside JSON string values (e.g. "\"quoted\"")
//   - nested JSON objects or arrays as property values
// If either construct appears in the global state the comparison may produce
// false positives or miss discrepancies.  This is an accepted limitation for
// the canary/dev-channel differential testing use-case.
struct GlobalStateSnapshot {
  struct Entry {
    std::string key;
    std::string value;  // JSON-serialised JS value
  };
  std::vector<Entry> properties;

  // False when the engine isolate or context could not be initialised.
  // A snapshot with init_ok == false is always considered non-matching.
  bool init_ok = true;
};

// The result returned by RunDualEngine().
struct DualEngineResult {
  // True when both engines produced identical GlobalStateSnapshots.
  bool states_match = false;

  // Snapshot produced by the primary (V8) engine.
  GlobalStateSnapshot v8_state;

  // Snapshot produced by the shadow (Stator) engine.
  GlobalStateSnapshot stator_state;

  // Human-readable description of the first discrepancy found, or empty if
  // states_match is true.
  std::string discrepancy_description;
};

// Runs `script_utf8` through both V8 (primary) and Stator (shadow) and
// returns a DualEngineResult.
//
// Parameters:
//   script_utf8  — null-terminated UTF-8 source of the script to evaluate.
//   context_json — JSON string representing the initial global state to seed
//                  both engines with (may be "{}" for an empty context).
//
// The function is synchronous; both evaluations complete before it returns.
// Caller must ensure the kStatorDualEngine feature flag is enabled before
// invoking this function.
DualEngineResult RunDualEngine(const char* script_utf8,
                               const char* context_json);

// Converts a DualEngineResult into a JSON string suitable for embedding in a
// DevTools Protocol event payload.
std::string DualEngineResultToJson(const DualEngineResult& result);

}  // namespace content

#endif  // CHROMIUM_CONTENT_STATOR_DUAL_ENGINE_H_
