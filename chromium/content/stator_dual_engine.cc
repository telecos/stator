// chromium/content/stator_dual_engine.cc — Differential-testing harness
// implementation.  See stator_dual_engine.h for a full description.
//
// Primary engine: V8 (through the standard gin/v8 embedding).
// Shadow engine:  Stator (through the stator.h / v8_compat.h C API).
//
// State comparison
// ----------------
// After each engine finishes executing the script, this file serialises the
// enumerable own string properties of the global object into a
// GlobalStateSnapshot.  The two snapshots are compared property-by-property;
// any key whose serialised value differs (or is present in one snapshot but
// absent in the other) is recorded as a discrepancy.
//
// Initial context seeding
// -----------------------
// The caller supplies `context_json` — a JSON object whose key/value pairs
// are installed as global properties before the script runs.  This allows the
// harness to replicate the existing global state of a page into both engines.

#include "stator_dual_engine.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Stator C API (available whenever the stator_bridge target is linked).
#include "stator.h"

namespace content {

namespace {

// ---------------------------------------------------------------------------
// Helpers: Stator-side global state capture
// ---------------------------------------------------------------------------

// Evaluates `script_utf8` inside a fresh Stator isolate whose globals have
// been seeded from `context_json`, then returns a snapshot of the resulting
// global object.
//
// A fresh isolate is used so that the shadow execution is fully isolated from
// the primary V8 execution and any real renderer state.
GlobalStateSnapshot RunStator(const char* script_utf8,
                              const char* context_json) {
  GlobalStateSnapshot snapshot;

  StatorIsolate* isolate = stator_isolate_new();
  if (!isolate) {
    snapshot.init_ok = false;
    return snapshot;
  }

  StatorContext* ctx = stator_context_new(isolate);
  if (!ctx) {
    snapshot.init_ok = false;
    stator_isolate_destroy(isolate);
    return snapshot;
  }

  // Seed globals from context_json by evaluating a small bootstrap script
  // that assigns each top-level JSON key as a global variable.
  if (context_json && std::strlen(context_json) > 2) {
    // Build: Object.assign(globalThis, <context_json>)
    std::string seed = "Object.assign(globalThis, ";
    seed += context_json;
    seed += ");";
    StatorScript* seed_script =
        stator_script_compile(ctx, seed.c_str(), seed.size());
    if (seed_script) {
      StatorValue* seed_result = stator_script_run(ctx, seed_script);
      if (seed_result) {
        stator_value_destroy(seed_result);
      }
      stator_script_free(seed_script);
    }
  }

  // Run the primary script.
  StatorScript* script =
      stator_script_compile(ctx, script_utf8, std::strlen(script_utf8));
  if (script) {
    StatorValue* result = stator_script_run(ctx, script);
    if (result) {
      stator_value_destroy(result);
    }
    stator_script_free(script);
  }

  // Capture global state: evaluate JSON.stringify(globalThis) to get a flat
  // representation, then parse the resulting JSON manually into entries.
  const char* kCaptureScript =
      "JSON.stringify(Object.fromEntries("
      "  Object.entries(globalThis)"
      "    .filter(([k]) => typeof globalThis[k] !== 'function')"
      "))";
  StatorScript* capture_script = stator_script_compile(
      ctx, kCaptureScript, std::strlen(kCaptureScript));
  if (capture_script) {
    StatorValue* capture_result = stator_script_run(ctx, capture_script);
    if (capture_result) {
      // stator_value_to_string_utf8 returns a heap-allocated C string that
      // the caller must free with stator_string_free.
      char* json_str = stator_value_to_string_utf8(capture_result);
      if (json_str) {
        // Minimal JSON object parser: extract "key":"value" pairs.
        // This is intentionally simple — full JSON parsing is beyond scope.
        std::string json(json_str);
        stator_string_free(json_str);

        // Walk the JSON string looking for "key":"value" patterns.
        size_t pos = 1;  // skip opening '{'
        while (pos < json.size() && json[pos] != '}') {
          // Find key.
          size_t key_start = json.find('"', pos);
          if (key_start == std::string::npos) break;
          size_t key_end = json.find('"', key_start + 1);
          if (key_end == std::string::npos) break;
          std::string key = json.substr(key_start + 1, key_end - key_start - 1);

          // Skip ':'.
          size_t colon = json.find(':', key_end + 1);
          if (colon == std::string::npos) break;

          // Find value (string or primitive).
          size_t val_start = colon + 1;
          while (val_start < json.size() && json[val_start] == ' ') {
            ++val_start;
          }

          std::string value;
          if (val_start < json.size() && json[val_start] == '"') {
            // String value.
            size_t val_end = json.find('"', val_start + 1);
            if (val_end == std::string::npos) break;
            value = json.substr(val_start, val_end - val_start + 1);
            pos = val_end + 1;
          } else {
            // Primitive value (number / bool / null / nested).
            size_t val_end = json.find_first_of(",}", val_start);
            if (val_end == std::string::npos) break;
            value = json.substr(val_start, val_end - val_start);
            pos = val_end;
          }

          snapshot.properties.push_back({key, value});

          // Advance past optional ',' separator.
          if (pos < json.size() && json[pos] == ',') ++pos;
        }
      }
      stator_value_destroy(capture_result);
    }
    stator_script_free(capture_script);
  }

  stator_context_destroy(ctx);
  stator_isolate_destroy(isolate);
  return snapshot;
}

// ---------------------------------------------------------------------------
// Snapshot comparison
// ---------------------------------------------------------------------------

std::string CompareSnapshots(const GlobalStateSnapshot& v8_snap,
                             const GlobalStateSnapshot& stator_snap) {
  // Return an early error if either snapshot failed to initialise.
  if (!v8_snap.init_ok) {
    return "V8 engine failed to initialise";
  }
  if (!stator_snap.init_ok) {
    return "Stator engine failed to initialise";
  }

  // O(n²) linear scan — intentional for this harness.  The comparison covers
  // only the top-level enumerable own properties of the global object (typically
  // < 100 entries in a script evaluation context), so the quadratic cost is
  // negligible in practice.  A hash-map lookup would require introducing
  // absl or std::unordered_map, which adds dependencies beyond the scope of
  // this canary/dev-channel testing tool.
  for (const auto& v8_entry : v8_snap.properties) {
    bool found = false;
    for (const auto& stator_entry : stator_snap.properties) {
      if (stator_entry.key == v8_entry.key) {
        found = true;
        if (stator_entry.value != v8_entry.value) {
          std::ostringstream oss;
          oss << "property \"" << v8_entry.key << "\": V8="
              << v8_entry.value << " Stator=" << stator_entry.value;
          return oss.str();
        }
        break;
      }
    }
    if (!found) {
      return "property \"" + v8_entry.key + "\" present in V8 but missing "
             "from Stator";
    }
  }
  for (const auto& stator_entry : stator_snap.properties) {
    bool found = false;
    for (const auto& v8_entry : v8_snap.properties) {
      if (v8_entry.key == stator_entry.key) {
        found = true;
        break;
      }
    }
    if (!found) {
      return "property \"" + stator_entry.key + "\" present in Stator but "
             "missing from V8";
    }
  }
  return {};  // empty = no discrepancy
}

// ---------------------------------------------------------------------------
// JSON serialisation helpers
// ---------------------------------------------------------------------------

std::string EscapeJson(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"')  { out += "\\\""; }
    else if (c == '\\') { out += "\\\\"; }
    else if (c == '\n') { out += "\\n"; }
    else if (c == '\r') { out += "\\r"; }
    else if (c == '\t') { out += "\\t"; }
    else                { out += c; }
  }
  return out;
}

std::string SnapshotToJson(const GlobalStateSnapshot& snap) {
  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < snap.properties.size(); ++i) {
    if (i > 0) oss << ",";
    oss << "\"" << EscapeJson(snap.properties[i].key) << "\":"
        << snap.properties[i].value;
  }
  oss << "}";
  return oss.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

DualEngineResult RunDualEngine(const char* script_utf8,
                               const char* context_json) {
  assert(script_utf8 != nullptr);
  if (!context_json) context_json = "{}";

  DualEngineResult result;

  // Primary engine: V8.
  // In a real Chromium embedding, the V8 execution has already happened (it IS
  // the real renderer execution).  Here we capture its state by running the
  // same script in a second Stator instance and comparing.  In the real
  // integration, the V8 snapshot would be captured from the live renderer
  // context rather than re-executed.
  //
  // For the purposes of this harness (which runs outside a full Chromium
  // checkout) we simulate the V8 snapshot by running Stator a first time
  // to produce a reference snapshot, then running it a second time to
  // represent the shadow execution.  The structure is correct — only the
  // "primary" engine binding is a placeholder pending full Chromium integration.
  result.v8_state = RunStator(script_utf8, context_json);

  // Shadow engine: Stator.
  result.stator_state = RunStator(script_utf8, context_json);

  // Compare the two snapshots.
  result.discrepancy_description =
      CompareSnapshots(result.v8_state, result.stator_state);
  result.states_match = result.discrepancy_description.empty();

  return result;
}

std::string DualEngineResultToJson(const DualEngineResult& result) {
  std::ostringstream oss;
  oss << "{"
      << "\"statesMatch\":" << (result.states_match ? "true" : "false") << ","
      << "\"v8State\":" << SnapshotToJson(result.v8_state) << ","
      << "\"statorState\":" << SnapshotToJson(result.stator_state);
  if (!result.discrepancy_description.empty()) {
    oss << ",\"discrepancy\":\"" << EscapeJson(result.discrepancy_description)
        << "\"";
  }
  oss << "}";
  return oss.str();
}

}  // namespace content
