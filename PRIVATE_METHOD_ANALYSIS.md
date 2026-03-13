# Private Class Method Implementation Gap Analysis

## Problem Summary
~2096 Test262 test failures related to private class methods all fail with:
`
CallProperty: callee is not a function (got Undefined)
`

## Root Cause: Private Methods Are Stored as Regular String Properties

### 1. How Private Methods Are Currently Compiled

**File**: crates/stator_core/src/bytecode/bytecode_generator.rs

Compilation of Private Methods (Lines 1201, 1207-1217):
- PropKey::Private(id) => Some(format!("#{}", id.name))  // Line 1201
- In compile_class_method(), stores as DefineNamedOwnProperty with key "#method"
- **Key Issue**: Private methods stored on prototype as string key, like regular properties

### 2. How Private Method Access Is Compiled

**File**: crates/stator_core/src/bytecode/bytecode_generator.rs

Reading Private Methods (Multiple locations):
- Line 459 (destructuring): LdaNamedProperty with key "#foo"
- Line 2647 (member access): LdaNamedProperty with key "#foo"  
- Line 2695 (optional member): LdaNamedProperty with key "#foo"
- Line 2950 (method call): LdaNamedProperty with key "#foo"
- Line 2754 (member assignment): StaNamedProperty with key "#foo"

### 3. Runtime: Private Methods Retrieved via LdaNamedProperty

**File**: crates/stator_core/src/interpreter/dispatch.rs

Handler for LdaNamedProperty (Lines 2370-2470):
- Gets property name from constant pool: "#method"
- Calls proto_lookup() to search by string key
- Returns value from object or prototype chain, or Undefined

### 4. The Real Problem

When a class is defined with a private method and getter:

class C {
  #method() { return 42; }
  get method() { return this.#method; }
}

The bytecode:
1. CreateClass - creates constructor and prototype
2. compile_class_method(proto, #method) - emits DefineNamedOwnProperty
3. compile_class_method(proto, get method) - defines getter

When getter runs and does this.#method:
- Compiles to LdaNamedProperty(this, "#method")
- Calls proto_lookup(this_instance, "#method")
- Looks for "#method" in instance and prototype chain
- Returns Undefined instead of the function

### 5. Why It Returns Undefined

Critical Issue in dispatch.rs, handle_define_named_own_property (lines 3651-3677):

fn handle_define_named_own_property(...) {
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut().insert(prop_name, val);
    }
    // What if obj is not a PlainObject? It silently does nothing!
    Ok(DispatchAction::Continue)
}

If the prototype is not a PlainObject (e.g., if it's a special class object), 
the property is never actually stored. The method is lost.

### 6. Proof: Minimal Failing Case

Test Case (from cls-decl-private-meth-args-trailing-comma-multiple.js):

class C {
  #method() { return 42; }
  get method() { return this.#method; }
}
new C().method();  // Error: callee is not a function

Bytecode Flow:
1. Compile #method function
2. DefineNamedOwnProperty(prototype, "#method", fn)
   - PROBLEM: If prototype is not a PlainObject, property is lost
3. Create instance via new C()
4. Call getter: get method() { return this.#method; }
5. LdaNamedProperty(instance, "#method")
   - proto_lookup() searches but doesn't find it
   - Returns Undefined
6. CallProperty tries to call Undefined
   - ERROR: "callee is not a function"

### 7. Files Involved

| File | Role |
|------|------|
| bytecode_generator.rs | Compiles #method() as DefineNamedOwnProperty(proto, "#method", fn) |
| dispatch.rs line 3651 | Handler silently skips if target is not PlainObject |
| dispatch.rs line 2370 | LdaNamedProperty retrieves with proto_lookup() |
| mod.rs line 2194 | proto_lookup() searches by string key in prototype chain |
| bytecodes.rs | Opcodes: DefineNamedOwnProperty, LdaNamedProperty |

### 8. Missing: Private Brand System

Bytecode has stubs for private brand checks:
- Opcode::TestPrivateBrand (line 5464-5471) - always passes, no-op
- Opcode::DefinePrivateBrand (line 5477-5482) - always passes, no-op

These are used for #x in obj syntax but don't actually track private fields/methods.

## Solution Directions

1. **Verify prototype type**: Ensure class prototype is PlainObject when storing
2. **Use special storage**: Implement WeakMap for private field indexing
3. **Fix proto_lookup**: Add special handling for "#" properties
4. **Implement brand system**: Add actual private member validation

Options 1 and 3 are quickest.