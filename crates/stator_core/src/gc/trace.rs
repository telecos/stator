/// Drives the mark-and-trace traversal during a GC cycle.
///
/// The tracer maintains a grey stack of pointers that have been *marked* as
/// reachable but whose outgoing references have not yet been visited.  The GC
/// loop pops entries from the grey stack and calls `Trace::trace` on them,
/// which in turn pushes their referents onto the grey stack.
pub struct Tracer {
    /// Raw pointers to heap objects that are marked but not yet fully traced.
    pub(crate) gray_stack: Vec<*mut u8>,
}

impl Tracer {
    /// Create a new, empty `Tracer`.
    pub fn new() -> Self {
        Self {
            gray_stack: Vec::new(),
        }
    }

    /// Mark a raw heap pointer as reachable and enqueue it for tracing.
    ///
    /// # Safety
    /// `ptr` must point to a live, properly-aligned heap object that will
    /// remain valid for the duration of the GC cycle.  Passing a null or
    /// dangling pointer is undefined behaviour.
    pub unsafe fn mark_raw(&mut self, ptr: *mut u8) {
        if !ptr.is_null() {
            self.gray_stack.push(ptr);
        }
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// All GC-managed types must implement `Trace` to expose their outgoing
/// heap references to the garbage collector.
///
/// # Contract
/// An implementation **must** call [`Tracer::mark_raw`] (or an equivalent
/// typed helper) for *every* heap pointer it owns.  Any pointer that is not
/// reported will be considered unreachable and may be freed or moved.
///
/// # Safety
/// Implementors must not hold any mutable borrows to GC-managed memory while
/// `trace` is running, as the tracer may inspect the same objects.
pub trait Trace {
    /// Visit all outgoing heap references, marking each via the tracer.
    fn trace(&self, tracer: &mut Tracer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct Leaf;
    impl Trace for Leaf {
        fn trace(&self, _tracer: &mut Tracer) {}
    }

    #[test]
    fn tracer_ignores_null() {
        let mut tracer = Tracer::new();
        // SAFETY: null pointer check is the point of this test.
        unsafe { tracer.mark_raw(std::ptr::null_mut()) };
        assert!(tracer.gray_stack.is_empty());
    }

    #[test]
    fn tracer_enqueues_non_null() {
        let mut x: u8 = 42;
        let mut tracer = Tracer::new();
        // SAFETY: &mut x is a valid, live pointer for this test.
        unsafe { tracer.mark_raw(&mut x as *mut u8) };
        assert_eq!(tracer.gray_stack.len(), 1);
    }

    // ── TaggedValue tracing ───────────────────────────────────────────────────

    #[test]
    fn test_trace_tagged_value_smi_not_marked() {
        use crate::objects::tagged::TaggedValue;
        let tv = TaggedValue::from_smi(42);
        let mut tracer = Tracer::new();
        tv.trace(&mut tracer);
        assert!(
            tracer.gray_stack.is_empty(),
            "Smi-tagged values must not be enqueued"
        );
    }

    #[test]
    fn test_trace_tagged_value_heap_ptr_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::tagged::TaggedValue;
        let mut obj = HeapObject::new_null();
        let obj_ptr = &raw mut obj;
        // SAFETY: obj_ptr is non-null and properly aligned.
        let tv = unsafe { TaggedValue::from_heap_object(obj_ptr) };
        let mut tracer = Tracer::new();
        tv.trace(&mut tracer);
        assert_eq!(tracer.gray_stack.len(), 1);
        assert_eq!(tracer.gray_stack[0], obj_ptr as *mut u8);
    }

    // ── JsValue tracing ───────────────────────────────────────────────────────

    #[test]
    fn test_trace_js_value_primitives_not_marked() {
        use crate::objects::value::JsValue;
        let mut tracer = Tracer::new();
        for v in [
            JsValue::Undefined,
            JsValue::Null,
            JsValue::Boolean(true),
            JsValue::Smi(0),
            JsValue::HeapNumber(1.0),
            JsValue::String("x".into()),
            JsValue::Symbol(1),
            JsValue::BigInt(99),
        ] {
            v.trace(&mut tracer);
        }
        assert!(
            tracer.gray_stack.is_empty(),
            "primitive JsValues must not enqueue any pointer"
        );
    }

    #[test]
    fn test_trace_js_value_object_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::value::JsValue;
        let mut obj = HeapObject::new_null();
        let obj_ptr = &raw mut obj;
        let v = JsValue::Object(obj_ptr);
        let mut tracer = Tracer::new();
        v.trace(&mut tracer);
        assert_eq!(tracer.gray_stack.len(), 1);
        assert_eq!(tracer.gray_stack[0], obj_ptr as *mut u8);
    }

    // ── JsObject reachability graph ───────────────────────────────────────────

    /// Mark a reachable graph: a `JsObject` with a fast property holding a
    /// `JsValue::Object` pointer.  The pointed-to `HeapObject` must appear in
    /// the gray stack; an unconnected `HeapObject` must not.
    #[test]
    fn test_trace_js_object_marks_reachable_and_skips_unreachable() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_object::JsObject;
        use crate::objects::map::PropertyAttributes;
        use crate::objects::value::JsValue;

        let mut reachable = HeapObject::new_null();
        let mut unreachable = HeapObject::new_null();
        let reachable_ptr = &raw mut reachable;
        let unreachable_ptr = &raw mut unreachable;

        let mut obj = JsObject::new();
        obj.define_own_property(
            "ref",
            JsValue::Object(reachable_ptr),
            PropertyAttributes::default(),
        )
        .unwrap();

        let mut tracer = Tracer::new();
        obj.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(reachable_ptr as *mut u8)),
            "reachable HeapObject must be in the gray stack"
        );
        assert!(
            !tracer.gray_stack.contains(&(unreachable_ptr as *mut u8)),
            "unreachable HeapObject must not be in the gray stack"
        );
    }

    /// A `JsObject` with indexed elements: element values are traced.
    #[test]
    fn test_trace_js_object_element_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_object::JsObject;
        use crate::objects::value::JsValue;

        let mut elem_obj = HeapObject::new_null();
        let elem_ptr = &raw mut elem_obj;

        let mut obj = JsObject::new();
        obj.set_element(0, JsValue::Object(elem_ptr));

        let mut tracer = Tracer::new();
        obj.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(elem_ptr as *mut u8)),
            "element HeapObject must be in the gray stack"
        );
    }

    /// Tracing a `JsObject` follows the prototype chain.
    #[test]
    fn test_trace_js_object_prototype_chain_traced() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_object::JsObject;
        use crate::objects::map::PropertyAttributes;
        use crate::objects::value::JsValue;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut proto_val = HeapObject::new_null();
        let proto_ptr = &raw mut proto_val;

        let mut proto = JsObject::new();
        proto
            .define_own_property(
                "x",
                JsValue::Object(proto_ptr),
                PropertyAttributes::default(),
            )
            .unwrap();

        let child = JsObject::with_prototype(Rc::new(RefCell::new(proto)));

        let mut tracer = Tracer::new();
        child.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(proto_ptr as *mut u8)),
            "pointer in the prototype must be reachable through the chain"
        );
    }

    // ── JsArray tracing ───────────────────────────────────────────────────────

    #[test]
    fn test_trace_js_array_element_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_array::JsArray;
        use crate::objects::value::JsValue;

        let mut elem_obj = HeapObject::new_null();
        let elem_ptr = &raw mut elem_obj;

        let mut arr = JsArray::new();
        arr.push(JsValue::Object(elem_ptr));

        let mut tracer = Tracer::new();
        arr.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(elem_ptr as *mut u8)),
            "JsArray element must be traced"
        );
    }

    // ── JsFunction tracing ────────────────────────────────────────────────────

    #[test]
    fn test_trace_js_function_context_binding_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_function::{Context, JsFunction, LanguageMode, SharedFunctionInfo};
        use crate::objects::value::JsValue;

        let mut captured = HeapObject::new_null();
        let captured_ptr = &raw mut captured;

        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        let mut ctx = Context::new();
        ctx.set("x", JsValue::Object(captured_ptr));
        let func = JsFunction::new_with_context(sfi, ctx);

        let mut tracer = Tracer::new();
        func.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(captured_ptr as *mut u8)),
            "HeapObject captured in closure context must be marked"
        );
    }

    #[test]
    fn test_trace_js_function_bound_args_marked() {
        use crate::objects::heap_object::HeapObject;
        use crate::objects::js_function::{JsFunction, LanguageMode, SharedFunctionInfo};
        use crate::objects::value::JsValue;
        use std::rc::Rc;

        let mut bound_obj = HeapObject::new_null();
        let bound_ptr = &raw mut bound_obj;

        let sfi = SharedFunctionInfo::new("g", 1, LanguageMode::Sloppy);
        let target = Rc::new(JsFunction::new(sfi));
        let bound = JsFunction::new_bound(
            Rc::clone(&target),
            JsValue::Object(bound_ptr),
            vec![JsValue::Smi(1)],
        );

        let mut tracer = Tracer::new();
        bound.trace(&mut tracer);

        assert!(
            tracer.gray_stack.contains(&(bound_ptr as *mut u8)),
            "bound_this HeapObject must be marked"
        );
    }

    // ── JsString (ConsString) tracing ─────────────────────────────────────────

    /// `ConsString` traces its left and right children recursively.
    /// Because `JsString` itself holds no GC-managed pointers, both halves
    /// must be traversed to reach any nested `ConsString` depth.
    #[test]
    fn test_trace_js_string_flat_no_marks() {
        use crate::objects::string::JsString;
        let s = JsString::new("hello");
        let mut tracer = Tracer::new();
        s.trace(&mut tracer);
        assert!(
            tracer.gray_stack.is_empty(),
            "flat JsString has no GC pointers"
        );
    }

    #[test]
    fn test_trace_js_string_cons_traverses_both_halves() {
        use crate::objects::string::JsString;
        // A Cons string's trace must visit both halves without panicking.
        // Neither half holds a GC pointer, but the traversal must complete.
        let left = JsString::new("hello");
        let right = JsString::new(" world");
        let cons = left.concat(right);
        let mut tracer = Tracer::new();
        cons.trace(&mut tracer);
        assert!(
            tracer.gray_stack.is_empty(),
            "ConsString of flat halves still has no GC pointers"
        );
    }
}
