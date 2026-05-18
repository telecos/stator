//! ECMAScript §6.2.6 Property Descriptor specification type.
//!
//! A *property descriptor* is a record that explains how a property behaves.
//! ECMAScript distinguishes two flavours:
//!
//! * **Data descriptors** — carry a `value` and a `writable` flag.
//! * **Accessor descriptors** — carry `get` and/or `set` callables.
//!
//! Both share the `enumerable` and `configurable` attributes.
//!
//! This module provides [`FullPropertyDescriptor`], which is the Rust
//! equivalent of the specification's *Property Descriptor* record, together
//! with helpers to convert to/from a plain [`JsObject`] (as required by
//! `Object.defineProperty` and `Object.getOwnPropertyDescriptor`).

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

/// ECMAScript §6.2.6 Property Descriptor.
///
/// Represents either a data descriptor, an accessor descriptor, or a generic
/// descriptor (one that has only `enumerable` and/or `configurable`).
#[derive(Debug, Clone)]
pub enum FullPropertyDescriptor {
    /// A data property descriptor: `{ value, writable, enumerable, configurable }`.
    Data {
        /// The property value (`[[Value]]`).
        value: JsValue,
        /// `[[Writable]]` — whether the value may be changed by assignment.
        writable: bool,
        /// `[[Enumerable]]` — whether the property is visited by `for…in`.
        enumerable: bool,
        /// `[[Configurable]]` — whether the descriptor may be changed and
        /// the property may be deleted.
        configurable: bool,
    },
    /// An accessor property descriptor: `{ get, set, enumerable, configurable }`.
    Accessor {
        /// `[[Get]]` — the getter function, or `Undefined` if absent.
        get: JsValue,
        /// `[[Set]]` — the setter function, or `Undefined` if absent.
        set: JsValue,
        /// `[[Enumerable]]`.
        enumerable: bool,
        /// `[[Configurable]]`.
        configurable: bool,
    },
    /// A generic descriptor that has only shared attributes and no
    /// value/writable/get/set fields.  Used when `Object.defineProperty`
    /// receives a descriptor that specifies only `enumerable` and/or
    /// `configurable`.
    Generic {
        /// `[[Enumerable]]`.
        enumerable: Option<bool>,
        /// `[[Configurable]]`.
        configurable: Option<bool>,
    },
}

impl FullPropertyDescriptor {
    /// Returns `true` if this is a data descriptor (has `value` or `writable`).
    pub fn is_data(&self) -> bool {
        matches!(self, Self::Data { .. })
    }

    /// Returns `true` if this is an accessor descriptor (has `get` or `set`).
    pub fn is_accessor(&self) -> bool {
        matches!(self, Self::Accessor { .. })
    }

    /// Returns `true` if this is a generic descriptor (neither data nor
    /// accessor).
    pub fn is_generic(&self) -> bool {
        matches!(self, Self::Generic { .. })
    }

    /// Returns the `[[Enumerable]]` attribute, if present.
    pub fn enumerable(&self) -> Option<bool> {
        match self {
            Self::Data { enumerable, .. } | Self::Accessor { enumerable, .. } => Some(*enumerable),
            Self::Generic { enumerable, .. } => *enumerable,
        }
    }

    /// Returns the `[[Configurable]]` attribute, if present.
    pub fn configurable(&self) -> Option<bool> {
        match self {
            Self::Data { configurable, .. } | Self::Accessor { configurable, .. } => {
                Some(*configurable)
            }
            Self::Generic { configurable, .. } => *configurable,
        }
    }

    /// Converts this descriptor into [`PropertyAttributes`] flags.
    ///
    /// For accessor descriptors `WRITABLE` is never set.  For generic
    /// descriptors absent flags default to `false`.
    pub fn to_attributes(&self) -> PropertyAttributes {
        let mut attrs = PropertyAttributes::empty();
        match self {
            Self::Data {
                writable,
                enumerable,
                configurable,
                ..
            } => {
                if *writable {
                    attrs |= PropertyAttributes::WRITABLE;
                }
                if *enumerable {
                    attrs |= PropertyAttributes::ENUMERABLE;
                }
                if *configurable {
                    attrs |= PropertyAttributes::CONFIGURABLE;
                }
            }
            Self::Accessor {
                enumerable,
                configurable,
                ..
            } => {
                if *enumerable {
                    attrs |= PropertyAttributes::ENUMERABLE;
                }
                if *configurable {
                    attrs |= PropertyAttributes::CONFIGURABLE;
                }
            }
            Self::Generic {
                enumerable,
                configurable,
            } => {
                if enumerable.unwrap_or(false) {
                    attrs |= PropertyAttributes::ENUMERABLE;
                }
                if configurable.unwrap_or(false) {
                    attrs |= PropertyAttributes::CONFIGURABLE;
                }
            }
        }
        attrs
    }

    /// Creates a default **data** descriptor for a newly created property.
    ///
    /// Equivalent to `{ value: undefined, writable: false, enumerable: false,
    /// configurable: false }`.
    pub fn default_data() -> Self {
        Self::Data {
            value: JsValue::Undefined,
            writable: false,
            enumerable: false,
            configurable: false,
        }
    }

    /// Builds a [`FullPropertyDescriptor`] by reading well-known keys from a
    /// plain JS object (or [`PlainObject`][JsValue::PlainObject]).
    ///
    /// This implements the ECMAScript §6.2.6.1 *ToPropertyDescriptor* abstract
    /// operation.
    pub fn from_object(obj: &JsValue) -> StatorResult<Self> {
        let lookup = |key: &str| -> Option<JsValue> {
            match obj {
                JsValue::PlainObject(map) => map.borrow().get(key).cloned(),
                _ => None,
            }
        };

        let has_get = lookup("get").is_some();
        let has_set = lookup("set").is_some();
        let has_value = lookup("value").is_some();
        let has_writable = lookup("writable").is_some();

        // §6.2.6.1 step 8: a descriptor cannot have both data and accessor fields.
        if (has_get || has_set) && (has_value || has_writable) {
            return Err(StatorError::TypeError(
                "Invalid property descriptor. Cannot both specify accessors and a value or writable attribute".to_string(),
            ));
        }

        let enumerable = lookup("enumerable").map(|v| v.to_boolean());
        let configurable = lookup("configurable").map(|v| v.to_boolean());

        if has_get || has_set {
            let get = lookup("get").unwrap_or(JsValue::Undefined);
            let set = lookup("set").unwrap_or(JsValue::Undefined);
            Ok(Self::Accessor {
                get,
                set,
                enumerable: enumerable.unwrap_or(false),
                configurable: configurable.unwrap_or(false),
            })
        } else if has_value || has_writable {
            let value = lookup("value").unwrap_or(JsValue::Undefined);
            let writable = lookup("writable").map(|v| v.to_boolean()).unwrap_or(false);
            Ok(Self::Data {
                value,
                writable,
                enumerable: enumerable.unwrap_or(false),
                configurable: configurable.unwrap_or(false),
            })
        } else {
            Ok(Self::Generic {
                enumerable,
                configurable,
            })
        }
    }

    /// Converts this descriptor into a plain JS object suitable for returning
    /// from `Object.getOwnPropertyDescriptor`.
    ///
    /// The returned [`JsValue::PlainObject`] has the appropriate keys
    /// (`value`, `writable`, `get`, `set`, `enumerable`, `configurable`).
    pub fn to_object(&self) -> JsValue {
        let mut map = PropertyMap::new();
        match self {
            Self::Data {
                value,
                writable,
                enumerable,
                configurable,
            } => {
                map.insert("value".to_string(), value.clone());
                map.insert("writable".to_string(), JsValue::Boolean(*writable));
                map.insert("enumerable".to_string(), JsValue::Boolean(*enumerable));
                map.insert("configurable".to_string(), JsValue::Boolean(*configurable));
            }
            Self::Accessor {
                get,
                set,
                enumerable,
                configurable,
            } => {
                map.insert("get".to_string(), get.clone());
                map.insert("set".to_string(), set.clone());
                map.insert("enumerable".to_string(), JsValue::Boolean(*enumerable));
                map.insert("configurable".to_string(), JsValue::Boolean(*configurable));
            }
            Self::Generic {
                enumerable,
                configurable,
            } => {
                if let Some(e) = enumerable {
                    map.insert("enumerable".to_string(), JsValue::Boolean(*e));
                }
                if let Some(c) = configurable {
                    map.insert("configurable".to_string(), JsValue::Boolean(*c));
                }
            }
        }
        JsValue::PlainObject(Rc::new(RefCell::new(map)))
    }

    /// ECMAScript §10.1.6.3 *ValidateAndApplyPropertyDescriptor*.
    ///
    /// Validates whether a property redefinition is allowed and returns the
    /// merged attribute set.  For [`Generic`](Self::Generic) descriptors the
    /// result is the *current* attributes with only the explicitly specified
    /// fields overridden — unspecified fields are preserved.  For [`Data`] and
    /// [`Accessor`](Self::Accessor) descriptors all four attribute slots are
    /// present so the result is simply [`to_attributes`](Self::to_attributes).
    ///
    /// Returns `Err(TypeError)` when the change violates the non-configurable
    /// invariants.
    pub fn validate_against(
        &self,
        key: &str,
        current_attrs: PropertyAttributes,
    ) -> StatorResult<PropertyAttributes> {
        let is_configurable = current_attrs.contains(PropertyAttributes::CONFIGURABLE);

        if !is_configurable {
            // Non-configurable → cannot make configurable.
            if self.configurable() == Some(true) {
                return Err(StatorError::TypeError(format!(
                    "Cannot redefine property '{key}': \
                     [[Configurable]] cannot change from false to true"
                )));
            }
            // Non-configurable → cannot change enumerable.
            if let Some(e) = self.enumerable()
                && e != current_attrs.contains(PropertyAttributes::ENUMERABLE)
            {
                return Err(StatorError::TypeError(format!(
                    "Cannot redefine property '{key}': \
                     [[Enumerable]] cannot change on a non-configurable property"
                )));
            }
            // Non-configurable data → cannot widen writable from false to true.
            if let Self::Data { writable, .. } = self
                && *writable
                && !current_attrs.contains(PropertyAttributes::WRITABLE)
            {
                return Err(StatorError::TypeError(format!(
                    "Cannot redefine property '{key}': \
                     [[Writable]] cannot change from false to true"
                )));
            }
        }

        Ok(self.merge_into(current_attrs))
    }

    /// Merges this descriptor into `current_attrs`, overriding only the
    /// attributes that are explicitly present in the descriptor.
    ///
    /// For [`Data`](Self::Data) and [`Accessor`](Self::Accessor) variants
    /// every attribute is present so the result is equivalent to
    /// [`to_attributes`](Self::to_attributes).  For
    /// [`Generic`](Self::Generic) only the specified fields are changed;
    /// unspecified fields are inherited from `current_attrs`.
    pub fn merge_into(&self, current_attrs: PropertyAttributes) -> PropertyAttributes {
        match self {
            Self::Data { .. } | Self::Accessor { .. } => self.to_attributes(),
            Self::Generic {
                enumerable,
                configurable,
            } => {
                let mut attrs = current_attrs;
                if let Some(e) = enumerable {
                    if *e {
                        attrs |= PropertyAttributes::ENUMERABLE;
                    } else {
                        attrs -= PropertyAttributes::ENUMERABLE;
                    }
                }
                if let Some(c) = configurable {
                    if *c {
                        attrs |= PropertyAttributes::CONFIGURABLE;
                    } else {
                        attrs -= PropertyAttributes::CONFIGURABLE;
                    }
                }
                attrs
            }
        }
    }
}

/// ECMAScript §6.2.6 Property Descriptor — *partial* form used when parsing a
/// descriptor object passed to `Object.defineProperty` /
/// `Object.defineProperties`.
///
/// Unlike [`FullPropertyDescriptor`] (which always carries fully resolved
/// attribute booleans for its `Data`/`Accessor` variants), this record mirrors
/// the specification's PropertyDescriptor exactly: every field is independently
/// *present* (`Some(_)`) or *absent* (`None`).  This distinction is essential
/// for the §10.1.6.3 *ValidateAndApplyPropertyDescriptor* algorithm, where
/// absent fields preserve the corresponding attribute of an existing property
/// (e.g. `Object.defineProperty(frozenObj, "x", { value: same })` must succeed
/// without resetting `[[Enumerable]]` to `false`).
#[derive(Debug, Clone, Default)]
pub struct InputPropertyDescriptor {
    /// `[[Value]]` if explicitly specified.
    pub value: Option<JsValue>,
    /// `[[Writable]]` if explicitly specified.
    pub writable: Option<bool>,
    /// `[[Get]]` if explicitly specified.
    pub get: Option<JsValue>,
    /// `[[Set]]` if explicitly specified.
    pub set: Option<JsValue>,
    /// `[[Enumerable]]` if explicitly specified.
    pub enumerable: Option<bool>,
    /// `[[Configurable]]` if explicitly specified.
    pub configurable: Option<bool>,
}

impl InputPropertyDescriptor {
    /// `true` iff the descriptor specifies any data-only field
    /// (`value` or `writable`).
    pub fn is_data(&self) -> bool {
        self.value.is_some() || self.writable.is_some()
    }

    /// `true` iff the descriptor specifies any accessor-only field
    /// (`get` or `set`).
    pub fn is_accessor(&self) -> bool {
        self.get.is_some() || self.set.is_some()
    }

    /// `true` iff the descriptor specifies neither data nor accessor fields
    /// (only `enumerable` and/or `configurable`, or completely empty).
    pub fn is_generic(&self) -> bool {
        !self.is_data() && !self.is_accessor()
    }

    /// `true` iff every field is absent.
    pub fn is_empty(&self) -> bool {
        self.value.is_none()
            && self.writable.is_none()
            && self.get.is_none()
            && self.set.is_none()
            && self.enumerable.is_none()
            && self.configurable.is_none()
    }

    /// Implements ECMAScript §6.2.6.1 *ToPropertyDescriptor* — converts a
    /// plain JS object into an [`InputPropertyDescriptor`].
    ///
    /// Returns `TypeError` if the descriptor mixes data and accessor fields.
    pub fn from_object(obj: &JsValue) -> StatorResult<Self> {
        let lookup = |key: &str| -> Option<JsValue> {
            match obj {
                JsValue::PlainObject(map) => map.borrow().get(key).cloned(),
                _ => None,
            }
        };

        let mut out = Self::default();
        if let Some(v) = lookup("enumerable") {
            out.enumerable = Some(v.to_boolean());
        }
        if let Some(v) = lookup("configurable") {
            out.configurable = Some(v.to_boolean());
        }
        if let Some(v) = lookup("value") {
            out.value = Some(v);
        }
        if let Some(v) = lookup("writable") {
            out.writable = Some(v.to_boolean());
        }
        if let Some(v) = lookup("get") {
            out.get = Some(v);
        }
        if let Some(v) = lookup("set") {
            out.set = Some(v);
        }

        if (out.get.is_some() || out.set.is_some())
            && (out.value.is_some() || out.writable.is_some())
        {
            return Err(StatorError::TypeError(
                "Invalid property descriptor. Cannot both specify accessors \
                 and a value or writable attribute"
                    .to_string(),
            ));
        }

        Ok(out)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Data descriptor ──────────────────────────────────────────────────────

    #[test]
    fn test_data_descriptor_classification() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(42),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        assert!(desc.is_data());
        assert!(!desc.is_accessor());
        assert!(!desc.is_generic());
    }

    #[test]
    fn test_data_descriptor_to_attributes() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(0),
            writable: true,
            enumerable: false,
            configurable: true,
        };
        let attrs = desc.to_attributes();
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(!attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_data_descriptor_to_object_roundtrip() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(99),
            writable: true,
            enumerable: false,
            configurable: true,
        };
        let obj = desc.to_object();
        let back = FullPropertyDescriptor::from_object(&obj).unwrap();
        assert!(back.is_data());
        if let FullPropertyDescriptor::Data {
            value,
            writable,
            enumerable,
            configurable,
        } = back
        {
            assert_eq!(value, JsValue::Smi(99));
            assert!(writable);
            assert!(!enumerable);
            assert!(configurable);
        }
    }

    // ── Accessor descriptor ──────────────────────────────────────────────────

    #[test]
    fn test_accessor_descriptor_classification() {
        let desc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Undefined,
            enumerable: false,
            configurable: false,
        };
        assert!(!desc.is_data());
        assert!(desc.is_accessor());
        assert!(!desc.is_generic());
    }

    #[test]
    fn test_accessor_descriptor_to_attributes_no_writable() {
        let desc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        let attrs = desc.to_attributes();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_accessor_descriptor_to_object_roundtrip() {
        let getter = JsValue::Boolean(true); // stand-in
        let setter = JsValue::Boolean(false); // stand-in
        let desc = FullPropertyDescriptor::Accessor {
            get: getter.clone(),
            set: setter.clone(),
            enumerable: true,
            configurable: false,
        };
        let obj = desc.to_object();
        let back = FullPropertyDescriptor::from_object(&obj).unwrap();
        assert!(back.is_accessor());
        if let FullPropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        } = back
        {
            assert_eq!(get, getter);
            assert_eq!(set, setter);
            assert!(enumerable);
            assert!(!configurable);
        }
    }

    // ── Generic descriptor ───────────────────────────────────────────────────

    #[test]
    fn test_generic_descriptor_classification() {
        let desc = FullPropertyDescriptor::Generic {
            enumerable: Some(true),
            configurable: None,
        };
        assert!(!desc.is_data());
        assert!(!desc.is_accessor());
        assert!(desc.is_generic());
    }

    // ── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn test_validate_rejects_configurable_on_nonconfigurable() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: true,
        };
        let current = PropertyAttributes::WRITABLE; // not configurable
        let result = desc.validate_against("p", current);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_enumerable_change_on_nonconfigurable() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: true, // was false
            configurable: false,
        };
        let current = PropertyAttributes::empty(); // not configurable, not enumerable
        let result = desc.validate_against("p", current);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_writable_false_to_true_on_nonconfigurable() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: false,
        };
        // non-configurable, non-writable
        let current = PropertyAttributes::empty();
        let result = desc.validate_against("p", current);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_allows_narrowing_writable_on_nonconfigurable() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        // non-configurable but writable
        let current = PropertyAttributes::WRITABLE;
        let result = desc.validate_against("p", current);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_allows_any_change_on_configurable() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        let current = PropertyAttributes::CONFIGURABLE;
        let result = desc.validate_against("p", current);
        assert!(result.is_ok());
    }

    // ── from_object validation ───────────────────────────────────────────────

    #[test]
    fn test_from_object_rejects_mixed_data_accessor() {
        let mut map = PropertyMap::new();
        map.insert("value".to_string(), JsValue::Smi(1));
        map.insert("get".to_string(), JsValue::Undefined);
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let result = FullPropertyDescriptor::from_object(&obj);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_object_empty_yields_generic() {
        let map = PropertyMap::new();
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let desc = FullPropertyDescriptor::from_object(&obj).unwrap();
        assert!(desc.is_generic());
    }

    // ── default_data ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_data_all_false() {
        let desc = FullPropertyDescriptor::default_data();
        if let FullPropertyDescriptor::Data {
            value,
            writable,
            enumerable,
            configurable,
        } = desc
        {
            assert_eq!(value, JsValue::Undefined);
            assert!(!writable);
            assert!(!enumerable);
            assert!(!configurable);
        } else {
            panic!("expected Data");
        }
    }

    // ── to_object produces correct shape ─────────────────────────────────

    #[test]
    fn test_data_descriptor_to_object_has_all_keys() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(42),
            writable: true,
            enumerable: false,
            configurable: true,
        };
        let obj = desc.to_object();
        if let JsValue::PlainObject(map) = obj {
            let borrow = map.borrow();
            assert!(borrow.contains_key("value"), "must have 'value'");
            assert!(borrow.contains_key("writable"), "must have 'writable'");
            assert!(borrow.contains_key("enumerable"), "must have 'enumerable'");
            assert!(
                borrow.contains_key("configurable"),
                "must have 'configurable'"
            );
            // Accessor fields must NOT be present on data descriptor.
            assert!(!borrow.contains_key("get"), "data desc must not have 'get'");
            assert!(!borrow.contains_key("set"), "data desc must not have 'set'");
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_accessor_descriptor_to_object_has_all_keys() {
        let desc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Undefined,
            enumerable: true,
            configurable: false,
        };
        let obj = desc.to_object();
        if let JsValue::PlainObject(map) = obj {
            let borrow = map.borrow();
            assert!(borrow.contains_key("get"), "must have 'get'");
            assert!(borrow.contains_key("set"), "must have 'set'");
            assert!(borrow.contains_key("enumerable"), "must have 'enumerable'");
            assert!(
                borrow.contains_key("configurable"),
                "must have 'configurable'"
            );
            // Data fields must NOT be present on accessor descriptor.
            assert!(
                !borrow.contains_key("value"),
                "accessor desc must not have 'value'"
            );
            assert!(
                !borrow.contains_key("writable"),
                "accessor desc must not have 'writable'"
            );
        } else {
            panic!("expected PlainObject");
        }
    }

    // ── validate_against: data→accessor conversion on non-configurable ──

    #[test]
    fn test_validate_configurable_allows_all() {
        // Configurable property allows any change.
        let desc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        let current = PropertyAttributes::CONFIGURABLE | PropertyAttributes::WRITABLE;
        assert!(desc.validate_against("p", current).is_ok());
    }

    #[test]
    fn test_validate_same_writable_false_on_non_configurable_ok() {
        // Setting writable:false when already false + non-configurable is a no-op.
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        // Already non-writable, non-configurable.
        let current = PropertyAttributes::empty();
        assert!(
            desc.validate_against("p", current).is_ok(),
            "narrowing writable (false→false) on non-configurable should succeed"
        );
    }

    // ── merge_into ──────────────────────────────────────────────────────────

    #[test]
    fn test_merge_into_data_ignores_current() {
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: true,
        };
        let current = PropertyAttributes::all();
        let merged = desc.merge_into(current);
        // Data descriptor fully replaces current attributes.
        assert!(merged.contains(PropertyAttributes::WRITABLE));
        assert!(!merged.contains(PropertyAttributes::ENUMERABLE));
        assert!(merged.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_merge_into_generic_preserves_unspecified() {
        let desc = FullPropertyDescriptor::Generic {
            enumerable: Some(true),
            configurable: None,
        };
        let current = PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE;
        let merged = desc.merge_into(current);
        // Writable preserved, enumerable overridden, configurable preserved.
        assert!(merged.contains(PropertyAttributes::WRITABLE));
        assert!(merged.contains(PropertyAttributes::ENUMERABLE));
        assert!(merged.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_merge_into_generic_clears_specified() {
        let desc = FullPropertyDescriptor::Generic {
            enumerable: None,
            configurable: Some(false),
        };
        let current = PropertyAttributes::WRITABLE
            | PropertyAttributes::ENUMERABLE
            | PropertyAttributes::CONFIGURABLE;
        let merged = desc.merge_into(current);
        // Configurable cleared, others preserved.
        assert!(merged.contains(PropertyAttributes::WRITABLE));
        assert!(merged.contains(PropertyAttributes::ENUMERABLE));
        assert!(!merged.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_merge_into_generic_empty_preserves_all() {
        let desc = FullPropertyDescriptor::Generic {
            enumerable: None,
            configurable: None,
        };
        let current = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        let merged = desc.merge_into(current);
        assert_eq!(merged, current);
    }
}
