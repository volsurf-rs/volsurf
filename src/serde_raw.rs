//! Serde shadow-struct generation for validated model types.
//!
//! Every model with construction invariants (SVI, SABR, spline, SSVI, eSSVI)
//! serializes through a plain shadow struct and deserializes back through its
//! own validating `new()`, so a hand-edited JSON file cannot produce a model
//! that `new()` would have rejected.
//!
//! Writing that by hand meant restating the field list three times per type —
//! once in the shadow struct, once in `TryFrom`, once in `From` — where a field
//! added to the model but missed in `From` would silently drop out of the
//! serialized form with nothing to catch it. Here the list appears once.

/// Generate the serde shadow struct and both conversions for a validated model.
///
/// The generated `TryFrom<Raw>` calls `Model::new(..)` with the fields in
/// declaration order, so **the field order must match `new()`'s parameter
/// order**. Field names must match the model's own field names, which is what
/// makes the round-trip total.
///
/// Use the `@shadow` form when the model's fields are not directly reachable
/// (e.g. a newtype wrapping another model); it emits the struct and `TryFrom`
/// and leaves `From<Model>` to the caller.
macro_rules! validated_serde {
    ($ty:ty => $raw:ident { $($field:ident : $fty:ty),+ $(,)? }) => {
        validated_serde!(@shadow $ty => $raw { $($field: $fty),+ });

        impl From<$ty> for $raw {
            fn from(model: $ty) -> Self {
                Self { $($field: model.$field),+ }
            }
        }
    };

    (@shadow $ty:ty => $raw:ident { $($field:ident : $fty:ty),+ $(,)? }) => {
        #[derive(::serde::Serialize, ::serde::Deserialize)]
        struct $raw { $($field: $fty),+ }

        impl TryFrom<$raw> for $ty {
            type Error = $crate::error::VolSurfError;

            fn try_from(raw: $raw) -> ::core::result::Result<Self, Self::Error> {
                Self::new($(raw.$field),+)
            }
        }
    };
}

pub(crate) use validated_serde;
