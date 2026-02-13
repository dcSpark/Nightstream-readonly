//! Serde helpers for proof types that contain non-trivially-serializable fields.
//!
//! In particular, `BatchedTimeProof` stores `Vec<&'static [u8]>` labels which
//! require custom serialization. We serialize them as `Vec<Vec<u8>>` and leak
//! the deserialized allocations so the `&'static` lifetime is valid.

use serde::de::{Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use std::fmt;

/// Serialize `Vec<&'static [u8]>` as `Vec<Vec<u8>>`.
pub fn serialize_static_byte_slices<S>(
    labels: &[&'static [u8]],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(labels.len()))?;
    for label in labels {
        seq.serialize_element(&label.to_vec())?;
    }
    seq.end()
}

/// Deserialize `Vec<Vec<u8>>` into `Vec<&'static [u8]>` by leaking allocations.
///
/// This is safe for proof deserialization where the proof lives for the duration
/// of the verification process. The leaked memory is small (label strings are
/// typically short domain-separation tags).
pub fn deserialize_static_byte_slices<'de, D>(
    deserializer: D,
) -> Result<Vec<&'static [u8]>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StaticByteSlicesVisitor;

    impl<'de> Visitor<'de> for StaticByteSlicesVisitor {
        type Value = Vec<&'static [u8]>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of byte arrays")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut result = Vec::with_capacity(seq.size_hint().unwrap_or(0));
            while let Some(bytes) = seq.next_element::<Vec<u8>>()? {
                // Leak the allocation to get a &'static [u8].
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                result.push(leaked);
            }
            Ok(result)
        }
    }

    deserializer.deserialize_seq(StaticByteSlicesVisitor)
}
