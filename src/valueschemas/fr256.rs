use std::convert::TryInto;

use crate::{RawId, Value, ValueSchema};
use hex_literal::hex;
use num_rational::Ratio;

use super::{PackValue, UnpackValue};

pub struct FR256LE;
pub struct FR256BE;

pub type FR256 = FR256LE;

impl ValueSchema for FR256LE {const ID: RawId = hex!("0A9B43C5C2ECD45B257CDEFC16544358");}
impl ValueSchema for FR256BE {const ID: RawId = hex!("CA5EAF567171772C1FFD776E9C7C02D1");}

impl UnpackValue<'_, FR256BE> for Ratio<i128> {
    fn unpack(v: &Value<FR256BE>) -> Self {
        let n = i128::from_be_bytes(v.bytes[0..16].try_into().unwrap());
        let d = i128::from_be_bytes(v.bytes[16..32].try_into().unwrap());

        Ratio::new(n, d)
    }
}

impl PackValue<FR256BE> for Ratio<i128> {
    fn pack(&self) -> Value<FR256BE> {
        let mut bytes = [0; 32];
        bytes[0..16].copy_from_slice(&self.numer().to_be_bytes());
        bytes[16..32].copy_from_slice(&self.denom().to_be_bytes());

        Value::new(bytes)
    }
}

impl UnpackValue<'_, FR256LE> for Ratio<i128> {
    fn unpack(v: &Value<FR256LE>) -> Self {
        let n = i128::from_le_bytes(v.bytes[0..16].try_into().unwrap());
        let d = i128::from_le_bytes(v.bytes[16..32].try_into().unwrap());

        Ratio::new(n, d)
    }
}

impl PackValue<FR256LE> for Ratio<i128> {
    fn pack(&self) -> Value<FR256LE> {
        let mut bytes = [0; 32];
        bytes[0..16].copy_from_slice(&self.numer().to_le_bytes());
        bytes[16..32].copy_from_slice(&self.denom().to_le_bytes());

        Value::new(bytes)
    }
}
