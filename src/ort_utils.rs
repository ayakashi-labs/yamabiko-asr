use crate::{Device, Error, Result};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Outlet, TensorElementType, ValueType};
use std::path::Path;

pub(crate) fn build_session(
    model_path: &Path,
    device: Device,
    execution_providers: Vec<ExecutionProviderDispatch>,
    model_load_error: fn(String) -> Error,
) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|err| model_load_error(err.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| model_load_error(err.to_string()))?
        .with_intra_threads(4)
        .map_err(|err| model_load_error(err.to_string()))?
        .with_inter_threads(1)
        .map_err(|err| model_load_error(err.to_string()))?;

    builder = builder
        .with_execution_providers(execution_providers)
        .map_err(|err| Error::DeviceUnavailable {
            device,
            message: err.to_string(),
        })?;

    builder
        .commit_from_file(model_path)
        .map_err(|err| model_load_error(err.to_string()))
}

pub(crate) fn require_tensor<'a>(
    outlets: &'a [Outlet],
    location: &str,
    name: &str,
    expected_type: TensorElementType,
    expected_rank: Option<usize>,
    expected_dimensions: &[(usize, usize)],
) -> std::result::Result<&'a [i64], String> {
    let outlet = outlets
        .iter()
        .find(|outlet| outlet.name() == name)
        .ok_or_else(|| format!("{location} is missing '{name}'"))?;
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(format!(
            "{location} '{name}' must be a tensor, got {}",
            outlet.dtype()
        ));
    };
    if *ty != expected_type {
        return Err(format!(
            "{location} '{name}' must contain {expected_type}, got {ty}"
        ));
    }
    if let Some(expected_rank) = expected_rank
        && shape.len() != expected_rank
    {
        return Err(format!(
            "{location} '{name}' must have rank {expected_rank}, got shape {shape:?}"
        ));
    }
    for &(index, expected) in expected_dimensions {
        require_compatible_dimension(shape, index, expected, &format!("{location} '{name}'"))?;
    }
    Ok(shape.as_ref())
}

pub(crate) fn known_dimension(
    shape: &[i64],
    index: usize,
    label: &str,
) -> std::result::Result<Option<usize>, String> {
    match shape.get(index).copied() {
        Some(-1) => Ok(None),
        Some(value) if value > 0 => usize::try_from(value)
            .map(Some)
            .map_err(|_| format!("{label} dimension is too large: {value}")),
        Some(value) => Err(format!(
            "{label} dimension must be positive or dynamic, got {value}"
        )),
        None => Err(format!("{label} dimension is missing from shape {shape:?}")),
    }
}

pub(crate) fn require_compatible_dimension(
    shape: &[i64],
    index: usize,
    expected: usize,
    label: &str,
) -> std::result::Result<(), String> {
    if let Some(actual) = known_dimension(shape, index, label)?
        && actual != expected
    {
        return Err(format!(
            "{label} dimension {actual} does not match required size {expected}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::{Shape, SymbolicDimensions};

    fn tensor_outlet(name: &str, ty: TensorElementType, shape: &[i64]) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty,
                shape: Shape::new(shape.iter().copied()),
                dimension_symbols: SymbolicDimensions::empty(shape.len()),
            },
        )
    }

    #[test]
    fn dimensions_accept_dynamic_and_reject_invalid_values() {
        assert_eq!(known_dimension(&[-1, 80, -1], 0, "test").unwrap(), None);
        assert_eq!(known_dimension(&[-1, 80, -1], 1, "test").unwrap(), Some(80));
        assert!(known_dimension(&[-1, 0, -1], 1, "test").is_err());
        assert!(known_dimension(&[-1, -2, -1], 1, "test").is_err());
    }

    #[test]
    fn tensor_contract_checks_name_type_rank_and_static_dimensions() {
        let valid = [tensor_outlet(
            "audio_signal",
            TensorElementType::Float32,
            &[-1, 80, -1],
        )];
        assert_eq!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(3),
                &[(0, 1)],
            )
            .unwrap(),
            [-1, 80, -1]
        );

        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "missing",
                TensorElementType::Float32,
                Some(3),
                &[],
            )
            .is_err()
        );
        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Int32,
                Some(3),
                &[],
            )
            .is_err()
        );
        assert!(
            require_tensor(
                &valid,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(2),
                &[],
            )
            .is_err()
        );

        let incompatible = [tensor_outlet(
            "audio_signal",
            TensorElementType::Float32,
            &[2, 80, -1],
        )];
        assert!(
            require_tensor(
                &incompatible,
                "encoder input",
                "audio_signal",
                TensorElementType::Float32,
                Some(3),
                &[(0, 1)],
            )
            .is_err()
        );
    }
}
