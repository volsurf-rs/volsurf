use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyType;
use volsurf::conventions::StickyKind;
use volsurf::smile::{ArbitrageReport, ButterflyViolation};
use volsurf::surface::{CalendarViolation, SmileModel, SurfaceDiagnostics};
use volsurf::{OptionType, SmileSection, VolSurface};

use crate::error::to_py_err;

// ── Enums ──

#[pyclass(eq, eq_int, frozen, from_py_object, name = "OptionType")]
#[derive(Clone, PartialEq)]
pub enum PyOptionType {
    Call,
    Put,
}

impl From<PyOptionType> for OptionType {
    fn from(v: PyOptionType) -> Self {
        match v {
            PyOptionType::Call => OptionType::Call,
            PyOptionType::Put => OptionType::Put,
        }
    }
}

impl From<OptionType> for PyOptionType {
    fn from(v: OptionType) -> Self {
        match v {
            OptionType::Call => PyOptionType::Call,
            OptionType::Put => PyOptionType::Put,
        }
    }
}

#[pyclass(eq, eq_int, frozen, from_py_object, name = "StickyKind")]
#[derive(Clone, PartialEq)]
pub enum PyStickyKind {
    StickyStrike,
    StickyDelta,
}

impl From<PyStickyKind> for StickyKind {
    fn from(v: PyStickyKind) -> Self {
        match v {
            PyStickyKind::StickyStrike => StickyKind::StickyStrike,
            PyStickyKind::StickyDelta => StickyKind::StickyDelta,
        }
    }
}

impl From<StickyKind> for PyStickyKind {
    fn from(v: StickyKind) -> Self {
        match v {
            StickyKind::StickyStrike => PyStickyKind::StickyStrike,
            StickyKind::StickyDelta => PyStickyKind::StickyDelta,
        }
    }
}

#[pyclass(frozen, from_py_object, name = "SmileModel")]
#[derive(Clone)]
pub struct PySmileModel {
    pub(crate) inner: SmileModel,
}

#[pymethods]
impl PySmileModel {
    #[classmethod]
    fn svi(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: SmileModel::Svi,
        }
    }

    #[classmethod]
    fn cubic_spline(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: SmileModel::CubicSpline,
        }
    }

    #[classmethod]
    fn sabr(_cls: &Bound<'_, PyType>, beta: f64) -> Self {
        Self {
            inner: SmileModel::Sabr { beta },
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

// ── PySmile ──

#[pyclass(frozen, name = "Smile")]
pub struct PySmile {
    pub(crate) inner: Box<dyn SmileSection>,
}

#[pymethods]
impl PySmile {
    fn vol(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.vol(strike).map_err(to_py_err)?.0)
    }

    fn variance(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.variance(strike).map_err(to_py_err)?.0)
    }

    fn density(&self, strike: f64) -> PyResult<f64> {
        self.inner.density(strike).map_err(to_py_err)
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry()
    }

    fn is_arbitrage_free(&self) -> PyResult<PyArbitrageReport> {
        Ok(self.inner.is_arbitrage_free().map_err(to_py_err)?.into())
    }
}

// ── PySurface ──

#[pyclass(frozen, from_py_object, name = "Surface")]
#[derive(Clone)]
pub struct PySurface {
    pub(crate) inner: Arc<dyn VolSurface>,
}

#[pymethods]
impl PySurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self.inner.black_vol(expiry, strike).map_err(to_py_err)?.0)
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self
            .inner
            .black_variance(expiry, strike)
            .map_err(to_py_err)?
            .0)
    }

    fn smile_at(&self, expiry: f64) -> PyResult<PySmile> {
        let smile = self.inner.smile_at(expiry).map_err(to_py_err)?;
        Ok(PySmile { inner: smile })
    }

    fn diagnostics(&self) -> PyResult<PySurfaceDiagnostics> {
        Ok(self.inner.diagnostics().map_err(to_py_err)?.into())
    }
}

// ── Diagnostics ──

#[pyclass(frozen, from_py_object, name = "ButterflyViolation")]
#[derive(Clone)]
pub struct PyButterflyViolation {
    #[pyo3(get)]
    strike: f64,
    #[pyo3(get)]
    density: f64,
    #[pyo3(get)]
    magnitude: f64,
}

#[pymethods]
impl PyButterflyViolation {
    fn __repr__(&self) -> String {
        format!(
            "ButterflyViolation(strike={}, density={}, magnitude={})",
            self.strike, self.density, self.magnitude
        )
    }
}

impl From<&ButterflyViolation> for PyButterflyViolation {
    fn from(v: &ButterflyViolation) -> Self {
        Self {
            strike: v.strike,
            density: v.density,
            magnitude: v.magnitude,
        }
    }
}

#[pyclass(frozen, from_py_object, name = "ArbitrageReport")]
#[derive(Clone)]
pub struct PyArbitrageReport {
    #[pyo3(get)]
    is_free: bool,
    violations: Vec<ButterflyViolation>,
}

#[pymethods]
impl PyArbitrageReport {
    #[getter]
    fn butterfly_violations(&self) -> Vec<PyButterflyViolation> {
        self.violations
            .iter()
            .map(PyButterflyViolation::from)
            .collect()
    }

    fn worst_violation(&self) -> Option<PyButterflyViolation> {
        self.violations
            .iter()
            .max_by(|a, b| {
                a.magnitude
                    .partial_cmp(&b.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(PyButterflyViolation::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "ArbitrageReport(is_free={}, violations={})",
            self.is_free,
            self.violations.len()
        )
    }
}

impl From<ArbitrageReport> for PyArbitrageReport {
    fn from(r: ArbitrageReport) -> Self {
        Self {
            is_free: r.is_free,
            violations: r.butterfly_violations,
        }
    }
}

#[pyclass(frozen, from_py_object, name = "CalendarViolation")]
#[derive(Clone)]
pub struct PyCalendarViolation {
    #[pyo3(get)]
    strike: f64,
    #[pyo3(get)]
    tenor_short: f64,
    #[pyo3(get)]
    tenor_long: f64,
    #[pyo3(get)]
    variance_short: f64,
    #[pyo3(get)]
    variance_long: f64,
}

#[pymethods]
impl PyCalendarViolation {
    fn __repr__(&self) -> String {
        format!(
            "CalendarViolation(strike={}, short={}, long={})",
            self.strike, self.tenor_short, self.tenor_long
        )
    }
}

impl From<&CalendarViolation> for PyCalendarViolation {
    fn from(v: &CalendarViolation) -> Self {
        Self {
            strike: v.strike,
            tenor_short: v.tenor_short,
            tenor_long: v.tenor_long,
            variance_short: v.variance_short,
            variance_long: v.variance_long,
        }
    }
}

#[pyclass(frozen, from_py_object, name = "SurfaceDiagnostics")]
#[derive(Clone)]
pub struct PySurfaceDiagnostics {
    #[pyo3(get)]
    is_free: bool,
    smile_reports_inner: Vec<ArbitrageReport>,
    calendar_violations_inner: Vec<CalendarViolation>,
}

#[pymethods]
impl PySurfaceDiagnostics {
    #[getter]
    fn smile_reports(&self) -> Vec<PyArbitrageReport> {
        self.smile_reports_inner
            .iter()
            .cloned()
            .map(PyArbitrageReport::from)
            .collect()
    }

    #[getter]
    fn calendar_violations(&self) -> Vec<PyCalendarViolation> {
        self.calendar_violations_inner
            .iter()
            .map(PyCalendarViolation::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurfaceDiagnostics(is_free={}, smiles={}, calendar={})",
            self.is_free,
            self.smile_reports_inner.len(),
            self.calendar_violations_inner.len()
        )
    }
}

impl From<SurfaceDiagnostics> for PySurfaceDiagnostics {
    fn from(d: SurfaceDiagnostics) -> Self {
        Self {
            is_free: d.is_free,
            smile_reports_inner: d.smile_reports,
            calendar_violations_inner: d.calendar_violations,
        }
    }
}
