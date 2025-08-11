use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, Scalar};
use num_traits::{Float, FromPrimitive, NumCast, One, Zero};
use simba::scalar::{
    ClosedAdd, ClosedAddAssign, ClosedDiv, ClosedDivAssign, ClosedMul, ClosedMulAssign, ClosedNeg,
    ClosedSub, ClosedSubAssign, SubsetOf,
};

// More general trait for float numbers
pub trait FloatNumber:
    Copy
    + Float
    + NumCast // Convert from other types to float
    + FromPrimitive // Convert from primitive types to float
    + SubsetOf<f64>
    + Scalar
    + ClosedAdd
    + ClosedMul
    + ClosedDiv
    + ClosedSub
    + ClosedNeg
    + ClosedAddAssign
    + ClosedMulAssign
    + ClosedDivAssign
    + ClosedSubAssign
    + Zero
    + One
    + std::fmt::Debug
    + std::marker::Send
    + std::marker::Sync
    + 'static
    + Send // Send/sync for safe concurrency
    + Sync
{
}

impl FloatNumber for f64 {}
impl FloatNumber for f32 {}

pub trait CloneBox<T: FloatNumber, D: Dim> {
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T, D>>;
}

pub trait CloneBoxConstraint<T: FloatNumber, D: Dim> {
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T, D>>;
}

impl<T: FloatNumber, D: Dim, F: ObjectiveFunction<T, D> + Clone + 'static> CloneBox<T, D> for F
where
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T, D>> {
        Box::new(self.clone())
    }
}

impl<T: FloatNumber, D: Dim, F: BooleanConstraintFunction<T, D> + Clone + 'static>
    CloneBoxConstraint<T, D> for F
where
    DefaultAllocator: Allocator<D>,
{
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T, D>> {
        Box::new(self.clone())
    }
}

pub trait ObjectiveFunction<T: FloatNumber, D: Dim>: CloneBox<T, D> + Send + Sync
where
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<T, D>) -> T;

    fn gradient(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }

    fn x_lower_bound(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }

    fn x_upper_bound(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }
}

pub trait BooleanConstraintFunction<T: FloatNumber, D: Dim>:
    CloneBoxConstraint<T, D> + Send + Sync
where
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<T, D>) -> bool;
}

pub struct OptProb<T: FloatNumber, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    pub objective: Box<dyn ObjectiveFunction<T, D>>,
    pub constraints: Option<Box<dyn BooleanConstraintFunction<T, D>>>,
}

impl<T: FloatNumber, D: Dim> OptProb<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(
        objective: Box<dyn ObjectiveFunction<T, D>>,
        constraints: Option<Box<dyn BooleanConstraintFunction<T, D>>>,
    ) -> Self {
        Self {
            objective,
            constraints,
        }
    }

    pub fn is_feasible(&self, x: &OVector<T, D>) -> bool {
        match &self.constraints {
            Some(constraints) => constraints.g(x),
            None => true,
        }
    }

    pub fn evaluate(&self, x: &OVector<T, D>) -> T {
        self.objective.f(x)
    }
}

impl<T, D> Clone for OptProb<T, D>
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone(&self) -> Self {
        Self {
            objective: self.objective.clone_box(),
            constraints: self.constraints.as_ref().map(|c| c.clone_box_constraint()),
        }
    }
}

#[derive(Clone)]
pub struct State<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    pub best_x: OVector<T, D>,
    pub best_f: T,
    pub pop: OMatrix<T, N, D>,
    pub fitness: OVector<T, N>,
    pub constraints: OVector<bool, N>,
    pub iter: usize,
}

pub trait OptimizationAlgorithm<T: FloatNumber, N: Dim, D: Dim>
where
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn step(&mut self);
    fn state(&self) -> &State<T, N, D>;
    fn get_simplex(&self) -> Option<&Vec<OVector<T, D>>> {
        None
    }
    fn get_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        None
    }
    fn get_replica_temperatures(&self) -> Option<Vec<T>> {
        None
    }
}
