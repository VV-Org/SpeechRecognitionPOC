extern crate nalgebra as na;

pub use gradient_descent::*;
use na::{Dynamic, VectorN};

pub mod data_viz;
mod gradient_descent;

pub trait Learner<A> {
    fn learn(
        inputs: &VectorN<f64, Dynamic>,
        outputs: &VectorN<f64, Dynamic>,
        parameters: A,
    ) -> VectorN<f64, Dynamic>;
}

pub struct Learners;

pub fn cost(
    outputs_actual: &VectorN<f64, Dynamic>,
    outputs_predicted: &VectorN<f64, Dynamic>,
) -> f64 {
    let diff = outputs_actual - outputs_predicted;
    let diff_squared = &diff.component_mul(&diff);
    let diff_squared_mean = VectorN::mean(&diff_squared);
    diff_squared_mean / 2.0
}

#[cfg(test)]
mod tests {
    use crate::cost;
    use nalgebra::VectorN;

    #[test]
    fn test_cost_with_outputs() {
        let outputs_actual = VectorN::from(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let outputs_predicted = VectorN::from(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(0.0, cost(&outputs_actual, &outputs_predicted));

        let outputs_actual = VectorN::from(vec![2.0, 2.0, 2.0, 2.0]);
        let outputs_predicted = VectorN::from(vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(0.0, cost(&outputs_actual, &outputs_predicted));

        let outputs_actual = VectorN::from(vec![2.0, 2.0, 2.0, 2.0, 0.0]);
        let outputs_predicted = VectorN::from(vec![2.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(0.4, cost(&outputs_actual, &outputs_predicted));

        let outputs_actual = VectorN::from(vec![2.0, 2.0, 4.0, 2.0, 2.0]);
        let outputs_predicted = VectorN::from(vec![2.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(0.4, cost(&outputs_actual, &outputs_predicted));
    }
}
