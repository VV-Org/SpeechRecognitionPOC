use crate::{Learner, Learners};
use na::{Dynamic, VectorN};

pub struct GradientDescentParameters {
    theta_init: VectorN<f64, Dynamic>,
    alpha: f64,
    num_iters: u32,
}

impl GradientDescentParameters {
    pub fn new(theta_init: VectorN<f64, Dynamic>, alpha: f64, num_iters: u32) -> Self {
        GradientDescentParameters {
            theta_init,
            alpha,
            num_iters,
        }
    }
}

impl Learner<GradientDescentParameters> for Learners {
    fn learn(
        inputs: &VectorN<f64, Dynamic>,
        outputs: &VectorN<f64, Dynamic>,
        parameters: GradientDescentParameters,
    ) -> VectorN<f64, Dynamic> {
        let x = inputs.clone().insert_columns(0, 1, 1.0);
        let m = outputs.len() as f64;
        let mut theta = parameters.theta_init;
        for _ in 1..parameters.num_iters {
            theta = &theta - (parameters.alpha / m) * x.transpose() * ((&x * &theta) - outputs);
        }
        theta
    }
}
