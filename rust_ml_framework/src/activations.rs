use ndarray::Array2;
use std::f64::consts::E;

pub trait Activation {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64>;
    fn backward(&self, inputs: &Array2<f64>) -> Array2<f64>;
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| 1.0 / (1.0 + E.powf(-x)))
    }

    fn backward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let s = self.forward(inputs);
        &s * (1.0 - &s)
    }
}

pub struct Tanh;
impl Activation for Tanh {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| x.tanh())
    }

    fn backward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        1.0 - self.forward(inputs).mapv(|x| x.powi(2))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
