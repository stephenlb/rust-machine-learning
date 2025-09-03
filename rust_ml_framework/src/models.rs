use crate::layers::{Dense, Layer};
use ndarray::Array2;

pub struct Sequential {
    pub layers: Vec<Dense>,
}

impl Sequential {
    pub fn new(layers: Vec<Dense>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, output_gradient: &Array2<f64>) {
        let mut gradient = output_gradient.clone();
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient);
        }
    }
}
