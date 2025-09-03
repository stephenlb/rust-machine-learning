use crate::activations::Activation;
use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Layer {
    fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, output_gradient: &Array2<f64>) -> Array2<f64>;
    fn get_output(&self) -> &Array2<f64>;
}

pub struct Dense {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    inputs: Array2<f64>,
    output: Array2<f64>,
    activation: Box<dyn Activation>,
    pub dweights: Array2<f64>,
    pub dbiases: Array2<f64>,
}

impl Dense {
    pub fn new(n_inputs: usize, n_neurons: usize, activation: Box<dyn Activation>) -> Self {
        let weights = Array2::random((n_inputs, n_neurons), Uniform::new(-1.0, 1.0));
        let biases = Array2::zeros((1, n_neurons));
        Self {
            weights,
            biases,
            inputs: Array2::zeros((0, 0)),
            output: Array2::zeros((0, 0)),
            activation,
            dweights: Array2::zeros((n_inputs, n_neurons)),
            dbiases: Array2::zeros((1, n_neurons)),
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        self.inputs = inputs.clone();
        let output = inputs.dot(&self.weights) + &self.biases;
        self.output = self.activation.forward(&output);
        self.output.clone()
    }

    fn backward(&mut self, output_gradient: &Array2<f64>) -> Array2<f64> {
        let activation_gradient = self.activation.backward(&self.output) * output_gradient;
        self.dweights = self.inputs.t().dot(&activation_gradient);
        self.dbiases = activation_gradient.sum_axis(Axis(0)).insert_axis(Axis(0));
        activation_gradient.dot(&self.weights.t())
    }

    fn get_output(&self) -> &Array2<f64> {
        &self.output
    }
}
