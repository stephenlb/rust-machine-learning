use crate::activations::Activation;
use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Layer {
    fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, output_gradient: &Array2<f64>) -> Array2<f64>;
    fn get_output(&self) -> &Array2<f64>;
    fn get_weights(&self) -> &Array2<f64>;
    fn get_biases(&self) -> &Array2<f64>;
    fn get_dweights(&self) -> &Array2<f64>;
    fn get_dbiases(&self) -> &Array2<f64>;
    fn set_weights(&mut self, weights: Array2<f64>);
    fn set_biases(&mut self, biases: Array2<f64>);
    fn set_dweights(&mut self, dweights: Array2<f64>);
    fn set_dbiases(&mut self, dbiases: Array2<f64>);
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

    pub fn new_with_weights(
        n_inputs: usize,
        n_neurons: usize,
        activation: Box<dyn Activation>,
        weights: Array2<f64>,
        biases: Array2<f64>,
    ) -> Self {
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

    fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    fn get_biases(&self) -> &Array2<f64> {
        &self.biases
    }

    fn get_dweights(&self) -> &Array2<f64> {
        &self.dweights
    }

    fn get_dbiases(&self) -> &Array2<f64> {
        &self.dbiases
    }

    fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights;
    }

    fn set_biases(&mut self, biases: Array2<f64>) {
        self.biases = biases;
    }

    fn set_dweights(&mut self, dweights: Array2<f64>) {
        self.dweights = dweights;
    }

    fn set_dbiases(&mut self, dbiases: Array2<f64>) {
        self.dbiases = dbiases;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Sigmoid;
    use ndarray::array;

    #[test]
    fn test_dense_forward() {
        let weights = array![[0.1, 0.2], [0.3, 0.4]];
        let biases = array![[0.5, 0.6]];
        let mut layer = Dense::new_with_weights(2, 2, Box::new(Sigmoid), weights, biases);
        let inputs = array![[1.0, 2.0]];
        let outputs = layer.forward(&inputs);
        let expected = array![[0.76852478, 0.83201839]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_dense_backward() {
        let weights = array![[0.1, 0.2], [0.3, 0.4]];
        let biases = array![[0.5, 0.6]];
        let mut layer =
            Dense::new_with_weights(2, 2, Box::new(Sigmoid), weights.clone(), biases.clone());
        let inputs = array![[1.0, 2.0]];
        layer.forward(&inputs);

        let output_gradient = array![[1.0, 1.0]];
        let input_gradient = layer.backward(&output_gradient);

        let expected_dweights = array![[0.17789444, 0.13976379], [0.35578888, 0.27952758]];
        let diff = &layer.dweights - &expected_dweights;
        assert!((&diff * &diff).sum() < 1e-8);

        let expected_dbiases = array![[0.17789444, 0.13976379]];
        let diff = &layer.dbiases - &expected_dbiases;
        assert!((&diff * &diff).sum() < 1e-8);

        let expected_input_gradient = array![[0.045742202, 0.109273848]];
        let diff = &input_gradient - &expected_input_gradient;
        assert!((&diff * &diff).sum() < 1e-8);
    }
}
