use ndarray::Array2;
use std::f64::consts::E;

pub trait Activation {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64>;
    fn backward(&self, forward_output: &Array2<f64>) -> Array2<f64>;
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| 1.0 / (1.0 + E.powf(-x)))
    }

    fn backward(&self, forward_output: &Array2<f64>) -> Array2<f64> {
        forward_output * (1.0 - forward_output)
    }
}

pub struct Tanh;
impl Activation for Tanh {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| x.tanh())
    }

    fn backward(&self, forward_output: &Array2<f64>) -> Array2<f64> {
        1.0 - forward_output.mapv(|x| x.powi(2))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, forward_output: &Array2<f64>) -> Array2<f64> {
        forward_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid;
        let inputs = array![[0.0, 1.0], [-1.0, -2.0]];
        let outputs = sigmoid.forward(&inputs);
        let expected = array![[0.5, 0.73105858], [0.26894142, 0.11920292]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_sigmoid_backward() {
        let sigmoid = Sigmoid;
        let forward_output = array![[0.5, 0.73105858], [0.26894142, 0.11920292]];
        let outputs = sigmoid.backward(&forward_output);
        let expected = array![[0.25, 0.19661193], [0.19661193, 0.10499359]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh;
        let inputs = array![[0.0, 1.0], [-1.0, -2.0]];
        let outputs = tanh.forward(&inputs);
        let expected = array![[0.0, 0.76159416], [-0.76159416, -0.96402758]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_tanh_backward() {
        let tanh = Tanh;
        let forward_output = array![[0.0, 0.76159416], [-0.76159416, -0.96402758]];
        let outputs = tanh.backward(&forward_output);
        let expected = array![[1.0, 0.41997434], [0.41997434, 0.07065082]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_relu_forward() {
        let relu = ReLU;
        let inputs = array![[0.0, 1.0], [-1.0, -2.0]];
        let outputs = relu.forward(&inputs);
        let expected = array![[0.0, 1.0], [0.0, 0.0]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_relu_backward() {
        let relu = ReLU;
        let forward_output = array![[0.0, 1.0], [0.0, 0.0]];
        let outputs = relu.backward(&forward_output);
        let expected = array![[0.0, 1.0], [0.0, 0.0]];
        assert!(outputs.abs_diff_eq(&expected, 1e-8));
    }
}
