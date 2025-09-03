use ndarray::Array2;

pub trait Loss {
    fn forward(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64;
    fn backward(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64>;
}

pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn forward(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let diff = y_true - y_pred;
        (&diff * &diff).mean().unwrap()
    }

    fn backward(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        (y_pred - y_true) * (2.0 / y_true.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_squared_error_forward() {
        let loss_fn = MeanSquaredError;
        let y_true = array![[0.0, 1.0], [1.0, 0.0]];
        let y_pred = array![[0.1, 0.9], [0.8, 0.2]];
        let loss = loss_fn.forward(&y_true, &y_pred);
        assert!((loss - 0.025).abs() < 1e-8);
    }

    #[test]
    fn test_mean_squared_error_backward() {
        let loss_fn = MeanSquaredError;
        let y_true = array![[0.0, 1.0], [1.0, 0.0]];
        let y_pred = array![[0.1, 0.9], [0.8, 0.2]];
        let grad = loss_fn.backward(&y_true, &y_pred);
        let expected = array![[0.05, -0.05], [-0.1, 0.1]];
        assert!(grad.abs_diff_eq(&expected, 1e-8));
    }
}
