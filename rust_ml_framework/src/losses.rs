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
