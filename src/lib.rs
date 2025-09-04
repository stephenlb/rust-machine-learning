pub mod activations;
pub mod layers;
pub mod losses;
pub mod models;
pub mod optimizers;

use activations::{Sigmoid, Tanh};
use layers::Dense;
use losses::{Loss, MeanSquaredError};
use models::Sequential;
use ndarray::{array, Array2};
use optimizers::SGD;

fn get_xor_data() -> (Array2<f64>, Array2<f64>) {
    let x_train = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y_train = array![[0.], [1.], [1.], [0.]];
    (x_train, y_train)
}

fn build_model() -> Sequential {
    Sequential::new(vec![
        Box::new(Dense::new(2, 3, Box::new(Tanh))),
        Box::new(Dense::new(3, 1, Box::new(Sigmoid))),
    ])
}

fn train_model(model: &mut Sequential, x_train: &Array2<f64>, y_train: &Array2<f64>) {
    let optimizer = SGD::new(0.1);
    let loss_fn = MeanSquaredError;

    for epoch in 0..10001 {
        let y_pred = model.forward(x_train);
        let loss = loss_fn.forward(y_train, &y_pred);
        let grad = loss_fn.backward(y_train, &y_pred);
        model.backward(&grad);
        optimizer.update(model);

        if epoch % 1000 == 0 {
            println!("Epoch {}/10000, Loss: {}", epoch, loss);
        }
    }
}

fn predict(model: &mut Sequential, x_train: &Array2<f64>) {
    println!("\nPredictions:");
    let y_pred = model.forward(x_train);
    for (x, y) in x_train.outer_iter().zip(y_pred.outer_iter()) {
        println!("Input: {}, Prediction: {}", x, y[0]);
    }
}

pub fn run() {
    let (x_train, y_train) = get_xor_data();
    let mut model = build_model();
    train_model(&mut model, &x_train, &y_train);
    predict(&mut model, &x_train);
}
