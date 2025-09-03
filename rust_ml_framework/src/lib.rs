pub mod activations;
pub mod layers;
pub mod losses;
pub mod models;
pub mod optimizers;

use activations::{ReLU, Sigmoid, Tanh};
use layers::Dense;
use losses::{Loss, MeanSquaredError};
use models::Sequential;
use ndarray::array;
use optimizers::SGD;

pub fn run() {
    // 1. Dataset
    let x_train = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y_train = array![[0.], [1.], [1.], [0.]];

    // 2. Model
    let mut model = Sequential::new(vec![
        Dense::new(2, 3, Box::new(Tanh)),
        Dense::new(3, 1, Box::new(Sigmoid)),
    ]);

    // 3. Optimizer
    let optimizer = SGD::new(0.1);

    // 4. Loss
    let loss_fn = MeanSquaredError;

    // 5. Training loop
    for epoch in 0..10001 {
        let y_pred = model.forward(&x_train);
        let loss = loss_fn.forward(&y_train, &y_pred);
        let grad = loss_fn.backward(&y_train, &y_pred);
        model.backward(&grad);
        optimizer.update(&mut model);

        if epoch % 1000 == 0 {
            println!("Epoch {}/10000, Loss: {}", epoch, loss);
        }
    }

    // 6. Predictions
    println!("\nPredictions:");
    let y_pred = model.forward(&x_train);
    for (x, y) in x_train.outer_iter().zip(y_pred.outer_iter()) {
        println!("Input: {}, Prediction: {}", x, y[0]);
    }
}
