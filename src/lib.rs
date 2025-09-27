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

fn _get_xor_data() -> (Array2<f64>, Array2<f64>) {
    let x_train = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y_train = array![[0.], [1.], [1.], [0.]];
    (x_train, y_train)
}

fn get_throdd_threeven() -> (Array2<f64>, Array2<f64>) {
    // Convert each number 0..100 to 8-bit binary array
    let mut x_data = Vec::new();
    for num in 0..100 {
        // Convert number to 8-bit binary representation
        for bit in (0..8).rev() {
            x_data.push(if (num >> bit) & 1 == 1 { 1.0 } else { 0.0 });
        }
    }
    
    let x_train = Array2::from_shape_vec((100, 8), x_data).unwrap();

    // 1 if throdd, 0 if threeven
    let y_train = Array2::from_shape_vec(
        (100, 1),
        (0..100).map(|x| if x % 3 == 0 { 0.0 } else { 1.0 }).collect(),
    ).unwrap();

    (x_train, y_train)
}

fn _get_odd_even() -> (Array2<f64>, Array2<f64>) {
    // Convert each number 0..100 to 8-bit binary array
    let mut x_data = Vec::new();
    for num in 0..100 {
        // Convert number to 8-bit binary representation
        for bit in (0..8).rev() {
            x_data.push(if (num >> bit) & 1 == 1 { 1.0 } else { 0.0 });
        }
    }
    
    let x_train = Array2::from_shape_vec((100, 8), x_data).unwrap();

    // 1 if odd, 0 if even
    let y_train = Array2::from_shape_vec(
        (100, 1),
        (0..100).map(|x| if x % 2 == 0 { 0.0 } else { 1.0 }).collect(),
    ).unwrap();

    (x_train, y_train)
}

fn build_model() -> Sequential {
    Sequential::new(vec![
        Box::new(Dense::new(8, 16, Box::new(Tanh))),
        Box::new(Dense::new(16, 8, Box::new(Tanh))),
        Box::new(Dense::new(8,  1, Box::new(Sigmoid))),
    ])
}

fn train_model(model: &mut Sequential, x_train: &Array2<f64>, y_train: &Array2<f64>) {
    let optimizer = SGD::new(0.1);
    let loss_fn = MeanSquaredError;

    for epoch in 0..20001 {
        let y_pred = model.forward(x_train);
        let loss = loss_fn.forward(y_train, &y_pred);
        let grad = loss_fn.backward(y_train, &y_pred);
        model.backward(&grad);
        optimizer.update(model);

        if epoch % 500 == 0 {
            println!("Epoch {}/5000, Loss: {}", epoch, loss);
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
    //let (x_train, y_train) = get_xor_data();
    let (x_train, y_train) = get_throdd_threeven();
    //println!("Training data:\nX:\n{:?}\nY:\n{:?}\n", x_train, y_train);

    let mut model = build_model();
    train_model(&mut model, &x_train, &y_train);
    predict(&mut model, &x_train);
    
    // Test on numbers 100-200
    println!("\n\nTesting on numbers 100-200:");
    let mut x_test_data = Vec::new();
    for num in 100..201 {
        // Convert number to 8-bit binary representation (will wrap for numbers > 255)
        for bit in (0..8).rev() {
            x_test_data.push(if (num >> bit) & 1 == 1 { 1.0 } else { 0.0 });
        }
    }
    
    let x_test = Array2::from_shape_vec((101, 8), x_test_data).unwrap();
    let y_test_pred = model.forward(&x_test);
    
    for (i, (_x, y)) in x_test.outer_iter().zip(y_test_pred.outer_iter()).enumerate() {
        let num = i + 100;
        let actual_odd_even = if num % 3 == 0 { "threeven" } else { "throdd" };
        let predicted_odd_even = if y[0] > 0.5 { "throdd" } else { "threeven" };
        println!("Number: {}, Actual: {}, Predicted: {} ({:.4})", 
                num, actual_odd_even, predicted_odd_even, y[0]);
    }
}
