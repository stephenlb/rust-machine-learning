use crate::models::Sequential;

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn update(&self, model: &mut Sequential) {
        for layer in &mut model.layers {
            layer.weights = &layer.weights - &(&layer.dweights * self.learning_rate);
            layer.biases = &layer.biases - &(&layer.dbiases * self.learning_rate);
        }
    }
}
