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
            let updated_weights = layer.get_weights() - &(layer.get_dweights() * self.learning_rate);
            let updated_biases = layer.get_biases() - &(layer.get_dbiases() * self.learning_rate);
            layer.set_weights(updated_weights);
            layer.set_biases(updated_biases);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Sequential;
    use crate::layers::{Dense};
    use crate::activations::Sigmoid;
    use ndarray::array;

    #[test]
    fn test_sgd_update() {
        let mut model = Sequential::new(vec![Box::new(Dense::new_with_weights(
            2,
            2,
            Box::new(Sigmoid),
            array![[0.1, 0.2], [0.3, 0.4]],
            array![[0.5, 0.6]],
        ))]);
        let optimizer = SGD::new(0.1);

        model.layers[0].set_dweights(array![[0.1, 0.2], [0.3, 0.4]]);
        model.layers[0].set_dbiases(array![[0.5, 0.6]]);

        optimizer.update(&mut model);

        let expected_weights = array![[0.09, 0.18], [0.27, 0.36]];
        let expected_biases = array![[0.45, 0.54]];

        assert!(model.layers[0]
            .get_weights()
            .abs_diff_eq(&expected_weights, 1e-8));
        assert!(model.layers[0]
            .get_biases()
            .abs_diff_eq(&expected_biases, 1e-8));
    }
}
