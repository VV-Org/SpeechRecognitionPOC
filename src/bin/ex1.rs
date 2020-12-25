use nalgebra::VectorN;
use speech_recognition_poc::cost_with_hypothesis;

fn main() {
    
    let outputs_actual = VectorN::from(vec![1.0, 4.0, 4.0, 9.0, 16.0]);
    let inputs = VectorN::from(vec![1.0, 2.0, 2.0, 3.0, 4.0]);

    let test = cost_with_hypothesis(&inputs, &outputs_actual, &|x|x*x);
    println!("Test = {}", test);
}
