use nalgebra::{Dynamic, VectorN};
use speech_recognition_poc::{cost, data_viz, GradientDescentParameters, Learner, Learners};
use std::fs::File;
use std::{error::Error, io::Read};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(about = "Ex1 implementation")]
pub struct Ex1 {
    #[structopt(
        default_value = "./resources/ex1data1.txt",
        short = "d",
        long = "dataset"
    )]
    dataset: String,
    #[structopt(default_value = "35000", short = "p", long = "population")]
    population: f64,
}

// The goal is to be able to call ex1 with as 1st argument the file that contains our dataset like
// ex1data1.txt and should then output the predicted profit for a population given as argument
fn main() -> Result<(), Box<dyn Error>> {
    let args = Ex1::from_args();

    println!("Dataset used: {}", args.dataset);
    println!("Target population: {}", args.population);

    let population = args.population / 10000.0;

    // Load and prepare data
    let mut file = File::open(args.dataset)?;
    let mut file_content = String::new();
    file.read_to_string(&mut file_content)?;
    let (inputs, outputs_actual): (Vec<f64>, Vec<f64>) = file_content
        .lines()
        .map(|line| {
            let split: Vec<&str> = line.split(",").collect();
            (
                split[0].parse::<f64>().expect("can't parse"),
                split[1].parse::<f64>().expect("can't parse"),
            )
        })
        .unzip();
    let outputs_actual = outputs_actual.into();
    let inputs: VectorN<f64, Dynamic> = inputs.into();

    // Plot the data set
    let path = "./resources/plot-dataset.png";
    data_viz::plot_2d(
        path,
        "Visualize the dataset",
        "profits by city size",
        &inputs,
        &outputs_actual,
    )?;
    println!("Dataset plotted in: {}", path);

    // Perform the gradient descent
    let params_gd = GradientDescentParameters::new(vec![0.0, 0.0].into(), 0.01, 1500);
    let theta = Learners::learn(&inputs, &outputs_actual, params_gd);
    println!("θ found after gradient descent: {}", theta);

    // Compute prediction
    let x: VectorN<f64, Dynamic> = vec![1.0, population].into();
    let predicted = x.transpose() * &theta;
    println!(
        "For a city size of {} we predict {:0.2}€ profits",
        args.population,
        predicted[(0, 0)] * 10000.0
    );

    // Print the cost with the found theta
    let x = inputs.clone().insert_columns(0, 1, 1.0);
    let outputs_predicted = x * theta;
    let identity_cost = cost(&outputs_actual, &outputs_predicted);

    println!("Cost for identity hypothesis {:0.2}", identity_cost);
    Ok(())
}
