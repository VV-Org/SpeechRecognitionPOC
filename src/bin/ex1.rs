use nalgebra::VectorN;
use speech_recognition_poc::{cost_with_hypothesis, data_viz};
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
    population: u32,
}

// The goal is to be able to call ex1 with as 1st argument the file that contains our dataset like
// ex1data1.txt and should then output the predicted profit for a population given as argument
fn main() -> Result<(), Box<dyn Error>> {
    let args = Ex1::from_args();
    println!("Dataset used: {}", args.dataset);

    println!("Target population: {}", args.population);

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

    let outputs_actual = VectorN::from(outputs_actual);
    let inputs = VectorN::from(inputs);

    let path = "./resources/plot-dataset.png";
    data_viz::plot_2d(
        path,
        "Visualize the dataset",
        "profits by city size",
        &inputs,
        &outputs_actual,
    )?;
    println!("Dataset plotted in: {}", path);

    let identity_cost = cost_with_hypothesis(&inputs, &outputs_actual, &|x| x);
    println!("Cost for identity hypothesis {}", identity_cost);
    Ok(())
}
