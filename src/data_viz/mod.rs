use nalgebra::{Dynamic, VectorN};
use plotters::prelude::*;
use std::error::Error;

pub fn plot_2d(
    path: &str,
    title: &str,
    legend: &str,
    x: &VectorN<f64, Dynamic>,
    y: &VectorN<f64, Dynamic>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-0.0..25.0, -5.0..25.0)?;

    chart.configure_mesh().draw()?;

    let points = x.iter().zip(y).map(|(&x, &y)| (x, y));

    chart
        .draw_series(PointSeries::of_element(points, 5, &RED, &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Cross::new((0,0),s,st.filled()); // At this point, the new pixel coordinate is established
        }))?
        .label(legend)
        .legend(|(x, y)| Cross::new((x, y), 5, &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
