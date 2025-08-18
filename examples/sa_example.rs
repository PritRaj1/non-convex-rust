mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{KBFConstraints, KBF};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif,
};

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf_json = r#"
    {
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "SA": {
                "initial_temp": 100.0,
                "cooling_rate": 0.998,
                "step_size": 2.0,
                "num_neighbors": 30,
                "x_min": 0.0,
                "x_max": 10.0,
                "min_step_size_factor": 0.5,
                "step_size_decay_power": 0.3,
                "min_temp_factor": 0.05,
                "use_adaptive_cooling": false,
                "advanced": {
                    "restart_strategy": {
                        "Stagnation": {
                            "max_iterations": 50,
                            "threshold": 1e-6
                        }
                    },
                    "stagnation_detection": {
                        "stagnation_window": 30,
                        "improvement_threshold": 1e-6
                    },
                    "adaptive_parameters": true,
                    "adaptation_rate": 0.2,
                    "improvement_history_size": 30,
                    "success_history_size": 30,
                    "cooling_schedule": "Exponential"
                }
            }
        }
    }
    "#;

    let config = Config::new(conf_json).unwrap();

    let obj_f = KBF;
    let constraints = KBFConstraints;

    let mut opt = NonConvexOpt::new(
        config,
        SMatrix::<f64, 1, 2>::from_vec(vec![
            rand::random::<f64>() * 10.0,
            rand::random::<f64>() * 10.0,
        ]),
        obj_f.clone(),
        Some(constraints.clone()),
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/sa_kbf.gif")?;

    for frame in 0..80 {
        let mut chart = setup_chart(
            frame,
            "Simulated Annealing",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/sa_frame.png",
        )?;

        // Draw current individual
        let population = opt.get_population();
        let current_x = population.row(0);
        chart.draw_series(std::iter::once(Circle::new(
            (current_x[0], current_x[1]),
            6,
            RGBColor(255, 0, 0).filled(),
        )))?;

        // Draw best individual
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;

        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/sa_frame.png")?
            .decode()?
            .into_rgba8();

        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }

        let mut frame = Frame::default();
        frame.width = 800;
        frame.height = 800;
        frame.delay = 10;
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/sa_frame.png")?;

    Ok(())
}
