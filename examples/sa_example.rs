mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{Kbf, KbfConstraints};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif, ChartParams,
};

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf_json = r#"
    {
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-8",
            "atol": "1e-8",
            "rtol_max_iter_fraction": 0.8
        },
        "alg_conf": {
            "SA": {
                "initial_temp": 50.0,
                "cooling_rate": 0.995,
                "step_size": 0.4,
                "num_neighbors": 50,
                "x_min": 0.0,
                "x_max": 10.0,
                "min_step_size_factor": 0.3,
                "step_size_decay_power": 0.2,
                "min_temp_factor": 0.01,
                "use_adaptive_cooling": true,
                "advanced": {
                    "restart_strategy": {
                        "Stagnation": {
                            "max_iterations": 30,
                            "threshold": 1e-8
                        }
                    },
                    "stagnation_detection": {
                        "stagnation_window": 50,
                        "improvement_threshold": 1e-8
                    },
                    "adaptive_parameters": true,
                    "adaptation_rate": 0.15,
                    "improvement_history_size": 50,
                    "success_history_size": 50,
                    "cooling_schedule": "Adaptive"
                }
            }
        }
    }
    "#;

    let config = Config::new(conf_json).unwrap();

    let obj_f = Kbf;
    let constraints = KbfConstraints;

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

    for frame_num in 0..100 {
        let mut chart = setup_chart(ChartParams {
            frame: frame_num,
            algorithm_name: "Simulated Annealing",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/sa_frame.png",
        })?;

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

        let frame = Frame::<'_> {
            width: 800,
            height: 800,
            delay: 10,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/sa_frame.png")?;

    Ok(())
}
