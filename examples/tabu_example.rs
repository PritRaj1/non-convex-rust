mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{KbfConstraints, Kbf};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif, ChartParams,
};

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf_json = r#"{
        "opt_conf": {
            "max_iter": 80,
            "rtol": "1e-6",
            "atol": "0.0"
        },
        "alg_conf": {
            "TS": {
                "common": {
                    "tabu_list_size": 20,
                    "num_neighbors": 50,
                    "step_size": 1.5,
                    "perturbation_prob": 0.3,
                    "tabu_threshold": 0.05
                },
                "list_type": {
                    "Reactive": {
                        "min_tabu_size": 10,
                        "max_tabu_size": 30,
                        "increase_factor": 1.1,
                        "decrease_factor": 0.9
                    }
                },
                "advanced": {
                    "adaptive_parameters": true,
                    "aspiration_criteria": true,
                    "neighborhood_strategy": {
                        "Adaptive": {
                            "base_step": 1.5,
                            "adaptation_rate": 0.1
                        }
                    },
                    "restart_strategy": {
                        "Stagnation": {
                            "max_iterations": 40,
                            "threshold": 1e-6
                        }
                    },
                    "intensification_cycles": 4,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
                }
            }
        }
    }"#;

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
    let mut encoder = setup_gif("examples/gifs/tabu_kbf.gif")?;

    for frame in 0..80 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "Tabu Search",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/tabu_frame.png",
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
        let img = ImageReader::open("examples/tabu_frame.png")?
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
            delay: 5,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/tabu_frame.png")?;

    Ok(())
}
