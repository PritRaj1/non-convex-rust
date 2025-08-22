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
    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 20,
            "rtol": "1e-8",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "CEM": {
                "common": {
                    "population_size": 100,
                    "elite_size": 20,
                    "initial_std": 2.0,
                    "min_std": 1e-6,
                    "max_std": 5.0
                },
                "sampling": {
                    "use_antithetic": true,
                    "antithetic_ratio": 0.3
                },
                "adaptation": {
                    "smoothing_factor": 0.7
                },
                "advanced": {
                    "use_restart_strategy": true,
                    "restart_frequency": 50,
                    "use_covariance_adaptation": true,
                    "covariance_regularization": 1e-6
                }
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let obj_f = Kbf;
    let constraints = KbfConstraints;

    let mut init_pop = SMatrix::<f64, 100, 2>::zeros();
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 10.0;
        }
    }

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints.clone()));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/cem_kbf.gif")?;

    for frame in 0..20 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "CEM",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/cem_frame.png",
        })?;

        // Draw population
        let population = opt.get_population();
        chart.draw_series(
            population
                .row_iter()
                .map(|row| Circle::new((row[0], row[1]), 3, RGBColor(255, 0, 0).filled())),
        )?;

        // Draw best individual
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;

        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/cem_frame.png")?
            .decode()?
            .into_rgb8();

        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }

        let frame = Frame::<'_> {
            width: 800,
            height: 800,
            delay: 8,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/cem_frame.png")?;

    Ok(())
}
