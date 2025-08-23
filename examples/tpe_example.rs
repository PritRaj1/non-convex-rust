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
            "max_iter": 30,
            "rtol": "1e-6",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0,
            "stagnation_window": 20
        },
        "alg_conf": {
            "TPE": {
                "n_initial_random": 20,
                "n_ei_candidates": 100,
                "gamma": 0.25,
                "prior_weight": 1.0,
                "kernel_type": "Gaussian",
                "max_history": 1000,
                "advanced": {
                    "use_restart_strategy": true,
                    "restart_frequency": 50,
                    "use_adaptive_gamma": true,
                    "use_meta_optimization": true,
                    "meta_optimization_frequency": 10,
                    "use_early_stopping": true,
                    "early_stopping_patience": 20,
                    "use_constraint_aware": true
                },
                "bandwidth": {
                    "method": "Adaptive",
                    "cv_folds": 5,
                    "adaptation_rate": 0.1,
                    "min_bandwidth": 1e-6,
                    "max_bandwidth": 10.0
                },
                "acquisition": {
                    "acquisition_type": "ExpectedImprovement",
                    "xi": 0.01,
                    "kappa": 2.0,
                    "use_entropy": false,
                    "entropy_weight": 0.1
                },
                "sampling": {
                    "strategy": "Hybrid",
                    "adaptive_noise": true,
                    "noise_scale": 0.1,
                    "use_thompson": true,
                    "local_search": true,
                    "local_search_steps": 10
                }
            }
        }
    }"#;

    let config = Config::new(config_json).unwrap();

    let obj_f = Kbf;
    let constraints = KbfConstraints;

    let mut init_pop = SMatrix::<f64, 100, 2>::zeros();
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 10.0;
        }
    }

    let mut opt = NonConvexOpt::new(
        config,
        init_pop,
        obj_f.clone(),
        Some(constraints.clone()),
        42,
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/tpe_kbf.gif")?;

    for frame in 0..30 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "TPE",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/tpe_frame.png",
        })?;

        let population = opt.get_population();
        chart.draw_series(
            population
                .row_iter()
                .map(|row| Circle::new((row[0], row[1]), 3, RGBColor(255, 0, 0).filled())),
        )?;

        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;

        let img = ImageReader::open("examples/tpe_frame.png")?
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
            delay: 6,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/tpe_frame.png")?;

    Ok(())
}
