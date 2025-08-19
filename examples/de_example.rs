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
    let conf_json = r#"
    {
        "opt_conf": {
            "max_iter": 50,
            "rtol": "1e-6",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "DE": {
                "common": {
                    "archive_size": 10,
                    "success_history_size": 50
                },
                "mutation_type": {
                    "Adaptive": {
                        "strategy": "Best2Bin",
                        "f_min": 0.4,
                        "f_max": 0.9,
                        "cr_min": 0.1,
                        "cr_max": 0.9,
                        "use_jade": false,
                        "memory_size": 5,
                        "learning_rate": 0.1
                    }
                }
            }
        }
    }"#;

    let config = Config::new(conf_json).unwrap();

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
    let mut encoder = setup_gif("examples/gifs/de_kbf.gif")?;

    for frame in 0..50 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "Differential Evolution",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/de_frame.png",
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
        let img = ImageReader::open("examples/de_frame.png")?
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

    std::fs::remove_file("examples/de_frame.png")?;

    Ok(())
}
