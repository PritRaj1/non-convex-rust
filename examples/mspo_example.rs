mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;
use serde_json;

use common::fcns::{KBFConstraints, KBF};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif,
};

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "MSPO": {
                "num_swarms": 10,
                "swarm_size": 10,
                "c1": 1.5,
                "c2": 1.5,
                "x_min": 0.0,
                "x_max": 10.0,
                "exchange_interval": 20,
                "exchange_ratio": 0.05,
                "inertia_start": 0.9,
                "inertia_end": 0.7
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let mut init_pop = SMatrix::<f64, 100, 2>::zeros();
    for i in 0..100 {
        for j in 0..2 {
            let r1 = rand::random::<f64>();
            let r2 = rand::random::<f64>();
            init_pop[(i, j)] = r1 * r2 * 10.0;
        }
    }

    let obj_f = KBF;
    let constraints = KBFConstraints;

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints.clone()));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/mspo_kbf.gif")?;

    let num_frames = 50;
    let frame_delay = 10;

    for frame in 0..num_frames {
        let mut chart = setup_chart(
            frame,
            "MSPO",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/mspo_frame.png",
        )?;

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
        let img = ImageReader::open("examples/mspo_frame.png")?
            .decode()?
            .into_rgb8();

        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }

        let mut frame = Frame::default();
        frame.width = 800;
        frame.height = 800;
        frame.delay = frame_delay;
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/mspo_frame.png")?;

    Ok(())
}
