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
            "max_iter": 50,
            "rtol": "1e-6",
            "atol": "0.0",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "CMCGS": {
                "epsilon": 0.6,
                "expansion_threshold": 4,
                "max_nodes_per_layer": 6,
                "max_depth": 3,
                "simulation_count": 6,
                "simulation_steps": 4,
                "discount_factor": 0.85,
                "restart_threshold": 8,
                "top_experiences_count": 2,
                "restart_max_attempts": 1000
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let obj_f = Kbf;
    let constraints = KbfConstraints;

    let mut opt = NonConvexOpt::new(
        config,
        SMatrix::<f64, 1, 2>::from_row_slice(&[4.0, 9.0]),
        obj_f.clone(),
        Some(constraints.clone()),
        42,
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/cmcgs_kbf.gif")?;

    for frame in 0..50 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "CMCGS",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/cmcgs_frame.png",
        })?;

        // Draw best individual in yellow
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        // Save frame and convert to GIF
        chart.plotting_area().present()?;

        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/cmcgs_frame.png")?
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
            delay: 4,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/cmcgs_frame.png")?;

    Ok(())
}
