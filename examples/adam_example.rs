mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{BoxConstraints, MultiModalFunction};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif, ChartParams,
};
use non_convex_opt::utils::config::{AdamConf, AlgConf, Config, OptConf};
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 0.0,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::Adam(AdamConf {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip: 0.0,
            amsgrad: false,
        }),
    };

    let obj_f = MultiModalFunction;
    let constraints = BoxConstraints;

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
    let mut encoder = setup_gif("examples/gifs/adam_kbf.gif")?;

    for frame in 0..100 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "Adam",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/adam_frame.png",
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
        let img = ImageReader::open("examples/adam_frame.png")?
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

    std::fs::remove_file("examples/adam_frame.png")?;

    Ok(())
}
