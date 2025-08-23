mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{Kbf, KbfConstraints};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif, ChartParams,
};

use non_convex_opt::utils::config::{AlgConf, Config, GRASPConf, OptConf};
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 50,
            rtol: 1e-6,
            atol: 0.0,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::GRASP(GRASPConf {
            num_candidates: 100,
            alpha: 0.5,
            num_neighbors: 50,
            step_size: 0.2,
            perturbation_prob: 0.5,
            max_local_iter: 100,
            cache_bounds: true,
            diversity_prob: 0.7,
            restart_threshold: 15,
            diversity_strength: 10.0,
        }),
    };

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
        42,
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/grasp_kbf.gif")?;

    for frame in 0..50 {
        let mut chart = setup_chart(ChartParams {
            frame,
            algorithm_name: "GRASP",
            resolution,
            z_values: &z_values,
            min_val,
            max_val,
            constraints: &constraints,
            frame_path: "examples/grasp_frame.png",
        })?;

        // Draw current solution in red
        let population = opt.get_population();
        let current_x = population.row(0);
        chart.draw_series(std::iter::once(Circle::new(
            (current_x[0], current_x[1]),
            6,
            RGBColor(255, 0, 0).filled(),
        )))?;

        // Draw best solution in yellow
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;

        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/grasp_frame.png")?
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

    std::fs::remove_file("examples/grasp_frame.png")?;

    Ok(())
}
