mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::prelude::*;

use common::fcns::{KBFConstraints, KBF};
use common::img::{
    create_contour_data, find_closest_color, get_color_palette, setup_chart, setup_gif,
};

use non_convex_opt::utils::config::{AlgConf, Config, NelderMeadConf, OptConf};
use non_convex_opt::NonConvexOpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }),
    };

    let obj_f = KBF;
    let constraints = KBFConstraints;

    // Create initial simplex directly
    let init_simplex = SMatrix::<f64, 3, 2>::from_rows(&[
        SMatrix::<f64, 1, 2>::from_row_slice(&[1.8, 1.0]),
        SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 4.0]),
        SMatrix::<f64, 1, 2>::from_row_slice(&[3.0, 3.0]),
    ]);

    let mut opt = NonConvexOpt::new(
        config,
        init_simplex,
        obj_f.clone(),
        Some(constraints.clone()),
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/nm_kbf.gif")?;

    for frame in 0..10 {
        let mut chart = setup_chart(
            frame,
            "Nelder-Mead Simplex",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/nm_frame.png",
        )?;

        // Get current simplex vertices and best point
        let vertices = opt
            .alg
            .get_simplex()
            .expect("Expected NelderMead algorithm");

        // Draw simplex vertices
        for vertex in vertices {
            chart.draw_series(std::iter::once(Circle::new(
                (vertex[0], vertex[1]),
                4,
                RGBColor(255, 0, 0).filled(),
            )))?;
        }

        // Draw lines connecting simplex vertices
        for i in 0..vertices.len() {
            let j = (i + 1) % vertices.len();
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (vertices[i][0], vertices[i][1]),
                    (vertices[j][0], vertices[j][1]),
                ],
                RGBColor(100, 100, 255).stroke_width(2),
            )))?;
        }

        // Draw current best point
        chart.draw_series(std::iter::once(Circle::new(
            (opt.get_best_individual()[0], opt.get_best_individual()[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;

        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/nm_frame.png")?
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
        frame.delay = 40;
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/nm_frame.png")?;

    Ok(())
}
