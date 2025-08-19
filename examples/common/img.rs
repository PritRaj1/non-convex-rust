use gif::{Encoder, Repeat};
use nalgebra::{SVector, U2};
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::fs::File;

pub fn create_contour_data<F: ObjectiveFunction<f64, U2>>(
    obj_f: &F,
    resolution: usize,
) -> (Vec<Vec<f64>>, f64, f64) {
    let mut z = vec![vec![0.0; resolution]; resolution];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for (i, z_row) in z.iter_mut().enumerate().take(resolution) {
        for (j, z_val) in z_row.iter_mut().enumerate().take(resolution) {
            let x = 10.0 * i as f64 / (resolution - 1) as f64;
            let y = 10.0 * j as f64 / (resolution - 1) as f64;
            let point = SVector::<f64, 2>::from_vec(vec![x, y]);
            let val = obj_f.f(&point);
            *z_val = val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }
    (z, min_val, max_val)
}

pub fn setup_gif(filename: &str) -> Result<Encoder<File>, Box<dyn std::error::Error>> {
    let gif = File::create(filename)?;
    let color_palette = get_color_palette();
    let mut encoder = Encoder::new(gif, 800, 800, &color_palette)?;
    encoder.set_repeat(Repeat::Infinite)?;
    Ok(encoder)
}

pub fn find_closest_color(r: u8, g: u8, b: u8, palette: &[u8]) -> usize {
    let mut best_diff = f64::INFINITY;
    let mut best_idx = 0;

    for i in (0..palette.len()).step_by(3) {
        let pr = palette[i];
        let pg = palette[i + 1];
        let pb = palette[i + 2];

        let diff = ((r as f64 - pr as f64).powi(2)
            + (g as f64 - pg as f64).powi(2)
            + (b as f64 - pb as f64).powi(2))
        .sqrt();

        if diff < best_diff {
            best_diff = diff;
            best_idx = i / 3;
        }
    }
    best_idx
}

pub struct ChartParams<'a> {
    pub frame: usize,
    pub algorithm_name: &'a str,
    pub resolution: usize,
    pub z_values: &'a [Vec<f64>],
    pub min_val: f64,
    pub max_val: f64,
    pub constraints: &'a dyn BooleanConstraintFunction<f64, U2>,
    pub frame_path: &'a str,
}

#[allow(dead_code)]
pub fn setup_chart<'a>(
    params: ChartParams<'a>,
) -> Result<
    ChartContext<'a, BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    Box<dyn std::error::Error>,
> {
    let root = BitMapBackend::new(params.frame_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "{}, Keane's Bump Function - Iteration {}",
                params.algorithm_name, params.frame
            ),
            ("sans-serif", 30),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..10f64, 0f64..10f64)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // Draw contour and feasible regions
    for (i, z_row) in params.z_values.iter().enumerate().take(params.resolution - 1) {
        for (j, _) in z_row.iter().enumerate().take(params.resolution - 1) {
            let x = 10.0 * i as f64 / (params.resolution - 1) as f64;
            let y = 10.0 * j as f64 / (params.resolution - 1) as f64;
            let dx = 10.0 / (params.resolution - 1) as f64;
            let val = (params.z_values[i][j] - params.min_val) / (params.max_val - params.min_val);
            let color = RGBColor(
                (255.0 * val) as u8,
                (255.0 * val) as u8,
                (255.0 * val) as u8,
            );

            let point = SVector::<f64, 2>::from_vec(vec![x, y]);
            if !params.constraints.g(&point) {
                let stripe_width = 0.2;
                let stripe_pos = ((x + y) / stripe_width).floor() as i32;
                if stripe_pos % 2 == 0 {
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(x, y), (x + dx, y + dx)],
                        RGBColor(128, 128, 128).mix(0.3).filled(),
                    )))?;
                }
            } else {
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + dx, y + dx)],
                    color.filled(),
                )))?;
            }
        }
    }

    Ok(chart)
}

pub fn get_color_palette() -> Vec<u8> {
    let mut color_palette = Vec::with_capacity(768);

    color_palette.extend_from_slice(&[
        255, 0, 0, // Bright red for current individual
        255, 255, 0, // Bright yellow for best individual
    ]);

    // Add grayscale colors
    for i in 0..254 {
        color_palette.push(i as u8);
        color_palette.push(i as u8);
        color_palette.push(i as u8);
    }

    color_palette
}
