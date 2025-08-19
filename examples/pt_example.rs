mod common;

use gif::Frame;
use image::ImageReader;
use nalgebra::SMatrix;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;

use common::fcns::{KbfConstraints, Kbf};
use common::img::{create_contour_data, find_closest_color, get_color_palette, setup_gif};

use non_convex_opt::utils::config::Config;
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};
use non_convex_opt::NonConvexOpt;

type ContourData = Vec<Vec<(f64, f64)>>;
type BestPoints = Vec<(f64, f64)>;
type Temperatures = Vec<f64>;

fn get_replica_data(
    opt: &NonConvexOpt<f64, nalgebra::Const<50>, nalgebra::Const<2>>,
    obj_f: &Kbf,
) -> (ContourData, BestPoints, Temperatures) {
    let replica_populations = opt
        .get_pt_replica_populations()
        .expect("This example requires Parallel Tempering algorithm");

    let replica_temperatures = opt
        .get_pt_replica_temperatures()
        .expect("This example requires Parallel Tempering algorithm");

    let mut replica_pops = Vec::new();
    let mut replica_bests = Vec::new();

    for replica_pop in &replica_populations {
        let mut replica_points = Vec::new();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_point = (0.0, 0.0);

        for row_idx in 0..replica_pop.nrows() {
            let row = replica_pop.row(row_idx);
            let point = (row[0], row[1]);
            replica_points.push(point);

            let fitness = obj_f.f(&nalgebra::SVector::<f64, 2>::from_vec(vec![row[0], row[1]]));
            if fitness > best_fitness {
                best_fitness = fitness;
                best_point = point;
            }
        }

        replica_pops.push(replica_points);
        replica_bests.push(best_point);
    }

    (replica_pops, replica_bests, replica_temperatures)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf_json = r#"
    {
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "0.0"
        },
        "alg_conf": {
            "PT": {
                "common": {
                    "num_replicas": 5,
                    "power_law_init": 2.0,
                    "power_law_final": 0.5,
                    "power_law_cycles": 0,
                    "alpha": 0.3,
                    "omega": 2.1,
                    "mala_step_size": 0.1
                },
                "swap_conf": {
                    "Periodic": {
                        "swap_frequency": 0.3
                    }
                },
                "update_conf": {
                "PCN": {
                    "step_size": 0.2,
                    "preconditioner": 1.0
                }
            }
            }
        }
    }
    "#;

    let config = Config::new(conf_json).unwrap();

    let obj_f = Kbf;
    let constraints = KbfConstraints;

    let mut init_pop = SMatrix::<f64, 50, 2>::zeros();
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 10.0;
        }
    }

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints.clone()));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/pt_kbf.gif")?;

    for _frame in 0..100 {
        let root = BitMapBackend::new("examples/pt_frame.png", (2000, 400)).into_drawing_area();
        root.fill(&WHITE)?;

        let replica_areas = root.split_evenly((1, 5));

        let (replica_populations, replica_bests, replica_temperatures) =
            get_replica_data(&opt, &obj_f);

        let replica_colors = [
            RGBColor(255, 0, 0),
            RGBColor(255, 0, 0),
            RGBColor(255, 0, 0),
            RGBColor(255, 0, 0),
            RGBColor(255, 0, 0),
        ];

        for (replica_idx, area) in replica_areas.iter().enumerate() {
            let mut chart = ChartBuilder::on(area)
                .caption(
                    format!(
                        "Replica {} (T={:.3})",
                        replica_idx,
                        replica_temperatures.get(replica_idx).unwrap_or(&0.0)
                    ),
                    ("sans-serif", 20),
                )
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(0.0..10.0, 0.0..10.0)?;

            chart.configure_mesh().draw()?;

            for (i, _) in z_values.iter().enumerate().take(resolution - 1) {
                for (j, _) in z_values[i].iter().enumerate().take(resolution - 1) {
                    let x = 10.0 * i as f64 / (resolution - 1) as f64;
                    let y = 10.0 * j as f64 / (resolution - 1) as f64;
                    let dx = 10.0 / (resolution - 1) as f64;
                    let val = (z_values[i][j] - min_val) / (max_val - min_val);
                    let color = RGBColor(
                        (255.0 * val) as u8,
                        (255.0 * val) as u8,
                        (255.0 * val) as u8,
                    );

                    let point = nalgebra::SVector::<f64, 2>::from_vec(vec![x, y]);
                    if !constraints.g(&point) {
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

            for (i, _) in z_values.iter().enumerate().take(resolution - 1) {
                for (j, _) in z_values[i].iter().enumerate().take(resolution - 1) {
                    let x = 10.0 * i as f64 / (resolution - 1) as f64;
                    let y = 10.0 * j as f64 / (resolution - 1) as f64;

                    let point = nalgebra::SVector::<f64, 2>::from_vec(vec![x, y]);
                    if !constraints.g(&point) {
                        let stripe_width = 0.2;
                        let stripe_pos = ((x + y) / stripe_width).floor() as i32;
                        if stripe_pos % 2 == 0 {
                            chart.draw_series(std::iter::once(Rectangle::new(
                                [
                                    (x, y),
                                    (
                                        x + 10.0 / (resolution - 1) as f64,
                                        y + 10.0 / (resolution - 1) as f64,
                                    ),
                                ],
                                RGBColor(100, 100, 100).filled(),
                            )))?;
                        }
                    }
                }
            }

            if replica_idx < replica_populations.len() {
                chart.draw_series(
                    replica_populations[replica_idx].iter().map(|&(x, y)| {
                        Circle::new((x, y), 3, replica_colors[replica_idx].filled())
                    }),
                )?;
            }

            if replica_idx < replica_bests.len() {
                let best_point = replica_bests[replica_idx];
                chart.draw_series(std::iter::once(Circle::new(
                    best_point,
                    6,
                    RGBColor(255, 255, 0).filled(),
                )))?;
            }
        }

        root.present()?;

        let img = ImageReader::open("examples/pt_frame.png")?
            .decode()?
            .into_rgba8();

        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }

        let frame = Frame::<'_> {
            width: 2000,
            height: 400,
            delay: 5,
            buffer: std::borrow::Cow::from(indexed_pixels),
            ..Default::default()
        };
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/pt_frame.png")?;

    Ok(())
}
