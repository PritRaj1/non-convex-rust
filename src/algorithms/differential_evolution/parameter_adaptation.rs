use rand::Rng;

pub enum ParameterAdaptationType {
    JADE(JADEParameterAdaptation),
    Standard(StandardParameterAdaptation),
}

impl ParameterAdaptationType {
    pub fn generate_parameters(&self) -> (f64, f64) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.generate_jade_parameters(),
            ParameterAdaptationType::Standard(standard) => standard.get_parameters(),
        }
    }

    pub fn record_success(&mut self, f: f64, cr: f64) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.record_success(f, cr),
            ParameterAdaptationType::Standard(_) => {
                // Standard adaptation doesn't track individual successes
            }
        }
    }

    pub fn update_parameters(&mut self) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.update_memory(),
            ParameterAdaptationType::Standard(_) => {
                // Standard adaptation updates are handled externally
            }
        }
    }
}

pub struct JADEParameterAdaptation {
    pub f_memory: Vec<f64>,
    pub cr_memory: Vec<f64>,
    pub memory_pointer: usize,
    pub successful_f: Vec<f64>,
    pub successful_cr: Vec<f64>,
    pub memory_size: usize,
}

impl JADEParameterAdaptation {
    pub fn new(memory_size: usize) -> Self {
        let mut f_memory = vec![0.5; memory_size];
        let mut cr_memory = vec![0.5; memory_size];

        // Initialize with random values
        for i in 0..memory_size {
            f_memory[i] = rand::random::<f64>() * 0.5 + 0.25;
            cr_memory[i] = rand::random::<f64>() * 0.5 + 0.25;
        }

        Self {
            f_memory,
            cr_memory,
            memory_pointer: 0,
            successful_f: Vec::new(),
            successful_cr: Vec::new(),
            memory_size,
        }
    }

    pub fn generate_jade_parameters(&self) -> (f64, f64) {
        let mut rng = rand::rng();
        let memory_idx = rng.random_range(0..self.f_memory.len()); // Random mem index

        // Sample from Cauchy distribution around memory value
        let f_memory = self.f_memory[memory_idx];
        let f = loop {
            let f_candidate = f_memory + 0.1 * (2.0 * rng.random::<f64>() - 1.0);
            if (0.1..=1.0).contains(&f_candidate) {
                break f_candidate;
            }
        };

        // CR using normal distribution
        let cr_memory = self.cr_memory[memory_idx];
        let cr = loop {
            let cr_candidate = cr_memory + 0.1 * (2.0 * rng.random::<f64>() - 1.0);
            if (0.0..=1.0).contains(&cr_candidate) {
                break cr_candidate;
            }
        };

        (f, cr)
    }

    pub fn record_success(&mut self, f: f64, cr: f64) {
        self.successful_f.push(f);
        self.successful_cr.push(cr);
    }

    pub fn update_memory(&mut self) {
        if self.successful_f.is_empty() || self.successful_cr.is_empty() {
            return;
        }

        // Update mem using Lehmer mean (arithmetic mean is more sensitive to outliers)
        let f_mean = self.lehmer_mean(&self.successful_f);
        let cr_mean = self.lehmer_mean(&self.successful_cr);

        // Update at current pointer
        self.f_memory[self.memory_pointer] = f_mean;
        self.cr_memory[self.memory_pointer] = cr_mean;

        // Move pointer
        self.memory_pointer = (self.memory_pointer + 1) % self.f_memory.len();
        self.successful_f.clear();
        self.successful_cr.clear();
    }

    fn lehmer_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.5;
        }
        let sum_squares: f64 = values.iter().map(|&x| x * x).sum();
        let sum: f64 = values.iter().sum();
        if sum.abs() < 1e-10 {
            0.5
        } else {
            sum_squares / sum
        }
    }
}

pub struct StandardParameterAdaptation {
    pub current_f: f64,
    pub current_cr: f64,
    pub f_min: f64,
    pub f_max: f64,
    pub cr_min: f64,
    pub cr_max: f64,
}

impl StandardParameterAdaptation {
    pub fn new(f: f64, cr: f64, f_min: f64, f_max: f64, cr_min: f64, cr_max: f64) -> Self {
        Self {
            current_f: f,
            current_cr: cr,
            f_min,
            f_max,
            cr_min,
            cr_max,
        }
    }

    pub fn update_parameters(&mut self, success_rate: f64) {
        self.current_f = self.f_min + success_rate * (self.f_max - self.f_min);
        self.current_cr = self.cr_min + success_rate * (self.cr_max - self.cr_min);
    }

    pub fn get_parameters(&self) -> (f64, f64) {
        (self.current_f, self.current_cr)
    }
}
