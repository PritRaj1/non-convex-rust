pub enum SwapCheck {
    Periodic(Periodic),
    Stochastic(Stochastic),
    Always(Always),
}

pub struct Periodic {
    pub swap_frequency: f64,
    pub total_steps: usize,
}

impl Periodic {
    pub fn new(swap_frequency: f64, total_steps: usize) -> Self {
        Self {
            swap_frequency,
            total_steps,
        }
    }

    pub fn should_swap(&self, current_step: usize) -> bool {
        current_step % (self.swap_frequency * self.total_steps as f64) as usize == 0
    }
}

pub struct Stochastic {
    pub swap_probability: f64,
}

impl Stochastic {
    pub fn new(swap_probability: f64) -> Self {
        Self { swap_probability }
    }

    pub fn should_swap(&self, _current_step: usize) -> bool {
        rand::random::<f64>() < self.swap_probability
    }
}

pub struct Always {}

impl Always {
    pub fn new() -> Self {
        Self {}
    }

    pub fn should_swap(&self, _current_step: usize) -> bool {
        true
    }
}
