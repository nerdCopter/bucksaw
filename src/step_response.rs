use realfft::num_complex::Complex32;
use std::collections::VecDeque; // Added for moving average history

// Constants for step response calculation
const FRAME_LENGTH_S: f64 = 2.0; // Length of each window in seconds
const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep (500ms)
const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows
const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)
const INITIAL_GYRO_SMOOTHING_WINDOW: usize = 15; // Initial Gyro Smoothing
const MOVEMENT_THRESHOLD_DEG_S: f32 = 20.0; // Minimum setpoint magnitude

// Quality control constants
const APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION: bool = true; // Individual Response "Y-Correction"
const Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS: f32 = 0.1; // Minimum absolute mean for Y-correction
const NORMALIZED_STEADY_STATE_MIN_VAL: f32 = 0.5; // Min steady-state value after normalization
const NORMALIZED_STEADY_STATE_MAX_VAL: f32 = 3.0; // Max steady-state value after normalization
const ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK: bool = true; // Enable additional check on mean
const NORMALIZED_STEADY_STATE_MEAN_MIN: f32 = 0.75; // Min steady-state mean after normalization
const NORMALIZED_STEADY_STATE_MEAN_MAX: f32 = 1.25; // Max steady-state mean after normalization
const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check
const STEADY_STATE_END_S: f64 = 0.5; // End time for steady-state check

fn fft_forward(data: &[f32]) -> Vec<Complex32> {
    // Ensure input is not empty
    if data.is_empty() {
        return Vec::new();
    }
    let mut input = data.to_vec();
    let planner = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(input.len());
    let mut output = planner.make_output_vec();
    // Use a match or expect for better error handling if desired
    let _ = planner.process(&mut input, &mut output); // Error ignored for simplicity like original
    output
}

// Corrected fft_inverse to use the original real signal length N
fn fft_inverse(data: &[Complex32], original_length_n: usize) -> Vec<f32> {
    // Ensure input is not empty and N is valid
    if data.is_empty() || original_length_n == 0 {
        return vec![0.0; original_length_n];
    }
    let mut input = data.to_vec();
    // The inverse planner needs the length of the original real signal (N)
    let planner = realfft::RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec(); // Output will have length N
    // Check if planner input length matches provided data length
    // Required complex input length depends on N (even/odd)
    let expected_complex_len = if original_length_n % 2 == 0 {
        original_length_n / 2 + 1
    } else {
        (original_length_n + 1) / 2
    };
    if input.len() != expected_complex_len {
        // Length mismatch, cannot perform inverse FFT correctly
        eprintln!("Warning: FFT inverse length mismatch. Expected complex length {}, got {}. Returning zeros.", expected_complex_len, input.len());
        return vec![0.0; original_length_n];
    }

    if planner.process(&mut input, &mut output).is_ok() {
        // Normalize the IFFT output (realfft doesn't normalize by default)
        let scale = 1.0 / original_length_n as f32;
        output.iter_mut().for_each(|x| *x *= scale);
        output
    } else {
        // Error during processing
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        vec![0.0; original_length_n]
    }
}

/// Creates a Tukey window for signal tapering with configurable alpha parameter.
fn tukeywin(num: usize, alpha: f64) -> Vec<f32> {
    if alpha <= 0.0 { 
        return vec![1.0; num]; 
    } else if alpha >= 1.0 {
        // Full Hanning window when alpha = 1.0
        let mut window = vec![0.0; num];
        for i in 0..num { 
            window[i] = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (num as f64 - 1.0)).cos()) as f32; 
        }
        return window;
    }
    
    let mut window = vec![1.0; num];
    let alpha_half = alpha / 2.0; 
    let n_alpha = (alpha_half * (num as f64 - 1.0)).floor() as usize;
    
    // Apply cosine tapering to both ends
    for i in 0..n_alpha {
        window[i] = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / (n_alpha as f64)).cos()) as f32;
        window[num - 1 - i] = window[i];
    }
    window
}

/// Generates overlapping windowed segments from input/output data for parallel analysis.
fn winstacker_contiguous(input_data: &[f32], output_data: &[f32], frame_length_samples: usize, superposition_factor: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let total_len = input_data.len();
    if total_len == 0 || frame_length_samples == 0 || superposition_factor == 0 { 
        return (Vec::new(), Vec::new()); 
    }
    
    let shift = frame_length_samples / superposition_factor;
    if shift == 0 { 
        eprintln!("Warning: Window shift is zero."); 
        return (Vec::new(), Vec::new()); 
    }
    
    let num_windows = if total_len >= frame_length_samples { 
        (total_len - frame_length_samples) / shift + 1 
    } else { 
        0 
    };
    
    if num_windows == 0 { 
        return (Vec::new(), Vec::new()); 
    }
    
    let mut stacked_input = Vec::with_capacity(num_windows);
    let mut stacked_output = Vec::with_capacity(num_windows);
    
    for i in 0..num_windows {
        let start = i * shift;
        let end = start + frame_length_samples;
        stacked_input.push(input_data[start..end].to_vec());
        stacked_output.push(output_data[start..end].to_vec());
    }
    
    (stacked_input, stacked_output)
}

/// Performs Wiener deconvolution to extract system impulse response from input/output signals.
fn wiener_deconvolution_window(input_window: &[f32], output_window: &[f32], _sample_rate: f64) -> Vec<f32> {
    let n = input_window.len(); 
    if n == 0 { 
        return Vec::new(); 
    }
    
    let padded_n = n.next_power_of_two();
    
    let mut input_padded = vec![0.0f32; padded_n];
    input_padded[0..n].copy_from_slice(input_window);
    
    let mut output_padded = vec![0.0f32; padded_n];
    output_padded[0..n].copy_from_slice(output_window);
    
    let h_spec = fft_forward(&input_padded);
    let g_spec = fft_forward(&output_padded);
    
    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() { 
        eprintln!("Warning: FFT output empty/mismatch in Wiener deconvolution."); 
        return vec![0.0; n]; 
    }
    
    // Apply regularization to prevent division by near-zero values
    let regularization_term = 0.0001; 
    let epsilon = 1e-9;
    
    let mut deconvolved_spec = Vec::with_capacity(h_spec.len());
    
    for i in 0..h_spec.len() {
        let h = h_spec[i]; 
        let g = g_spec[i]; 
        let h_conj = h.conj();
        
        let denominator = (h * h_conj).re + regularization_term;
        
        if denominator.abs() > epsilon { 
            deconvolved_spec.push((g * h_conj) / denominator); 
        } else { 
            deconvolved_spec.push(Complex32::new(0.0, 0.0)); 
        }
    }
    
    let deconvolved_impulse = fft_inverse(&deconvolved_spec, padded_n);
    
    // Return only the first n samples
    deconvolved_impulse[0..n].to_vec()
}

/// Converts impulse response to step response via cumulative integration.
fn cumulative_sum(data: &[f32]) -> Vec<f32> {
    let mut cumulative = vec![0.0; data.len()]; 
    let mut current_sum = 0.0;
    
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() { 
            current_sum += val; 
        } else { 
            eprintln!("Warning: Non-finite impulse value ({}) at index {}.", val, i); 
        }
        cumulative[i] = current_sum;
    }
    
    cumulative
}

// Helper function for moving average smoothing
fn moving_average_smooth(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_vec(); // No smoothing needed or possible
    }

    let mut smoothed_data = Vec::with_capacity(data.len());
    let mut current_sum: f32 = 0.0;
    // Using VecDeque to efficiently manage the sliding window sum
    let mut history: VecDeque<f32> = VecDeque::with_capacity(window_size);

    for &val in data.iter() {
        history.push_back(val);
        current_sum += val;

        // If the window is full, remove the oldest element from the sum and the deque
        if history.len() > window_size {
            current_sum -= history.pop_front().unwrap(); // unwrap is safe due to check
        }

        // Calculate the average over the current window contents
        // The effective window size grows until it reaches `window_size`
        smoothed_data.push(current_sum / (history.len() as f32));
    }

    smoothed_data
}

/// Averages multiple step responses for final result.
fn average_responses(stacked_responses: &[Vec<f32>], weights: &[f32], response_len: usize) -> Option<Vec<f64>> {
    let num_windows = stacked_responses.len();
    if num_windows == 0 || response_len == 0 || weights.len() != num_windows {
        return None;
    }
    
    let mut averaged_response = vec![0.0; response_len];
    let mut active_window_counts = vec![0.0; response_len];
    
    for i in 0..num_windows {
        let weight = weights[i] as f64; 
        if weight <= 1e-9 { 
            continue; 
        }
        
        let response = &stacked_responses[i];
        for j in 0..response_len.min(response.len()) {
            let response_value = response[j] as f64;
            if response_value.is_finite() { 
                averaged_response[j] += response_value * weight; 
                active_window_counts[j] += weight; 
            }
        }
    }
    
    for j in 0..response_len { 
        if active_window_counts[j] > 1e-9 { 
            averaged_response[j] /= active_window_counts[j]; 
        } 
    }
    
    Some(averaged_response)
}

pub fn calculate_step_response(
    times: &[f64],
    setpoint: &[f32],
    gyro_filtered: &[f32],
    sample_rate: f64,
) -> Vec<(f64, f64)> {
    // Basic validation
    if times.is_empty() || setpoint.is_empty() || gyro_filtered.is_empty() || 
       setpoint.len() != gyro_filtered.len() || times.len() != setpoint.len() || sample_rate <= 0.0 {
        eprintln!("Warning: Invalid input to calculate_step_response. Empty data, length mismatch, or invalid sample rate.");
        return Vec::new(); // Return empty if inputs are invalid
    }

    // Apply initial smoothing to gyro data if configured
    let gyro_processed = if INITIAL_GYRO_SMOOTHING_WINDOW > 1 {
        moving_average_smooth(gyro_filtered, INITIAL_GYRO_SMOOTHING_WINDOW)
    } else {
        gyro_filtered.to_vec()
    };

    // Calculate frame and response lengths in samples
    let frame_length_samples = (FRAME_LENGTH_S * sample_rate).ceil() as usize;
    let response_length_samples = (RESPONSE_LENGTH_S * sample_rate).ceil() as usize;

    if frame_length_samples == 0 || response_length_samples == 0 {
        eprintln!("Warning: Calculated window length is zero.");
        return Vec::new();
    }

    // Define steady-state region for quality control checks
    let ss_start_sample = (STEADY_STATE_START_S * sample_rate).floor() as usize;
    let ss_end_sample = (STEADY_STATE_END_S * sample_rate).ceil() as usize;
    let effective_ss_start_sample = ss_start_sample.min(response_length_samples.saturating_sub(1));
    let effective_ss_end_sample = ss_end_sample.min(response_length_samples).max(effective_ss_start_sample + 1);

    let mut stacked_step_responses_qc: Vec<Vec<f32>> = Vec::new();
    let mut window_max_setpoints_qc: Vec<f32> = Vec::new();

    // Generate overlapping windows for parallel processing
    let (stacked_input_raw, stacked_output_raw) = winstacker_contiguous(
        setpoint, &gyro_processed, frame_length_samples, SUPERPOSITION_FACTOR);

    let num_windows = stacked_input_raw.len();
    if num_windows == 0 {
        eprintln!("Warning: No complete windows generated.");
        return Vec::new();
    }

    let window_func = tukeywin(frame_length_samples, TUKEY_ALPHA);

    // Process each window through deconvolution and quality control
    for i in 0..num_windows {
        let input_window_raw = &stacked_input_raw[i];
        let output_window_raw = &stacked_output_raw[i];

        // Filter out low-activity windows based on setpoint magnitude
        let max_setpoint_in_window = input_window_raw.iter().fold(0.0f32, |max_val, &v| max_val.max(v.abs()));
        if max_setpoint_in_window < MOVEMENT_THRESHOLD_DEG_S {
            continue;
        }

        // Apply windowing function to reduce spectral leakage
        let input_window_windowed: Vec<f32> = input_window_raw.iter().zip(window_func.iter())
                                             .map(|(&x, &w)| x * w)
                                             .collect();
        let output_window_windowed: Vec<f32> = output_window_raw.iter().zip(window_func.iter())
                                              .map(|(&x, &w)| x * w)
                                              .collect();

        // Extract impulse response via Wiener deconvolution
        let impulse_response = wiener_deconvolution_window(&input_window_windowed, &output_window_windowed, sample_rate);
        if impulse_response.len() < frame_length_samples {
            continue;
        }

        // Truncate impulse response to frame length
        let impulse_response_truncated = impulse_response[0..frame_length_samples].to_vec();
        if impulse_response_truncated.is_empty() { 
            continue; 
        }

        // Convert impulse to step response and truncate to desired length
        let unnormalized_step_response = cumulative_sum(&impulse_response_truncated);
        if unnormalized_step_response.len() < response_length_samples { 
            continue; 
        }
        
        let truncated_unnormalized_response = unnormalized_step_response[0..response_length_samples].to_vec();
        let mut response_for_qc = truncated_unnormalized_response.clone();
        let mut y_correction_attempted_and_valid_for_qc = false;

        // Apply Y-correction (normalization) if enabled and conditions are met
        if APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION {
            if effective_ss_start_sample < effective_ss_end_sample && truncated_unnormalized_response.len() >= effective_ss_end_sample {
                // Calculate mean of the steady-state segment
                let unnorm_ss_segment = &truncated_unnormalized_response[effective_ss_start_sample..effective_ss_end_sample];
                if !unnorm_ss_segment.is_empty() {
                    let unnorm_ss_sum: f32 = unnorm_ss_segment.iter().sum();
                    let unnorm_ss_mean = unnorm_ss_sum / unnorm_ss_segment.len() as f32;
                    
                    if unnorm_ss_mean.is_finite() && unnorm_ss_mean.abs() > Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS {
                        // Normalize response to target steady-state of 1.0
                        response_for_qc = truncated_unnormalized_response.iter().map(|v| v / unnorm_ss_mean).collect();
                        y_correction_attempted_and_valid_for_qc = true;
                    }
                }
            }
        } else {
            // If Y-correction is not applied, QC will run on the unnormalized response.
            y_correction_attempted_and_valid_for_qc = true; // Allow QC to proceed on unnormalized data
        }

        // Quality control checks
        let mut current_window_passes_qc = false;
        if (APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION && y_correction_attempted_and_valid_for_qc) || !APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION {
            if effective_ss_start_sample < effective_ss_end_sample && response_for_qc.len() >= effective_ss_end_sample {
                let qc_ss_segment = &response_for_qc[effective_ss_start_sample..effective_ss_end_sample];
                if !qc_ss_segment.is_empty() {
                    let min_val_ss = qc_ss_segment.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max_val_ss = qc_ss_segment.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    
                    // Calculate mean for optional check
                    let ss_sum: f32 = qc_ss_segment.iter().sum();
                    let mean_val_ss = ss_sum / qc_ss_segment.len() as f32;

                    // Check min/max bounds for steady-state region
                    if min_val_ss.is_finite() && max_val_ss.is_finite() &&
                       min_val_ss > NORMALIZED_STEADY_STATE_MIN_VAL && max_val_ss < NORMALIZED_STEADY_STATE_MAX_VAL {
                        // Optional additional check on steady-state mean
                        if ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK {
                            if mean_val_ss.is_finite() &&
                               mean_val_ss > NORMALIZED_STEADY_STATE_MEAN_MIN &&
                               mean_val_ss < NORMALIZED_STEADY_STATE_MEAN_MAX {
                                current_window_passes_qc = true;
                            }
                        } else {
                            current_window_passes_qc = true;
                        }
                    }
                }
            }
        }

        // Collect windows that pass quality control
        if current_window_passes_qc {
            stacked_step_responses_qc.push(response_for_qc);
            window_max_setpoints_qc.push(max_setpoint_in_window);
        }
    }

    if stacked_step_responses_qc.is_empty() {
        eprintln!("Warning: No windows passed quality control.");
        return Vec::new();
    }

    // Create weights vector (all 1.0 for equal weighting)
    let weights = vec![1.0; stacked_step_responses_qc.len()];
    
    // Average the quality-controlled responses
    let combined_response = match average_responses(&stacked_step_responses_qc, &weights, response_length_samples) {
        Some(response) => response,
        None => {
            eprintln!("Warning: Failed to average responses.");
            return Vec::new();
        }
    };

    // Apply post-processing: shift to start at 0 and normalize
    let first_val = combined_response[0];
    let mut shifted_response: Vec<f64> = combined_response.iter().map(|&v| v - first_val).collect();
    
    // Calculate the steady-state mean for final normalization
    if effective_ss_start_sample < effective_ss_end_sample && shifted_response.len() >= effective_ss_end_sample {
        let ss_segment = &shifted_response[effective_ss_start_sample..effective_ss_end_sample];
        if !ss_segment.is_empty() {
            let ss_sum: f64 = ss_segment.iter().sum();
            let ss_mean = ss_sum / ss_segment.len() as f64;
            
            if ss_mean.abs() > 1e-9 {
                // Normalize to make steady-state approach 1.0
                for val in &mut shifted_response {
                    *val /= ss_mean;
                }
            }
        }
    }

    // Create time vector for the response (starting from 0)
    let start_time = times.first().cloned().unwrap_or(0.0);
    let time_step = 1.0 / sample_rate;
    let response_times: Vec<f64> = (0..response_length_samples)
        .map(|i| i as f64 * time_step)
        .collect();

    // Combine time data with the normalized step response
    response_times
        .iter()
        .zip(shifted_response.iter())
        .map(|(&t, &s)| (t, s))
        .collect()
}
