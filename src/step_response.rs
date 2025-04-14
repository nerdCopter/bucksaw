use realfft::num_complex::Complex32;
use std::f32::EPSILON; // Corrected // Use a small constant for division stability

fn fft_forward(data: &[f32]) -> Vec<Complex32> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut input = data.to_vec();
    // Planner requires mutable input buffer, but we have a copy
    let planner = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(input.len());
    // Create output buffer of the correct size
    let mut output = planner.make_output_vec();
    // Process takes mutable slice references
    planner.process(&mut input, &mut output).unwrap();
    output
}

fn fft_inverse(data: &[Complex32], original_len: usize) -> Vec<f32> {
    if data.is_empty() || original_len == 0 {
        return Vec::new();
    }
    let mut input = data.to_vec();
    // Plan inverse FFT using the length of the *original* real signal
    let planner = realfft::RealFftPlanner::<f32>::new().plan_fft_inverse(original_len);
    // Make output buffer of the correct size (original_len)
    let mut output = planner.make_output_vec();

    // Process inverse FFT
    if planner.process(&mut input, &mut output).is_ok() {
        // Inverse FFT often requires scaling by 1/N
        let scale = 1.0 / original_len as f32;
        output.iter_mut().for_each(|x| *x *= scale);
        output
    } else {
        // Provide a zeroed vector of the correct length on error
        vec![0.0; original_len]
    }
}

/// Applies a Simple Moving Average filter to the data.
/// Handles edges by averaging fewer points.
fn smooth_moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_vec(); // No smoothing needed or possible
    }

    let mut smoothed = Vec::with_capacity(data.len());
    let half_window = window_size / 2;

    for i in 0..data.len() {
        // Determine window bounds, clamping to the data array limits
        let start = i.saturating_sub(half_window);
        // End index is exclusive, clamp to data.len()
        let end = (i + half_window + 1).min(data.len());

        // Slice the data within the window
        let window = &data[start..end];

        // Calculate the sum and count for the average
        let sum: f32 = window.iter().sum();
        let count = window.len() as f32;

        if count > 0.0 {
            smoothed.push(sum / count);
        } else {
            // Should not happen if data is not empty, but handle defensively
            smoothed.push(0.0);
        }
    }
    smoothed
}


/// Calculates the step response of a system given input (setpoint) and output (gyro) data.
///
/// # Arguments
/// * `times` - Timestamps corresponding to the data points.
/// * `setpoint` - The input signal (e.g., desired angle or rate).
/// * `gyro_filtered` - The measured output signal (e.g., filtered gyroscope reading).
/// * `sample_rate` - The rate at which data was sampled (samples per second).
/// * `smoothing_window_size` - The number of points to use for the moving average smoothing filter.
///                             A value of 1 disables smoothing. Recommended: 5-21 depending on noise.
///
/// # Returns
/// A vector of (time, smoothed_normalized_step_response) tuples.
pub fn calculate_step_response(
    times: &[f64],
    setpoint: &[f32],
    gyro_filtered: &[f32],
    sample_rate: f64,
    smoothing_window_size: usize, // Added parameter for smoothing
) -> Vec<(f64, f64)> {

    // Ensure inputs have the same length and are not empty
    if setpoint.len() != gyro_filtered.len() || setpoint.len() != times.len() || setpoint.is_empty() {
        eprintln!("Error: Input arrays must have the same non-zero length.");
        return Vec::new();
    }

    let n = setpoint.len(); // Original length of the time-domain signals

    // 1. Calculate FFT of input and output
    let input_spectrum = fft_forward(setpoint);
    let output_spectrum = fft_forward(gyro_filtered);

    // Check if FFT produced expected output length
     // Expected length for real FFT output is N/2 + 1
    let expected_fft_len = n / 2 + 1;
    if input_spectrum.len() != expected_fft_len || output_spectrum.len() != expected_fft_len {
         eprintln!("Error: FFT output length mismatch.");
         return Vec::new();
    }

    // 2. Calculate Frequency Response H(f) = Y(f) / X(f)
    //    Using H(f) = (conj(X(f)) * Y(f)) / (conj(X(f)) * X(f) + epsilon) for stability
    let frequency_response: Vec<Complex32> = input_spectrum
        .iter()
        .zip(output_spectrum.iter())
        .map(|(i_spec, o_spec)| {
            let i_conj = i_spec.conj();
            let denominator = i_conj * i_spec; // This is |i_spec|^2 (real)
            // Add small epsilon to avoid division by zero/near-zero
            let safe_denominator = denominator.re + EPSILON;
            if safe_denominator > EPSILON { // Check if denominator is reasonably non-zero
                (i_conj * o_spec) / safe_denominator
            } else {
                Complex32::new(0.0, 0.0) // Or handle as appropriate, e.g., skip this frequency
            }
        })
        .collect();

    // 3. Calculate Impulse Response h(t) = IFFT(H(f))
    //    Pass the original signal length 'n' to fft_inverse
    let impulse_response = fft_inverse(&frequency_response, n);
    if impulse_response.len() != n {
         eprintln!("Error: Inverse FFT output length mismatch.");
         return Vec::new(); // Ensure impulse response has correct length
    }


    // 4. Calculate Step Response s(t) = cumulative sum of h(t)
    let step_response_raw: Vec<f32> = impulse_response
        .iter()
        .scan(0.0, |cum_sum, x| {
            *cum_sum += *x;
            Some(*cum_sum)
        })
        .collect();

    // 5. Smooth the Step Response using Moving Average
    let step_response_smoothed = smooth_moving_average(&step_response_raw, smoothing_window_size);


    // 6. Normalize the Smoothed Step Response
    //    Limit to a relevant time window (e.g., 500ms) for normalization
    //    Consider normalizing based on the *end* value for better steady-state representation
    let analysis_window_len = ((sample_rate / 2.0) as usize).min(step_response_smoothed.len());
    if analysis_window_len == 0 {
        return Vec::new(); // Not enough data for analysis window
    }

    // Take the analysis window from the *smoothed* data
    let windowed_smoothed_response = &step_response_smoothed[0..analysis_window_len];

    // Normalize based on the average of the *last part* of the window for stability
    let norm_avg_len = (analysis_window_len / 10).max(1); // Average last 10% or at least 1 point
    let norm_start_index = analysis_window_len.saturating_sub(norm_avg_len);
    let norm_slice = &windowed_smoothed_response[norm_start_index..analysis_window_len];
    let avg_end_value : f32 = norm_slice.iter().sum::<f32>() / (norm_slice.len() as f32);


    // Avoid division by zero if average is zero
    let normalization_factor = if avg_end_value.abs() > EPSILON {
        1.0 / avg_end_value
    } else {
        1.0 // Avoid scaling if average is zero, data might be all zeros
    };

    let normalized_smoothed: Vec<f64> = step_response_smoothed // Normalize the *entire* smoothed response
        .iter()
        .map(|x| (*x * normalization_factor) as f64)
        .collect();

    // 7. Combine with Timestamps
    let start_time = times.first().cloned().unwrap_or(0.0);
    times
        .iter()
        .zip(normalized_smoothed.iter())
        .map(|(t, s)| (*t - start_time, *s))
        .collect()
}