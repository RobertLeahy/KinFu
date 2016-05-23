
// setup how we will sample the raster (clamp to nearest edge and nearest)
// const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/**
 *  Bilateral filter on 2d image of uint16_t
 *
 *  src - source image
 *  dest - dest image
 *  sigma_s_inv_sq - inverse square of std dev of gaussian for spatial kernel (default = 4.5 px)
 *  sigma_r_inv_sq - inverse square of std dev of gaussian for range kernel   (default = 30 mm)
 *
 *
 */
kernel void bilateral_filter(__global float* src, __global float* dest, 
    const float sigma_s_inv_sq, const float sigma_r_inv_sq,
    const unsigned int width, const unsigned int height, const unsigned int window_width, const unsigned int window_height) {

    unsigned int x = (unsigned int)get_global_id(0); // x-dimension of work item???
    unsigned int y = (unsigned int)get_global_id(1); // y-dimension of work item???

    unsigned int start_x = 0;
    if (x > window_width) start_x = x - window_width;
    unsigned int start_y = 0;
    if (y > window_height) start_y = y - window_height;
    unsigned int end_x = x + window_width;
    if (end_x > width) end_x = width;
    unsigned int end_y = y + window_height;
    if (end_y > height) end_y = height;

    float summation = 0;
    float W_p = 0;

    float R_k_u = src[y * width + x];
    
    // loop over all src pixels (q = [i,j])
    // TODO: use a windowed approach instead of the whole raster...?
    for (unsigned int i = start_x; i < end_x; i++) {
        for (unsigned int j = start_y; j < end_y; j++) {

            float R_k_q = src[j * width + i];

            float range2 = (R_k_u - R_k_q) * (R_k_u - R_k_q);
            float spatial2 = (x-i)*(x-i) + (y-j)*(y-j);

            float weight = exp(-(spatial2 * sigma_s_inv_sq + range2 * sigma_r_inv_sq));

            summation += weight * R_k_q;
            W_p += weight;
        }

    }

    dest[y * width + x] = summation/W_p;

}