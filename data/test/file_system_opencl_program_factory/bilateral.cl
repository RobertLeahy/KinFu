
// setup how we will sample the raster (clamp to nearest edge and nearest)
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

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
kernel void bilateral_filter(read_only image2d_t src, write_only image2d_t dest, const float sigma_s_inv_sq, const float sigma_r_inv_sq) {

    int x = (int)get_global_id(0); // x-dimension of work item???
    int y = (int)get_global_id(1); // y-dimension of work item???

    if (x >= get_image_width(src) || y >= get_image_height(src)) return; //OOB

    float summation = 0;
    float W_p = 0;

    uint4 R_k_u = read_imageui(src, sampler, (int2)(x,y));

    // loop over all src pixels (q = [i,j])
    // TODO: use a windowed approach instead of the whole raster...?
    for (int i = 0; i < get_image_width(src); i++) {
        for (int j = 0; j < get_image_height(src); j++) {

            uint4 R_k_q = read_imageui(src, sampler, (int2)(i,j));

            float range2 = (R_k_u.x - R_k_q.x) * (R_k_u.x - R_k_q.x);
            float spatial2 = (x-i)*(x-i) + (y-j)*(y-j);

            float weight = exp(-(spatial2 * sigma_s_inv_sq + range2 * sigma_r_inv_sq));

            summation += weight * R_k_q.x;
            W_p += weight;
        }

    }

    uint res = convert_uint(summation/W_p);
    write_imageui(dest, (int2)(x,y), (uint4)(res, 0.0, 0.0, 1.0));


}