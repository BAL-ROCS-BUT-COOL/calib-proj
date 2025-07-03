import random



def generate_grid(
    projector_resolution, 
    margin, 
    grid_size,
    n_grids
):
    n_shifts_per_direction = int(n_grids**(1/2))
    w_proj, h_proj = projector_resolution
    n_h, n_w = grid_size
    s_h = (h_proj - 2 * margin) / (n_h + (n_shifts_per_direction-1)/(n_shifts_per_direction))
    s_w = (w_proj - 2 * margin) / (n_w + (n_shifts_per_direction-1)/(n_shifts_per_direction))
    s = min(s_h, s_w)

    # print(f"s_h = {s_h}, s_w = {s_w}, s = {s}")
    h_c = s * (n_h + (n_shifts_per_direction-1)/(n_shifts_per_direction))
    w_c = s * (n_w + (n_shifts_per_direction-1)/(n_shifts_per_direction))
    # print(f"h_c = {h_c}, w_c = {w_c}")
    margin_h = (h_proj - h_c) / 2
    margin_w = (w_proj - w_c) / 2
    # print(f"margin_h = {margin_h}, margin_w = {margin_w}")

    grids_coords = {}
    shifts = [i * s / n_shifts_per_direction for i in range(n_shifts_per_direction)]
    shift_id = 0
    for shift_h in shifts:
        for shift_w in shifts:
            grid_coords = {}
            id = 0
            for i in range(n_h):
                for j in range(n_w):
                    x = margin_w + j * s + shift_w + s/2
                    y = margin_h + i * s + shift_h + s/2
                    grid_coords[id] = (x, y)
                    id += 1
            grids_coords[shift_id] = grid_coords
            shift_id += 1
    

    return grids_coords