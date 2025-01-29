import random



def generate_grid(projector_resolution, 
                  margin, 
                  grid_size,
                  n_shifts_per_direction):
    
    w_proj, h_proj = projector_resolution
    n_h, n_w = grid_size
    s_h = (h_proj - 2 * margin) / (n_h + n_shifts_per_direction/(n_shifts_per_direction-1))
    s_w = (w_proj - 2 * margin) / (n_w + n_shifts_per_direction/(n_shifts_per_direction-1))
    s = min(s_h, s_w)

    print(f"s_h = {s_h}, s_w = {s_w}, s = {s}")
    h_c = s * (n_h + n_shifts_per_direction/(n_shifts_per_direction-1))
    w_c = s * (n_w + n_shifts_per_direction/(n_shifts_per_direction-1))
    print(f"h_c = {h_c}, w_c = {w_c}")
    margin_h = (h_proj - h_c) / 2
    margin_w = (w_proj - w_c) / 2
    print(f"margin_h = {margin_h}, margin_w = {margin_w}")

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
         

    # import matplotlib.pyplot as plt

    # colors = {}
    # for i in range(100):
    #     colors[i] = (random.random(), random.random(), random.random())


    # for shift_id, grid_coords in grids_coords.items():
    #     alpha = shift_id /len(grids_coords)
    #     color = (1 - alpha, 0.5, alpha)
    #     x_coords = [coord[0]+s/2 for coord in grid_coords.values()]
    #     y_coords = [coord[1]+s/2 for coord in grid_coords.values()]
    #     # colors = plt.cm.viridis(shift_id / (n_shifts_per_direction**2 - 1))
    #     # plt.scatter(x_coords, y_coords, label=f'Shift {shift_id}', s=1, color=colors)
    #     plt.scatter(x_coords, y_coords, label=f'Shift {shift_id}', s = 1, color=colors[shift_id])


    # plt.xlim(0, w_proj)
    # plt.ylim(0, h_proj)
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.title('Grid Points')
    # plt.legend()
    # plt.show()


    return grids_coords