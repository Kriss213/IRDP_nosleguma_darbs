# Importē nepieciešamās bibliotēkas
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from contour_estimator import main as process_image

def is_part_of_shape(point, shape):
    x, y = point
    intersection_count = 0

    for i in range(len(shape)):
        x1, y1 = shape[i]
        x2, y2 = shape[(i + 1) % len(shape)]
        if y == y1 or y == y2:
            y += 1e-8
        if y1 <= y <= y2 or y2 <= y <= y1:
            if x1 == x2 and x <= x1:
                intersection_count += 1
            elif x1 != x2:
                intersect_x = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x <= intersect_x:
                    intersection_count += 1

    return intersection_count % 2 != 0

def round_up_to_multiplier(number, N):
    if N == 0:
        return number
    rem = number % N
    return number - rem + N

def generate_grid(x_max, y_max, row_count, col_count):
    end_x = round_up_to_multiplier(x_max, col_count-1)
    end_y = round_up_to_multiplier(y_max, row_count-1)

    grid_points_x = np.concatenate( [np.array([0]), np.linspace(end_x/(col_count-1), end_x, col_count-1)] )
    grid_points_y = np.concatenate( [np.array([0]), np.linspace(end_y/(row_count-1), end_y, row_count-1)] )

    X, Y = np.meshgrid(grid_points_x, grid_points_y)
    return X, Y, grid_points_x, grid_points_y

def plot_grid(X, Y, colors, ax=plt, plot_center_points=False):
    ax.figure(figsize=(8, 8))
    ax.pcolormesh(X, Y, colors, shading='nearest', edgecolors="gray", alpha=0.5, linewidth=.2)
    if plot_center_points: ax.scatter(X, Y, color="red", s=5)

def get_grid_colors(gridX, gridY, vertices_2D, outer_contour_id):
    def check_point(X, Y):
        # pārbaude vienam punktam
        for i, polygon_vertices in enumerate(vertices_2D):

            is_part_of_obstacle = is_part_of_shape((X, Y), polygon_vertices)
            if not is_part_of_obstacle and i == outer_contour_id:
                # Punkti, kas atrodas ārpus kartes ir melni
                return 0
            elif is_part_of_obstacle and i != outer_contour_id:
                # Punkti, kas atrodas šķēršļos ir sarkani
                return 1
        # Punkti, kas nav šķērslī ir balti
        return 2

    
    check_point_vectorized = np.vectorize(check_point)
    classification = check_point_vectorized(gridX, gridY)
    
    rows, cols = gridX.shape
    colors = np.zeros((rows, cols, 3))
    colors[classification == 0] = (0, 0, 0) # Melns
    colors[classification == 1] = (1, 0, 0) # Sarkans
    colors[classification == 2] = (1, 1, 1) # Balts

    return colors, classification

def get_vertices_2D(poly_functions):
    # Saraksts, kurā glabā visas virsotnes
    all_vertices = []
    outer_poly_index = None
    for polygon_index, polygon_info in enumerate(poly_functions.values()):
        polygon_vertices = []
        for key in polygon_info.keys():
            if key == "is_outer":
                if polygon_info[key]:
                    outer_poly_index = polygon_index
                continue
            
            # iegūst y funkciju no dict
            x_start = polygon_info[key]["x_start"]
            y = eval(polygon_info[key]["y"]) # funkcija
            P_start = (x_start, y(x_start))
            polygon_vertices.append(P_start)
        all_vertices.append(polygon_vertices)
    return all_vertices, outer_poly_index

def plot_polygon(vertices_2D, ax=plt):
    for polygon_vertices in vertices_2D:
        for i in range(len(polygon_vertices)):
            x_1, y_1 = polygon_vertices[i]
            x_2, y_2 = polygon_vertices[(i+1)%len(polygon_vertices)]

            ax.plot([x_1, x_2], [y_1, y_2], color="black")


def main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE, VISUALIZE):
    # Izsauc contour_estimator.py
    poly_functions, X_pixels, Y_pixels = process_image(INPUT_FILE, ALPHA, OUTPUT_PATH=None, VERBOSE_MODE=False, VISUALIZE=False, OVERWRITE_JSON=False, INTERNAL_CALL=True)
    all_vertices, outer_poly_index = get_vertices_2D(poly_functions)

    # uzstāda režģa rindu un kolonu skaitu
    rows, cols = (100, 80)

    # ģenerē režģa punktus
    # grid_points_X/Y ir 1D saraksti ar attiecīgās koord. ass vērtībām
    # grid_X/Y ir visa režģa punktu koordinātas
    grid_X, grid_Y, grid_points_X, grid_points_Y = generate_grid(X_pixels, Y_pixels, rows, cols)

    # iegūst režģa šūnu krāsas atkarībā no klasifikācijas
    # Klasifikācija katram punktam
    # satur informāciju, vai tas ir:
    # 0 - ārpus kartes,
    # 1 - šķērsļī
    # 2 - brīvs
    colors, classification = get_grid_colors(grid_X, grid_Y, all_vertices, outer_poly_index)

    # Piemērs, kā iegūt klasifikāciju kādam no režģa punktiem
    rdm_x_idx = np.random.randint(0, len(grid_points_X))
    rdm_y_idx = np.random.randint(0, len(grid_points_Y))
    flag = classification[rdm_y_idx][rdm_x_idx]
    print(f"Point ({grid_points_X[rdm_x_idx]}, {grid_points_Y[rdm_y_idx]}) chosen and it has a flag of: {flag}")
    

    if VISUALIZE:
        # vizualizē režģi un daudzstūrus
        plot_grid(grid_X, grid_Y, colors)
        plot_polygon(all_vertices)

        plt.gca().invert_yaxis()
        aspect_ratio = X_pixels / Y_pixels
        plt.gca().set_aspect(aspect_ratio)
        plt.show()

    #TODO
    # izveidot grafu no grid_points_X, grid_points_Y (moš var for loopā pa taisno pievienot arī edges idk)
    # grafu (un networkX) vajag, lai viegli varētu dabūt kaimiņus
    # Implementēt A*
    # Vizualizēt ne tika gala ceļu, bet arī visus apskatītos ceļus
    # Sīkumi:
    #   verbose_print
    #   komandrindas parametri
    #   ceļa saglabāšana failā (punktu (pikseļu koord.) secība)
    #   clean up

if __name__ == "__main__":
    # Komandrindas argumentu apstrāde
    parser = argparse.ArgumentParser(description="Diskrēta ceļa plānošana, izmantojot A* algortimu")

    parser.add_argument("--input", "-i", help="Ceļš uz ieejas failu - attēlu", type=str)
    parser.add_argument("--alpha", "-a", help="Daudzstūru aproksimācijas precizitāte (zemāka -> precīzāk).", default=0.01, type=float)
    parser.add_argument("--output", "-o", help="Izejas ceļa atrašanās vieta.", type=str)
    parser.add_argument("--verbose", "-v", action="store_true", help="Rādīt pilnu programmas izvadi.")
    parser.add_argument("--draw", "-d", action="store_true", help="Vizualizēt rezultātu.")
    args = parser.parse_args()

    INPUT_FILE = args.input
    ALPHA = args.alpha
    OUTPUT_PATH = args.output
    VERBOSE_MODE = args.verbose
    VISUALIZE = args.draw

    if INPUT_FILE == None:
        raise Exception("Norādi ieejas failu!")
    
    if not VISUALIZE and not OUTPUT_PATH:
        VISUALIZE = True

    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Fails \"{INPUT_FILE}\" netika atrasts!")
    
    if ALPHA <= 0:
        raise Exception(f"Vērtībai alpha jābūt lielākai par 0. Tika ievadīts {ALPHA}")

    if OUTPUT_PATH and not os.path.exists(OUTPUT_PATH):
        raise Exception(f"Izejas mape \"{OUTPUT_PATH}\" netika atrasta.")
    
    main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE, VISUALIZE)