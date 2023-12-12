# Importē nepieciešamās bibliotēkas
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from contour_estimator import main as process_image
import networkx as nx

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

def generate_grid_graph(grid_points_X, grid_points_Y, classification):
    step_X = grid_points_X[1]
    step_Y = grid_points_Y[1]

    # Inicializē tukšu grafu
    graph = nx.Graph()

    # Pievieno grafa virsotnes
    for col, x in enumerate(grid_points_X):
        for row, y in enumerate(grid_points_Y):
            node_id = (x,y)
            if classification[row][col] == 2:
                graph.add_node(node_id, pos=node_id)
    
    # Pievieno grafa lokus
    for col, x in enumerate(grid_points_X):
        for row, y in enumerate(grid_points_Y):
            if classification[row][col] != 2:
                continue
            if (x + step_X, y) in graph.nodes:
                graph.add_edge((x, y), (x + step_X, y))
            if (x, y + step_Y) in graph.nodes:
                graph.add_edge((x, y), (x, y + step_Y))

    return graph

def draw_graph(graph):
    pos = nx.get_node_attributes(graph,"pos")
    nx.draw_networkx(graph, pos, with_labels=False, node_size=2)
    plt.gca().invert_yaxis()
    plt.show()

# Ceļa konstruēšanas metode
def reconstruct_path(cameFrom, current):
    total_path = [current]

    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0, current)
    return total_path

# Definē A* algoritmu
def A_star(graph, start, goal):
    # Heiristiskas funkcija - tiešais attālums
    def h(n):
        goal_x, goal_y = graph.nodes(data=True)[goal]["pos"]
        cur_x, cur_y = graph.nodes(data=True)[n]["pos"]
        return np.sqrt( (goal_x-cur_x)**2 + (goal_y-cur_y)**2 )
    def get_dist(node1, node2):
        x1, y1 = node1
        x2, y2 = node2
        return np.sqrt( (x2-x1)**2 + (y2-y1)**2 )

    open = [start]
    cameFrom = {}

    # saraksts, kur glabā apskatītās virsotnes
    visited_nodes = []

    f_score = {}
    g_score = {}
    for node in graph.nodes:
        f_score[node] = float("inf")
        g_score[node] = float("inf")
    
    g_score[start] = 0
    f_score[start] = h(start)

    while len(open) != 0:
        current = min(open, key=lambda x, f_score=f_score: f_score[x] )
        visited_nodes.append(current)

        if current == goal:
            return reconstruct_path(cameFrom, current), visited_nodes
        
        open.remove(current)

        for neighbor in list(graph.neighbors(current)):
            dist = get_dist(current, neighbor)#graph[current][neighbor]["length"]
            tentative_g_score = g_score[current] + dist

            if tentative_g_score < g_score[neighbor]:
                cameFrom[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h(neighbor)

                if neighbor not in open:
                    open.append(neighbor)
    
    return None

def visualize_path_n_visited_nodes(path, visited_nodes, colors, grid_points_X, grid_points_Y, gridX, gridY, ax=plt):
    start = path[0]
    goal = path[-1]

    for col, x in enumerate(grid_points_X):
        for row, y in enumerate(grid_points_Y):
            point = (x, y)
            if point in path:
                colors[row][col] = (0,1,0) # ceļa punktus iekrāso zaļus
            elif point in visited_nodes:
                colors[row][col] = (0.5, 0, 0.5) # apmeklētās virsotnes - violetas
    
    # Iezīmē sākuma un mērķa punktus
    plot_grid(gridX, gridY, colors)
    ax.scatter([ start[0], goal[0]], [start[1], goal[1]], color="dodgerblue", s=20)
    ax.annotate("Starts", start, color="navy", fontsize=14)
    ax.annotate("Mērķis", goal, color="navy", fontsize=14)



def main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE, VISUALIZE):
    def verbose_print(message):
        if VERBOSE_MODE:
            print(message)

    # Izsauc contour_estimator.py
    poly_functions, X_pixels, Y_pixels = process_image(INPUT_FILE, ALPHA, OUTPUT_PATH=None, VERBOSE_MODE=False, VISUALIZE=False, OVERWRITE_JSON=False, INTERNAL_CALL=True)
    verbose_print(f"Daudzstūra funkcijas iegūtas.")
    all_vertices, outer_poly_index = get_vertices_2D(poly_functions)
    verbose_print("Daudzstūru stūra punkti noteikti.")
    # uzstāda režģa rindu un kolonu skaitu
    rows, cols = (100, 80)

    # ģenerē režģa punktus
    # grid_points_X/Y ir 1D saraksti ar attiecīgās koord. ass vērtībām
    # grid_X/Y ir visa režģa punktu koordinātas
    grid_X, grid_Y, grid_points_X, grid_points_Y = generate_grid(X_pixels, Y_pixels, rows, cols)
    verbose_print(f"Ģenerēts režģis ar izmēru {rows}x{cols}")

    # iegūst režģa šūnu krāsas atkarībā no klasifikācijas
    # Klasifikācija katram punktam
    # satur informāciju, vai tas ir:
    # 0 - ārpus kartes,
    # 1 - šķērsļī
    # 2 - brīvs
    colors, classification = get_grid_colors(grid_X, grid_Y, all_vertices, outer_poly_index)
    verbose_print("Režģa punkti klasifcēti")

    # Piemērs, kā iegūt klasifikāciju kādam no režģa punktiem
    # rdm_x_idx = np.random.randint(0, len(grid_points_X))
    # rdm_y_idx = np.random.randint(0, len(grid_points_Y))
    # flag = classification[rdm_y_idx][rdm_x_idx]
    # print(f"Point ({grid_points_X[rdm_x_idx]}, {grid_points_Y[rdm_y_idx]}) chosen and it has a flag of: {flag}")
    
    # Izveido grafu
    graph = generate_grid_graph(grid_points_X, grid_points_Y, classification)
    verbose_print(f"Ģenerēts grafs ar {len(graph.nodes)} virsotnēm.")
    
    # Uzstāda sākuma un mērķa punkuts (koordinātes)
    start_X, start_Y = (300, 300)
    goal_X, goal_Y = (1200, 1200)

    # Nav garantēts, ka tieši tādi punkti ir grafā, tāpēc atrod tuvākos:
    start_X = min(grid_points_X, key=lambda x: abs(x - start_X))
    goal_X = min(grid_points_X, key=lambda x: abs(x - goal_X))
    start_Y = min(grid_points_Y, key=lambda y: abs(y - start_Y))
    goal_Y = min(grid_points_Y, key=lambda y: abs(y - goal_Y))

    A_star_path, all_visited_nodes = A_star(graph, start=(start_X, start_Y), goal=(goal_X, goal_Y))
    verbose_print(f"Ceļš ar {len(A_star_path)} virsotnēm atrasts.")

    if VISUALIZE:
        # Vizualizē režģi, daudzstūrus, ceļu un apmeklētās virsotnes
        visualize_path_n_visited_nodes(A_star_path, all_visited_nodes, colors, grid_points_X, grid_points_Y, grid_X, grid_Y)
        plot_polygon(all_vertices)
    
        plt.gca().invert_yaxis()
        aspect_ratio = X_pixels / Y_pixels
        plt.gca().set_aspect(aspect_ratio)
        plt.show()

    #TODO
    # Sīkumi:
    #   verbose_print
    #   komandrindas parametri
    #   ceļa saglabāšana failā (punktu (pikseļu koord.) secība)
    #   clean up
    # pievienot armugenu start point, end point, un grid size

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