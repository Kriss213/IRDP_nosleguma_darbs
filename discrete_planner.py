# Importē nepieciešamās bibliotēkas
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from contour_estimator import main as process_image
import networkx as nx
from shapely.geometry import LineString, Polygon
from matplotlib.animation import PillowWriter

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

def crosses_polygon(line, polygon):
    line = LineString(line)
    polygon = Polygon(polygon)

    # Pārbauda, vai līnija šķērso kādu šķērsli
    if line.crosses(polygon):
        return True

    return False

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
    #ax.figure(figsize=(8, 8))
    color_mesh_plot = ax.pcolormesh(X, Y, colors, shading='nearest', edgecolors="gray", alpha=0.5, linewidth=.2)
    if plot_center_points:
        scat = ax.scatter(X, Y, color="red", s=5)
        return color_mesh_plot, scat
    return color_mesh_plot

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
    lines_x = []
    lines_y = []
    for polygon_vertices in vertices_2D:
        for i in range(len(polygon_vertices)):
            x_1, y_1 = polygon_vertices[i]
            x_2, y_2 = polygon_vertices[(i+1)%len(polygon_vertices)]
            lines_x.append([x_1, x_2])
            lines_y.append([y_1, y_2])
        ax.plot(lines_x, lines_y, color="black")
        lines_x.clear()
        lines_y.clear()

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

def visualize_path_n_visited_nodes(path, visited_nodes, colors, grid_points_X, grid_points_Y, gridX, gridY, animate=False, ax=plt, batch=25):
    grid_plot = None
    grid_points_X_list = list(grid_points_X)
    grid_points_Y_list = list(grid_points_Y)
    i = 0
    for j, node in enumerate(visited_nodes):
        i+= 1
        try:
            grid_plot.remove()
        except:
            pass
        x, y = node
        col = grid_points_X_list.index(x)
        row = grid_points_Y_list.index(y)

        colors[row][col] = (0.5, 0, 0.5) # apmeklētās virsotnes - violetas
        if animate and (i==batch or j==len(visited_nodes)-1):
            i=0
            grid_plot = plot_grid(gridX, gridY, colors, ax=ax)
            plt.pause(0.01)

    for node in path:
        x, y = node
        col = grid_points_X_list.index(x)
        row = grid_points_Y_list.index(y)
        colors[row][col] = (0,1,0) # ceļa punktus iekrāso zaļus
    if grid_plot:
        grid_plot.remove()
    plot_grid(gridX, gridY, colors, ax=ax)
    
    if not animate:
        plot_grid(gridX, gridY, colors)

# Atrod tuvāko grafa virsotni
def find_closest_vertex(all_vertices_1d, point, return_dist=False):
    distances = np.linalg.norm(np.array(all_vertices_1d) - np.array(point), axis=1)
    closest_vertex_index = np.argmin(distances)
    closest_vertex = all_vertices_1d[closest_vertex_index]
    if return_dist:
        distance = distances[closest_vertex_index]
        return closest_vertex, distance
    return closest_vertex

def generate_RRT_set(length, xlim, ylim, p_target_dir, seed=None):
    if seed != None:
        rng = np.random.default_rng(seed)
        points_x = rng.integers(low=0, high=xlim, size=(length, 1))
        points_y = rng.integers(low=0, high=ylim, size=(length, 1))
        probabilities = rng.random((length, 1))
    
    else:
        points_x = np.random.randint(0, xlim, size=(length, 1), dtype=int )
        points_y = np.random.randint(0, ylim, size=(length, 1), dtype=int )
        probabilities = np.random.rand(length, 1)
        
    # binārs masīvs, kas nosaka, vai jāvirzās mērķa virzienā (1) vai punkta virzienā (0)
    direction_target = (probabilities > (1-p_target_dir)).astype(int)

    return np.hstack( (points_x, points_y, direction_target))

def distance(point1, point2):
    return np.sqrt( (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 )

# Definē RRT algoritmu
def RRT(start, goal, polygons2D, outer_poly_index, xlim, ylim, L=20, R=10, random_point_count=100, seed=None):
    # Ģenerē punktu kopu, kā arī varbūtību, ar kādu dodas mērķa virzienā
    # un ar kādu dodas n-tā punkta virzienā. Nosaka soļa garumu L
    # Nosaka rādisu R ap mērķa punktu, kas, ja tiek sasniegts, tiek uzkatīts,
    # ka sasniegts mērķis.

    graph_RRT = nx.Graph()
    # Pievieno sākuma un mērķa virsotnes
    graph_RRT.add_node( start, pos=start, label="start" )
    graph_RRT.add_node( goal, pos=goal, label="goal" )

    random_points = generate_RRT_set(length=random_point_count, xlim=xlim, ylim=ylim, p_target_dir=0.3, seed=seed)
    x_g, y_g = goal
    origin = None

    # iegūst sarakstu ar visām grafa virsotnēm izņemot mērķa virsotni
    all_nodes_except_goal = [start]

    # Iteratīvi caur random_points
    for row in random_points:
        point = row[0:2]
        direction = row[2]

        new_point = None
        origin = find_closest_vertex(all_nodes_except_goal, point)
        x_o, y_o = origin
        # Virzās mērķa virzienā
        if direction == 1:
            # Aprēķina jaunā punkta koordinātes
            theta = np.arctan2(y_g - y_o, x_g - x_o)
            x_new = x_o + L * np.cos(theta)
            y_new = y_o + L * np.sin(theta)
            new_point = (x_new, y_new)
        
        # Virzās nejaušā virzienā
        elif direction == 0:
            x_t, y_t = point
            theta = np.arctan2(y_t - y_o, x_t - x_o)
            x_new = x_o + L * np.cos(theta)
            y_new = y_o + L * np.sin(theta)
            new_point = (x_new, y_new)

        in_obstacle = False
        # Pārbauda, vai jaunais punkts nepieder kādam šķērslim
        for j, polygon in enumerate(polygons2D):
            if j != outer_poly_index:
                if crosses_polygon([origin, new_point], polygon):
                    in_obstacle = True
                    break
            else:
                if not is_part_of_shape(new_point, polygon):
                    in_obstacle = True
                    break
        if in_obstacle:
            continue

        graph_RRT.add_node(new_point, pos=new_point)
        graph_RRT.add_edge(origin, new_point)
        all_nodes_except_goal.append(new_point)
        found_goal = False
        if distance(goal, new_point) < R:
            found_goal = True
            graph_RRT.nodes[new_point]["label"] = "in_R_of_goal"
            break
    
    return graph_RRT, found_goal

def draw_RRT(edge_list, ax=plt, animate=False, batch=10):
    iter_nr = 0
    for i, edge in enumerate(edge_list):
        iter_nr += 1
        p1 = edge[0]
        p2 = edge[1]

        #uzzīmē līniju
        line_x = [p1[0], p2[0]]
        line_y = [p1[1], p2[1]]
        ax.plot(line_x, line_y, color="black", linewidth=1)
        ax.scatter(*p1, color="black", s=20)
        ax.scatter(*p2, color="black", s=20)
        
        if animate and (iter_nr == batch or i ==len(edge_list)-1 ):
            iter_nr = 0
            plt.pause(0.001)
          
def main(INPUT_FILE, ALGORITHM, ALPHA, VERBOSE_MODE, VISUALIZE, START, GOAL, GRID_SIZE_X, GRID_SIZE_Y, ANIMATE, SEED):
    def verbose_print(message):
        if VERBOSE_MODE:
            print(message)

    # Izsauc contour_estimator.py
    poly_functions, X_pixels, Y_pixels = process_image(INPUT_FILE, ALPHA, OUTPUT_PATH=None, VERBOSE_MODE=False, VISUALIZE=False, OVERWRITE_JSON=False, INTERNAL_CALL=True)
    verbose_print(f"Daudzstūra funkcijas iegūtas.")
    all_vertices, outer_poly_index = get_vertices_2D(poly_functions)
    verbose_print("Daudzstūru stūra punkti noteikti.")
    # uzstāda režģa rindu un kolonu skaitu
    rows, cols = (GRID_SIZE_Y, GRID_SIZE_X)

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
    
    # Izveido grafu
    graph = generate_grid_graph(grid_points_X, grid_points_Y, classification)
    verbose_print(f"Ģenerēts grafs ar {len(graph.nodes)} virsotnēm.")
    
    # Uzstāda sākuma un mērķa punkuts (koordinātes)
    START = tuple(START)
    GOAL = tuple(GOAL)
    start_prim = tuple(int(coord) for coord in min(graph.nodes, key=lambda graph_point: np.sum(np.abs( np.array(graph_point) - np.array(START) ))))
    goal_prim = tuple(int(coord) for coord in min(graph.nodes, key=lambda graph_point: np.sum(np.abs( np.array(graph_point) - np.array(GOAL) ))))
    start_X, start_Y = start_prim
    goal_X, goal_Y = goal_prim

    if START != start_prim:
        print(f"Sākuma punkts pārbītīdts no {START} uz {(start_X, start_Y)}")

    if  GOAL != goal_prim:
        print(f"Mērķa punkts pārbītīdts no {GOAL} uz {(goal_X, goal_Y)}")

    if ALGORITHM == "A*":
        A_star_path, all_visited_nodes = A_star(graph, start=(start_X, start_Y), goal=(goal_X, goal_Y))
        if A_star_path == None:
            verbose_print(f"Ceļš no {START} uz {GOAL} netika atrasts")
            return
        verbose_print(f"Ceļš no {START} uz {GOAL} ar {len(A_star_path)} virsotnēm atrasts.")

    elif ALGORITHM == "RRT":
        RRT_graph, found_goal = RRT(
            start=(start_X, start_Y),
            goal=(goal_X, goal_Y),
            polygons2D=all_vertices,
            outer_poly_index=outer_poly_index,
            xlim=X_pixels,
            ylim=Y_pixels,
            L=100, R=50, random_point_count=500, 
            seed = SEED)
        if found_goal:
            verbose_print(f"RRT atrada mērķi ar {len(list(RRT_graph.nodes))-2} iterācijām")

    if VISUALIZE:
        # Vizualizē režģi, daudzstūrus, ceļu un apmeklētās virsotnes
        fig, ax = plt.subplots(figsize=(10,10))
        plot_grid(grid_X, grid_Y, colors)
        plot_polygon(all_vertices)
        ax.scatter([ start_prim[0], goal_prim[0]], [start_prim[1], goal_prim[1]], color="dodgerblue", s=30)
        ax.annotate("Starts", start_prim, color="red", fontsize=14)
        ax.annotate("Mērķis", goal_prim, color="red", fontsize=14)
        ax.set_title(f"Ceļš no {tuple(start_prim)} uz {tuple(goal_prim)}")
        ax.invert_yaxis()
        aspect_ratio = X_pixels / Y_pixels
        ax.set_aspect(aspect_ratio)

        if ALGORITHM == "A*":
            visualize_path_n_visited_nodes(A_star_path, all_visited_nodes, colors, grid_points_X, grid_points_Y, grid_X, grid_Y, animate=ANIMATE, ax=ax)
        elif ALGORITHM == "RRT":
            rrt_edge_list = list(RRT_graph.edges)
            node_in_R_of_goal = list(RRT_graph.nodes)[-1]
            draw_RRT(rrt_edge_list, animate=ANIMATE, ax=ax, batch=10)

            if found_goal:
                # Atrod īsāko ceļu grafā
                RRT_shorthest_path, _ = A_star(RRT_graph, start_prim, node_in_R_of_goal)
                lines_x = []
                lines_y = []
                for i in range(len(RRT_shorthest_path)-1):
                    p1 = RRT_shorthest_path[i]
                    p2 = RRT_shorthest_path[i+1]
                    lines_x.append([p1[0], p2[0]])
                    lines_y.append([p1[1], p2[1]])
                plt.plot(lines_x, lines_y, color="red", linewidth=1.5)
                   
        plt.show()

if __name__ == "__main__":
    # Komandrindas argumentu apstrāde
    parser = argparse.ArgumentParser(description="Diskrēta ceļa plānošana, izmantojot A* algortimu")

    parser.add_argument("--input", "-i", help="Ceļš uz ieejas failu - attēlu", type=str)
    parser.add_argument("--start", "-s", help="Sākuma punkts (x,y)", nargs="+", type=int)
    parser.add_argument("--goal", "-g", help="Mērķa punkts (x,y)", nargs="+", type=int)
    parser.add_argument("--algorithm", help="Algoritma izvēle (RRT vai A*)", type=str)
    parser.add_argument("--seed", help="RRT algoritmam iegūst atkārtojamus rezultātus", type=int)
    parser.add_argument("--animate", help="Animēt rezultātu", action="store_true")
    parser.add_argument("--grid_cols", help="Režģa kolonu skaits.", type=int, default=50)
    parser.add_argument("--grid_rows", help="Režģa ridnu skaits.", type=int, default=50)
    parser.add_argument("--alpha", "-a", help="Daudzstūru aproksimācijas precizitāte (zemāka -> precīzāk).", default=0.01, type=float)
    parser.add_argument("--verbose", "-v", action="store_true", help="Rādīt pilnu programmas izvadi.")
    parser.add_argument("--draw", "-d", action="store_true", help="Vizualizēt rezultātu.")
    args = parser.parse_args()

    INPUT_FILE = args.input
    ALGORITHM = args.algorithm
    ALPHA = args.alpha
    SEED = args.seed
    VERBOSE_MODE = args.verbose
    VISUALIZE = args.draw
    START = args.start
    GOAL = args.goal
    GRID_SIZE_X = args.grid_cols
    GRID_SIZE_Y = args.grid_rows
    ANIMATE = args.animate


    if INPUT_FILE == None:
        raise Exception("Norādi ieejas failu!")

    if ALGORITHM == None or (ALGORITHM != "A*" and ALGORITHM != "RRT"):
        raise Exception("Nederīga algoritma izvēle. Izvēlies RRT vai A*")

    if START==None or GOAL==None:
        raise Exception("Norādi sākuma un mērķa punktu!")

    if not VISUALIZE and ANIMATE:
         VISUALIZE = True

    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Fails \"{INPUT_FILE}\" netika atrasts!")
    
    if ALPHA <= 0:
        raise Exception(f"Vērtībai alpha jābūt lielākai par 0. Tika ievadīts {ALPHA}")
    
    main(INPUT_FILE, ALGORITHM, ALPHA, VERBOSE_MODE, VISUALIZE, START, GOAL, GRID_SIZE_X, GRID_SIZE_Y, ANIMATE, SEED)