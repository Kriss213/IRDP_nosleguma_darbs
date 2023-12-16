import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse
from contour_estimator import main as process_image
from discrete_planner import get_vertices_2D, plot_polygon, is_part_of_shape, A_star
from itertools import chain
import networkx as nx
import ast

def find_closest_point_on_edge(all_vertices, point, same_x=False, return_edge_points=False):
    # Atrod punktu uz īsākās tuvākās malas
    # same_x = True -> tiks atrasts tuvākais punkts virs vai zem point
    min_distance = float('inf')
    min_edge_length = float('inf')
    closest_point = None
    x_t, y_t = point

    closest_edge_end_point_1 = None
    closest_edge_end_point_2 = None
    distances = []
    for shape_vertices in all_vertices:
        for i in range(len(shape_vertices)):
            x1, y1 = np.array(shape_vertices[i])
            x2, y2 = np.array(shape_vertices[(i + 1) % len(shape_vertices)])
            tmp_closest_edge_end_point_1 = (x1, y1)
            tmp_closest_edge_end_point_2 = (x2, y2)

            # iegūst taisnes vienādojumu daudzstūra malai
            if x2-x1 == 0:
                x_on_edge = x1
                y_on_edge = y_t
            else:
                k1 = (y2-y1) / (x2-x1)
                b1 = y1 - k1*x1

                if same_x:
                    x_on_edge = x_t
                    y_on_edge = k1 * x_t + b1
                else:
                    k2 = -1/k1
                    b2 = y_t - k2*x_t

                    x_on_edge = (b2-b1) / (k1-k2)
                    y_on_edge = k1 * x_on_edge + b1
                
            if not (min(y1,y2) <= y_on_edge <= max(y1, y2)) or not (min(x1,x2) <= x_on_edge <= max(x1, x2)):
                continue
            
            edge_length = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
            distance = np.sqrt( (x_on_edge - x_t)**2 + (y_on_edge - y_t)**2 )

            if distance < min_distance:
                
                min_distance = distance
                closest_point = (x_on_edge, y_on_edge)
                
                if distance in distances:
                    if edge_length < min_edge_length:
                        min_edge_length = edge_length
                        closest_edge_end_point_1 = tmp_closest_edge_end_point_1
                        closest_edge_end_point_2 = tmp_closest_edge_end_point_2
                else:
                    closest_edge_end_point_1 = tmp_closest_edge_end_point_1
                    closest_edge_end_point_2 = tmp_closest_edge_end_point_2
                
                distances.append(distance)
    
    if return_edge_points:
        return closest_point, closest_edge_end_point_1, closest_edge_end_point_2
    return closest_point

def find_closest_vertex(all_vertices_1d, point, return_dist=False):
    distances = np.linalg.norm(np.array(all_vertices_1d) - np.array(point), axis=1)
    closest_vertex_index = np.argmin(distances)
    closest_vertex = all_vertices_1d[closest_vertex_index]
    if return_dist:
        distance = distances[closest_vertex_index]
        return closest_vertex, distance
    return closest_vertex

def get_point_str(point, twoD=False):
    if twoD:
        res = "["
        for p in point: # vairāki punkti
            px_str = f"{p[0]}" if type(p[0]) == int else f"{p[0]:.2f}"
            py_str = f"{p[1]}" if type(p[1]) == int else f"{p[1]:.2f}"
            res += f"({px_str}, {py_str})"
            res += "]" if tuple(p) == tuple(point)[-1] else ","
    else:
        px_str = f"{point[0]}" if type(point[0]) == int else f"{point[0]:.2f}"
        py_str = f"{point[1]}" if type(point[1]) == int else f"{point[1]:.2f}"
        res = f"({px_str}, {py_str})"
      
    return res

def get_side_edge(cell, right=True):
    # Atrod punktus ar vienādu maksimālo x vērtību
    cells_sorted_by_x = sorted(cell, key=lambda point: point[0])
    if right:
        # Pēdējiem diviem punktiem ir jābūt vienādām x koordinātēm
        p1 = cells_sorted_by_x[-1]
        p2 = cells_sorted_by_x[-2]
        if p1[0] != p2[0]:
            # nav taisnas labās malas
            return None
    else:
        # Atrod kreiso malu
        p1 = cells_sorted_by_x[0]
        p2 = cells_sorted_by_x[1]
        if p1[0] != p2[0]:
            # nav taisnas kreisās malas
            return None
    return [p1, p2]  
    
def get_1D_cells(cells_2D): # jeb kopīgās malas starp 2D šūnām
    # 2D saraksts ar 1D šūnām (daudzstūru vertikālajām malām)
    cells_1D = []
    for cell in cells_2D:
        vertical_sides = [get_side_edge(cell, right=False), get_side_edge(cell, right=True)]
        for side in vertical_sides:
            if side == None:
                continue
            cells_1D.append(side)
    return cells_1D

def get_centroid(polygon):
    polygon = np.array(polygon)
    n = len(polygon)
    x_sum = np.sum(polygon[:, 0])
    y_sum = np.sum(polygon[:, 1])
    Cx = x_sum / n
    Cy = y_sum / n
    return (Cx, Cy)

def distance(point1, point2):
    return np.sqrt( (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 )

def vertical_line_contains_sub_lines(line, all_lines, eps=1e-8):#0.1):
    # Pārbauda, vai vertikālai taisnei ir kāda taisne
    # sarakstā all_lines, kas pārklāj
    # daļu no taisnes
    Lx1, Ly1 = line[0]
    Lx2, Ly2 = line[1]
    
    line = np.array(line)
    for test_line in all_lines:
        test_line = np.array(test_line)
        if np.array_equal(line, test_line) or np.array_equal(line, test_line[::-1]):
            continue
        Tx1, Ty1 = test_line[0]
        Tx2, Ty2 = test_line[1]
        
        if Tx1 >= Lx1-eps and Tx1 <= Lx1+eps:
            if max(Ty1, Ty2)-eps <= max(Ly1, Ly2) and min(Ty1, Ty2)+eps >= min(Ly1, Ly2):
                return True
    return False

def draw_loaded_cells(cells, ax=plt, fig=None, alpha=None):
    # Sakārto šūnas augošā secībā pēc vidējās X vērtības
    
    cells_2D = sorted(cells, key=lambda cell: sum( x for x, y in cell) / len(cell))
    cells_1D = get_1D_cells(cells_2D)
   
    centroids_2D = []
    centroids_1D = []
    graph = nx.Graph()
    drawn = False
    for cell in cells_2D:
        cell = np.array(cell)

        # iegūst centroidu
        centroid_2D = get_centroid(cell)
        centroids_2D.append(centroid_2D)
        # attēlo 2D centroīdu
        ax.scatter(*centroid_2D, s=20, color="gold")
        graph.add_node(centroid_2D, pos=centroid_2D)

        # Iegūst šūnas vertikālās sānu malas:
        vertical_sides = [get_side_edge(cell, right=False), get_side_edge(cell, right=True)]
        for side in vertical_sides:
            if side == None:
                continue
            # Zīmē vertikālās malas
            ax.plot([side[0][0], side[1][0]], [side[0][1], side[1][1]], color="lawngreen")

            # Iegūst centroīdus, balstoties uz to, vai vertikālās sānu malas ir sadalītas
            if not vertical_line_contains_sub_lines(side, cells_1D):
                line_centroid = get_centroid(side)
                ax.scatter(*line_centroid, s=20, color="cornflowerblue")
                graph.add_node(line_centroid, pos=line_centroid)
                centroids_1D.append(line_centroid)
               
    eps=1e-8
    for i in range(len(cells_2D)):
        # cells_2D un centroids_2D ir vienāda garuma
        cell = cells_2D[i]
        centroid_2D = centroids_2D[i]
        x_c_2d, y_c_2d = centroid_2D
        for centroid_1D in centroids_1D:
            x_c_1d, y_c_1d = centroid_1D
            # Pieskaita vai atņem eps atkarībā no tā, kurā
            # pusē 2D centrodam atrodas 1D centroids
            x_c_1d_prim = x_c_1d + eps if x_c_2d > x_c_1d else x_c_1d - eps
            # Iegūst jaunu 1D šūnas centroīdu, kas ir iebīdīts 2D šūnā
            c_1d_prim = (x_c_1d_prim, y_c_1d)
            if is_part_of_shape(c_1d_prim, cell):
                # Pievieno grafu, ja iebīdītais punkts pieder kārtējai 2D šūnai
                graph.add_edge(centroid_2D, centroid_1D, length=distance(centroid_1D, centroid_2D))

    title_str = f"Ceļa kartes izveide: α={alpha}" if alpha else f"Ceļa kartes izveide"
    ax.set_title(title_str, fontsize=14)
    instructions_text ="""
    ENTER - animēt grafa izveidi
    """
    info_plot = ax.text(0, 1.0, instructions_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, color="orangered")
    
    start = None
    goal = None
    start_scat = None
    goal_scat = None
    start_line_plot = None
    goal_line_plot = None
    closest_2D_centroid_start = None
    closest_2D_centroid_goal = None
    start_label = None
    goal_label = None
    path_plot = None
    def on_click(event):
        nonlocal start, goal, start_scat, goal_scat
        nonlocal start_line_plot, goal_line_plot
        nonlocal closest_2D_centroid_start, closest_2D_centroid_goal
        nonlocal start_label, goal_label, path_plot
        if not drawn:
            return
        if event.inaxes != None and (event.button == 1 or event.button == 3):
            # LMB - iestata sākuma punktu
            if event.button == 1:
                clicked_point = (event.xdata, event.ydata)
                
                # Pārbauda, vai izvēlētais punkts pieder kādai 2D šūnai
                point_valid = False
                for i, cell in enumerate(cells_2D):
                    if is_part_of_shape(clicked_point, cell):
                        closest_2D_centroid_start = centroids_2D[i]
                        point_valid = True
                        break
                if not point_valid:
                    print("Izvēlies 2D šūnai piederīgu sākuma punktu")
                    return
                # Punkts ir derīgs
                start = clicked_point
                # Dzēš punktu, ja iepriekš jau bija izvēlēts
                if start_scat and start_line_plot and start_label:
                    start_line_plot.remove()
                    start_scat.remove()
                    start_label.remove()
                    start_line_plot = None
                    start_scat = None
                    start_label = None
                    
                if path_plot:
                    for pp in path_plot:
                        pp.remove()
                    path_plot = None

                # atrod tuvāko 2D šunas centroīdu, kurai pievienot punktu
                start_scat = ax.scatter(*start, s=30, color="red")
                start_line_plot, = ax.plot([ closest_2D_centroid_start[0], start[0] ],
                                          [ closest_2D_centroid_start[1], start[1] ],
                                          color="dodgerblue", linewidth=1.5)
                start_label = ax.annotate("Starts", start, color="red", fontsize=14)
        
                plt.draw()
                
            # RMB - iestata mērķa punktu
            elif event.button == 3:
                clicked_point = (event.xdata, event.ydata)
                
                # Pārbauda, vai izvēlētais punkts pieder kādai 2D šūnai
                point_valid = False
                for i, cell in enumerate(cells_2D):
                    if is_part_of_shape(clicked_point, cell):
                        closest_2D_centroid_goal = centroids_2D[i]
                        point_valid = True
                        break
                if not point_valid:
                    print("Izvēlies 2D šūnai piederīgu mērķa punktu")
                    return
                # Punkts ir derīgs
                goal = clicked_point
                # Dzēš punktu, ja iepriekš jau bija izvēlēts
                if goal_scat and goal_line_plot and goal_label:
                    goal_line_plot.remove()
                    goal_scat.remove()
                    goal_label.remove()
                    goal_line_plot = None
                    goal_scat = None
                    goal_label = None
                if path_plot:
                    for pp in path_plot:
                        pp.remove()
                    path_plot = None

                # atrod tuvāko 2D šunas centroīdu, kurai pievienot punktu
                goal_scat = ax.scatter(*goal, s=30, color="lime")
                goal_line_plot, = ax.plot([ closest_2D_centroid_goal[0], goal[0] ],
                                          [ closest_2D_centroid_goal[1], goal[1] ],
                                          color="dodgerblue", linewidth=1.5)
                goal_label = ax.annotate("Mērķis", goal, color="red", fontsize=14)
                plt.draw()
                

            if start != None and goal != None:
                #info_plot.remove()
                # pievieno virsotnes un lokus grafam
                graph.add_node(start, pos=start)
                graph.add_edge(start, closest_2D_centroid_start, length=distance(start, closest_2D_centroid_start))

                graph.add_node(goal, pos=goal)
                graph.add_edge(goal, closest_2D_centroid_goal, length=distance(goal, closest_2D_centroid_goal))
                
                # Izplāno ceļu
                path,_ = A_star(graph, start, goal)
                # Vizualizē ceļu
                lines_x = []
                lines_y = []
                for i in range(len(path)-1):
                    x1, y1 = path[i]
                    x2, y2 = path[(i+1)]
                    lines_x.append([x1, x2])
                    lines_y.append([y1, y2])
                path_plot = ax.plot(lines_x, lines_y, color="tab:red", linewidth=2)
                plt.draw()

                # noņem virsotnes un lokus gadījumam, ja veic pārplānošanu
                graph.remove_edge(start, closest_2D_centroid_start)
                graph.remove_edge(goal, closest_2D_centroid_goal)
                graph.remove_node(start)
                graph.remove_node(goal)
                        



    
    def on_key(event):
        nonlocal drawn
        nonlocal info_plot
        if event.key == "enter":
            if drawn:
                return
            info_plot.remove()
            for edge in list(graph.edges):
                x1, y1 = edge[0]
                x2, y2 = edge[1]

                ax.plot([x1, x2], [y1, y2], color="dodgerblue", linewidth=1.5)
                plt.pause(0.01)
            drawn = True
            instructions_text = """
            LMB - iestata sākuma punktu
            RMB - iestata mēŗka punktu
            """
            info_plot = ax.text(0, 1.01, instructions_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, color="orangered")
            plt.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.draw()

def vertical_decomposition(all_vertices, input, output, fig, alpha=None, ax=plt):
    # Saraksts, kurā glabā visu šūnu stūra koordinātes
    cells = []
    all_vertices_1d = list(chain(*all_vertices))
    cell_points = []
    last_point_scatter = None
    all_vertices_1d_np = np.array(all_vertices_1d)
    available_vertice_scatter = ax.scatter(all_vertices_1d_np[:, 0], all_vertices_1d_np[:, 1], color="cyan", s=50)
    cell_lines_plot = []
    all_cell_lines_plot = []
    cell_colors_iter = iter(list(mcolors.TABLEAU_COLORS.keys()))
    accepted_fills = []
    
    # Kad nospiež uz pirmo punktu, tiek atrasta tuvākā
    # virsotne. Kad nospiež uz otru punktu, x-koordināte
    # tiek nomainīta uz pirmā punkta x koordināti, un
    # tiek atrasta punkta projekcija pa vert. asi uz 
    # tuvākās malas. Punktu pāris tiek saglabāts.
    # Ar peli tiek norādīti atlikušie 1 vai 2 punkti
    # 1 punkts, ja veidojas trijstūris, 2 ja 4stūris
    title_str = f"Vertikālā dekompozīcija: α={alpha}" if alpha else f"Vertikālā dekompozīcija"
    ax.set_title(title_str, fontsize=14)
    instructions_text ="""
    LMB - Pievieno šūnai eksistējošu virsotni               !!Pirmā virsotne jāizvēlas OTRĀ no kreisās puses!!
    RMB - Izveido jaunu virsotni, pievieno to šūnai
    ENTER - saglabā jauno šūnu
    c - dzēš iesākto šūnu
    d - dzēš pēdējo saglabāto šūnu
    a - saglabā pašreizējo stāvokli failā
    q - KAD VISS PABEIGTS!
    """
    vertice_selected = False
    ax.text(-0.1, 1.17, instructions_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, color="orangered")
    def on_click(event):
        nonlocal available_vertice_scatter
        nonlocal last_point_scatter
        nonlocal cell_lines_plot
        nonlocal all_cell_lines_plot
        nonlocal vertice_selected
        if event.inaxes != None and (event.button == 1 or event.button == 3):
            # Nolasa nospiestā punkta koordināted
            clicked_point = (event.xdata, event.ydata)
            cells_1d = list(chain(*cells))
            vertices_to_be_checked = all_vertices_1d + cells_1d + cell_points
            
            # Zīmēt visas pieejamās virsotnes
            available_vertice_scatter.remove()
            vtbc = np.array(vertices_to_be_checked)
            available_vertice_scatter = ax.scatter(vtbc[:, 0], vtbc[:, 1], color="cyan", s=50)

            # KREISAIS PELES KLIKŠĶIS, LAI PIEVIENOTU EKSISTĒJOŠU VIRSOTNI
            if event.button == 1:
                if last_point_scatter != None:
                    last_point_scatter.remove()

                # Nosaka tuvāko stūra vai krustpunku
                closest_vertex = find_closest_vertex(vertices_to_be_checked, clicked_point)
                cell_points.append(closest_vertex)
                
                last_point_scatter = ax.scatter(*closest_vertex, s=30, color="red")
                print(f"Punkts {get_point_str(closest_vertex)} pievienots {len(cells)+1}. šūnai")
                vertice_selected = True
            # LABAIS PELES KLIKŠĶIS, LAI PIEVIENOTU JAUNU VIRSOTNI
            elif event.button == 3:
                if not vertice_selected:
                    return
                if last_point_scatter != None:
                    last_point_scatter.remove()
                # Nomaina X koordināti uz ierpiekšējā punkta X
                clicked_point_POE = (cell_points[0][0], clicked_point[1])

                # Nosaka tuvāko krustpunktu pa vertikālo asi iepriekšējam punktam
                closest_POE = find_closest_point_on_edge(all_vertices, clicked_point_POE, same_x=True)

                # uzzīmē taisni
                line, = ax.plot([cell_points[0][0], closest_POE[0]], [cell_points[0][1], closest_POE[1]], color="lawngreen")
                cell_lines_plot.append(line)
                all_cell_lines_plot.append(line)
                last_point_scatter = ax.scatter(*closest_POE, s=30, color="red")
                
                cell_points.append(closest_POE)
                
                print(f"Punkts {get_point_str(closest_POE)} pievienots {len(cells)+1}. šūnai")
                vertice_selected = False
            plt.draw()

    def on_key(event):
        nonlocal available_vertice_scatter
        nonlocal cell_colors_iter
        nonlocal last_point_scatter
        nonlocal accepted_fills
        nonlocal vertice_selected
        if event.key == "enter":
            if len(cell_points) < 3:
                print(f"Pārāk maz šūnas punktu: {len(cell_points)}")
                return

            # Noņem duplikātus un saglabā cell_points sarakstu
            cell_points_wo_dup = list(dict.fromkeys(cell_points))
            cells.append(cell_points_wo_dup)

            x_fill, y_fill = zip(*cell_points_wo_dup)
            color = next(cell_colors_iter, None)
            if color == None:
                #attiestata iterator
                cell_colors_iter = iter(list(mcolors.TABLEAU_COLORS.keys()))
                color=next(cell_colors_iter)
            fill, = ax.fill(x_fill, y_fill, color=color, alpha=0.5)
            accepted_fills.append(fill)
            plt.draw()

            print(f"Šūna saglabāta:\n\tPunktu skaits: {len(cell_points_wo_dup)}\n\t{get_point_str(cell_points_wo_dup, twoD=True)}")

            cell_points.clear()
            cell_lines_plot.clear()
            vertice_selected = False
        
        if event.key == "c":
            # Atjauno pēdējo stāvokli
            cells_1d = list(chain(*cells))
            vertices_to_be_checked = all_vertices_1d + cells_1d
            
            # Dzēš zīmēto par pēdējo šūnu
            available_vertice_scatter.remove()
            if last_point_scatter != None:
                last_point_scatter.remove()
                last_point_scatter = None
            for line in cell_lines_plot:
                try:
                    line.remove()
                except:
                    pass
            
            # Zīmēt visas pieejamās virsotnes
            vtbc = np.array(vertices_to_be_checked)
            available_vertice_scatter = ax.scatter(vtbc[:, 0], vtbc[:, 1], color="cyan", s=50)

            plt.draw()
            cell_points.clear()
            print(f"{len(cells)+1}. šūna iztīrīta!")
            vertice_selected = False
        if event.key == "d":
            # Dzēš pēdējo šūnu
            if len(cells) == 0:
                print("Nav pievienota neviena šūna")
                return
            cells.remove(cells[-1])
            cells_1d = list(chain(*cells))
            vertices_to_be_checked = all_vertices_1d + cells_1d
            last_fill = accepted_fills.pop(-1)
            last_fill.remove()

            # Dzēš zīmēto par pēdējo šūnu
            available_vertice_scatter.remove()
            if last_point_scatter != None:
                last_point_scatter.remove()
                last_point_scatter = None

            # Dzēš pēdējās šūnas līnijas
            line1 = all_cell_lines_plot.pop(-1)
            line1_data_x = line1.get_xdata()
            if len(all_cell_lines_plot) > 0:
                line2_data_x = all_cell_lines_plot[-1].get_xdata()
                if line1_data_x[0] == line2_data_x[0]:
                    line2 = all_cell_lines_plot.pop(-1)
                    line2.remove()
            line1.remove()
            
            # Zīmēt visas pieejamās virsotnes
            vtbc = np.array(vertices_to_be_checked)
            available_vertice_scatter = ax.scatter(vtbc[:, 0], vtbc[:, 1], color="cyan", s=50)
            print("Dzēsta pēdējā šūna!")
            vertice_selected = False
            plt.draw()
        if event.key == "a":
            # Saglabā šūnas, pabeidz vertikālo dekompozīciju
            filename, ext = os.path.splitext(input)
            filename_split = filename.split(os.sep)[-1]
            if filename_split == filename:
                # nesakrīt slīpsvītras virziens
                sep = "/" if os.sep=="\\" else "\\"
                filename_split = filename.split(sep)[-1]
            path_cells = os.path.join(output, f"{filename_split}_cells.txt")
            path_img = os.path.join(output, f"{filename_split}_cells.png")

            with open(path_cells, mode="w") as file:
                file.write(str(cells))
                file.close()

            plt.savefig(path_img)
            print(f"Saglabāti faili \"{path_cells}\" un \"{path_img}\"")
            vertice_selected = False
         
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    return cells

def main(INPUT_FILE, LOADED_CELLS_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE):
    fig, ax = plt.subplots(figsize=(10,10))

    def verbose_print(message):
        if VERBOSE_MODE:
            print(message)
    
    poly_functions, X_pixels, Y_pixels = process_image(INPUT_FILE, ALPHA, OUTPUT_PATH=None, VERBOSE_MODE=False, VISUALIZE=False, OVERWRITE_JSON=False, INTERNAL_CALL=True)
    verbose_print(f"Daudzstūra funkcijas iegūtas.")
    all_vertices, outer_poly_index = get_vertices_2D(poly_functions)
    verbose_print("Daudzstūru stūra punkti noteikti.")
    plot_polygon(all_vertices, ax=ax)
    verbose_print("Daudzstūris vizualizēts")
    
    # Ja tiek ielādēts šūnu fails, no tā iegūst masīvu
    cells = None
    stage_2 = False
    if LOADED_CELLS_FILE != None:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        f = open(LOADED_CELLS_FILE, "r")
        loaded_cells_str = f.read()
        cells = ast.literal_eval(loaded_cells_str)
        draw_loaded_cells(cells, alpha=ALPHA, fig=fig, ax=ax)
        stage_2 = True
    
    else:
        cells = vertical_decomposition(all_vertices, INPUT_FILE, OUTPUT_PATH, fig=fig, alpha=ALPHA, ax=ax)
    

    ax.invert_yaxis()
    aspect_ratio = X_pixels / Y_pixels
    ax.set_aspect(aspect_ratio)
    plt.show()
    
    if len(cells) != 0 and not stage_2:
        # cells satur jaunākās izmaiņas
        fig1, ax1 = plt.subplots(figsize=(10,10))
        plot_polygon(all_vertices, ax=ax1)
        verbose_print("Daudzstūris vizualizēts")
        draw_loaded_cells(cells, alpha=ALPHA, fig=fig1, ax=ax1)
        verbose_print("Nolasītās šūnas vizualizētas")

        ax1.invert_yaxis()
        aspect_ratio = X_pixels / Y_pixels
        ax1.set_aspect(aspect_ratio)
        plt.show()
        

if __name__ == "__main__":
    # Komandrindas argumentu apstrāde
    parser = argparse.ArgumentParser(description="2D kartes vertikālā dekompozīcija")

    parser.add_argument("--input", "-i", help="Ceļš uz ieejas failu - attēlu", type=str)
    parser.add_argument("--load_data", "-l", help="Ielādē šūnu failu", type=str)
    parser.add_argument("--alpha", "-a", help="Daudzstūru aproksimācijas precizitāte (zemāka -> precīzāk).", default=0.01, type=float)
    parser.add_argument("--output", "-o", help="Izejas ceļa atrašanās vieta.", type=str)
    parser.add_argument("--verbose", "-v", action="store_true", help="Rādīt pilnu programmas izvadi.")
    args = parser.parse_args()

    INPUT_FILE = args.input
    LOADED_CELLS_FILE = args.load_data
    ALPHA = args.alpha
    OUTPUT_PATH = args.output
    VERBOSE_MODE = args.verbose


    if LOADED_CELLS_FILE != None and ALPHA == 0.01:
        print(f"BRĪDINĀJUMS - tiek izmantota noklusējuma alpha vērtība un ielādētas šunas.")
        
    if INPUT_FILE == None:
        raise Exception("Norādi ieejas failu!")
    
    if LOADED_CELLS_FILE and not os.path.exists(LOADED_CELLS_FILE):
        raise Exception(f"Norādītais fails \"{LOADED_CELLS_FILE}\" netika atrasts.")

    if OUTPUT_PATH == None and LOADED_CELLS_FILE == None:
        raise Exception("Norādi izejas failu mapi!")
    
    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Fails \"{INPUT_FILE}\" netika atrasts!")
    
    if ALPHA <= 0:
        raise Exception(f"Vērtībai alpha jābūt lielākai par 0. Tika ievadīts {ALPHA}")

    if OUTPUT_PATH and not os.path.exists(OUTPUT_PATH):
        raise Exception(f"Izejas mape \"{OUTPUT_PATH}\" netika atrasta.")
    
    main(INPUT_FILE, LOADED_CELLS_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE)
