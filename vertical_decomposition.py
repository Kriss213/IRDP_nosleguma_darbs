import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse
from contour_estimator import main as process_image
from discrete_planner import get_vertices_2D, plot_polygon
from itertools import chain

def find_closest_point_on_edge(all_vertices, point, same_x=False, return_edge_points=False):
    # same_x = True -> tiks atrasts tuvākais punkts virs vai zem point
    min_distance = float('inf')
    closest_point = None
    x_t, y_t = point

    closest_edge_end_point_1 = None
    closest_edge_end_point_2 = None

    for shape_vertices in all_vertices:
        for i in range(len(shape_vertices)):
            x1, y1 = np.array(shape_vertices[i])
            x2, y2 = np.array(shape_vertices[(i + 1) % len(shape_vertices)])
            tmp_closest_edge_end_point_1 = (x1, y1)
            tmp_closest_edge_end_point_2 = (x2, y2)

            # iegūst taisnes vienādojumu daudzstūra malai
            if x2-x1 == 0:
                x_on_edge = x1 if same_x else x1 - x_t
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
            
            distance = np.sqrt( (x_on_edge - x_t)**2 + (y_on_edge - y_t)**2 )

            if distance < min_distance:
                closest_edge_end_point_1 = tmp_closest_edge_end_point_1
                closest_edge_end_point_2 = tmp_closest_edge_end_point_2
                min_distance = distance
                closest_point = (x_on_edge, y_on_edge)
    
    if return_edge_points:
        return closest_point, closest_edge_end_point_1, closest_edge_end_point_2
    return closest_point

def find_closest_vertex(all_vertices_1d, point):
    distances = np.linalg.norm(np.array(all_vertices_1d) - np.array(point), axis=1)
    closest_vertex_index = np.argmin(distances)
    closest_vertex = all_vertices_1d[closest_vertex_index]
    return closest_vertex

def get_all_edges(vertices, two_dim=False):
    edges = []
    
    for i in range(len(vertices)):
        if two_dim:
            for shape_vertices in vertices[i]:
                p1 = shape_vertices[i]
                p2 = shape_vertices[ (i+1) % len(shape_vertices)]
                edge = [p1, p2]
                edges.append(edge)
        else:
            p1 = vertices[i]
            p2 = vertices[ (i+1) % len(vertices)]
            edge = [p1, p2]
            edges.append(edge)
    return edges

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

def vertical_decomposition(all_vertices, input, output, fig, alpha=None, ax=plt):
    # Saraksts, kurā glabā visu šūnu stūra koordinātes
    cells = []
    # if len(cell[i]) == 2 -> 1D cell
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
    LMB - Pievieno šūnai eksistējošu virsotni
    RMB - Izveido jaunu virsotni, pievieno to šūnai
    ENTER - saglabā jauno šūnu
    c - dzēš iesākto šūnu
    d - dzēš pēdējo saglabāto šūnu
    a - saglabā pašreizējo stāvokli failā
    """
    ax.text(-0.1, 1.15, instructions_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, color="orangered")
    def on_click(event):
        nonlocal available_vertice_scatter
        nonlocal last_point_scatter
        nonlocal cell_lines_plot
        nonlocal all_cell_lines_plot
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

            # LABAIS PELES KLIKŠĶIS, LAI PIEVIENOTU JAUNU VIRSOTNI
            elif event.button == 3:
                if last_point_scatter != None:
                    last_point_scatter.remove()
                # Nomaina X koordināti uz ierpiekšējā punkta X
                clicked_point_POE = (cell_points[0][0], clicked_point[1])

                # Nosaka tuvāko krustpunktu pa vertikālo asi iepriekšējam punktam
                closest_POE = find_closest_point_on_edge(all_vertices, clicked_point_POE, same_x=True)

                cell_points.append(closest_POE)

                # uzzīmē taisni
                line, = ax.plot([cell_points[0][0], closest_POE[0]], [cell_points[0][1], closest_POE[1]], color="lawngreen")
                cell_lines_plot.append(line)
                all_cell_lines_plot.append(line)
                last_point_scatter = ax.scatter(*closest_POE, s=30, color="red")
                
                print(f"Punkts {get_point_str(closest_POE)} pievienots {len(cells)+1}. šūnai")
        
            plt.draw()

    def on_key(event):
        nonlocal available_vertice_scatter
        nonlocal cell_colors_iter
        nonlocal last_point_scatter
        nonlocal accepted_fills
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
                line.remove()
            
            # Zīmēt visas pieejamās virsotnes
            vtbc = np.array(vertices_to_be_checked)
            available_vertice_scatter = ax.scatter(vtbc[:, 0], vtbc[:, 1], color="cyan", s=50)

            plt.draw()
            cell_points.clear()
            print(f"{len(cells)+1}. šūna iztīrīta!")
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
            plt.draw()
        if event.key == "a":
            # Saglabā šūnas, pabeidz vertikālo dekompozīciju
            filename, ext = os.path.splitext(input)
            filename = filename.split(os.sep)[-1]
            path_cells = os.path.join(output, f"{filename}_cells.txt")
            path_img = os.path.join(output, f"{filename}_cells.png")

            with open(path_cells, mode="w") as file:
                file.write(str(cells))
                file.close()

            plt.savefig(path_img)
            print(f"Saglabāti faili \"{path_cells}\" un \"{path_img}\"")
         
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

def main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE):
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

    vertical_decomposition(all_vertices, INPUT_FILE, OUTPUT_PATH, fig=fig, alpha=ALPHA, ax=ax)

    


    ax.invert_yaxis()
    aspect_ratio = X_pixels / Y_pixels
    ax.set_aspect(aspect_ratio)
    plt.show()


if __name__ == "__main__":
    # Komandrindas argumentu apstrāde
    parser = argparse.ArgumentParser(description="2D kartes vertikālā dekompozīcija")

    parser.add_argument("--input", "-i", help="Ceļš uz ieejas failu - attēlu", type=str)
    parser.add_argument("--alpha", "-a", help="Daudzstūru aproksimācijas precizitāte (zemāka -> precīzāk).", default=0.01, type=float)
    parser.add_argument("--output", "-o", help="Izejas ceļa atrašanās vieta.", type=str)
    parser.add_argument("--verbose", "-v", action="store_true", help="Rādīt pilnu programmas izvadi.")
    args = parser.parse_args()

    INPUT_FILE = args.input
    ALPHA = args.alpha
    OUTPUT_PATH = args.output
    VERBOSE_MODE = args.verbose



    if INPUT_FILE == None:
        raise Exception("Norādi ieejas failu!")

    if OUTPUT_PATH == None:
        raise Exception("Norādi izejas failu mapi!")
    
    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Fails \"{INPUT_FILE}\" netika atrasts!")
    
    if ALPHA <= 0:
        raise Exception(f"Vērtībai alpha jābūt lielākai par 0. Tika ievadīts {ALPHA}")

    if OUTPUT_PATH and not os.path.exists(OUTPUT_PATH):
        raise Exception(f"Izejas mape \"{OUTPUT_PATH}\" netika atrasta.")
    
    main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE)