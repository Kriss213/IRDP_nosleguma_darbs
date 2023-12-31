# Importē nepieciešamās bibliotēkas
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import json

def main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE, VISUALIZE, OVERWRITE_JSON, INTERNAL_CALL):
    def verbose_print(message):
        if VERBOSE_MODE:
            print(message)

    # Nolasa attēlu un to pārveido par grayscale.
    # Visi pikseļi, kuru vērtības ir lielākas par 128 tiek uzskatīti par melniem,
    # pārējie - par baltiem.
    image = cv2.imread(INPUT_FILE)
    if np.all(image == None):
        raise Exception(f"Nolasītais fails \"{INPUT_FILE}\" nav derīgs attēls!")
    filename, ext = os.path.splitext(INPUT_FILE)
    filename = filename.split(os.sep)[-1]
    json_file_name = f"{filename}_polygon_functions.json"
    JSON_EXISTS = os.path.exists(json_file_name)
    verbose_print(f"Fails {INPUT_FILE} veiksmīgi nolasīts")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # Pārliecinās, ka visas līnijas ir biezākas par 1 px,
    # lai atrastās kontūras varētu izšķirt hierarhiski
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel=kernel, iterations=1)
    verbose_print("Attēls pārveidots par pelēktoņu attēlu.")

    # No attēla iegūst kontūras (daudzstūrus)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.array(hierarchy)
    if len(contours) == 0:
        raise Exception(f"Neizdevās attēlā noteikt kontūras")
    verbose_print(f"Attēlā pirms filtrēšanas atrastas {len(contours)} kontūras.")

    # Hierarhijas struktūra
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # [Next, Previous, First_Child, Parent]

    # Atlasa visas kontūras, kurām ir "iepriekšējā" kontūra tajā pašā hierarhijas līmenī
    cond1 = (hierarchy[0,:,0] != -1)
    # Atlasa visas kontūras, kurām ir "nākamajā" kontūra tajā pašā hierarhijas līmenī
    cond2 = (hierarchy[0,:,1] != -1)
    cond_12 = np.logical_or(cond1, cond2)

    # Atlasa kontūras, kurai nav "iepriekšējā", nav "nākamā" un nav "vecāka"
    # citiem vārdiem sakot - ārējo kontūru
    cond3_0 = np.logical_and(hierarchy[0,:,0] == -1, hierarchy[0,:,1] == -1)
    cond3 = np.logical_and(cond3_0, hierarchy[0,:,3] == -1) 

    indices = np.logical_or(cond_12, cond3)
    old_contours_len = len(contours)
    contours = [c for index, c in enumerate(contours) if indices[index]]
    verbose_print(f"Izfiltrētas {old_contours_len - len(contours)} liekas kontūras")
    

    # Izveido attēlus, kuros attēlot rezultātu
    result_only_contours = np.ones_like(image) * 255
    result_overlap = image.copy()

    # šeit tiks glabāti funkcijas, kas apraksta daudzstūrus 
    poly_functions = {}

    # corner_points saraksta id, kas atbilst ārējai kontūrai
    outer_contour_id = np.argmax(indices)

    # Analizē atrastās kontūras
    for i, contour in enumerate(contours):      
        # aprēķina precizitātes parametru
        closed_shape=True
        epsilon = ALPHA * cv2.arcLength(contour, closed_shape)

        # Aproksimē attiecīgo kontūru līdz daudzstūriem 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_flat = [np.squeeze(arr, axis=0) for arr in approx]
        
        if len(approx) == 0:
            raise Exception("Neizdevās aproksimēt poligonus no kontūrām")
        verbose_print(f"Aproksimēta kontūra ar {len(approx)} punktiem")

        # Uzzīmē kontūras uz rezultējošiem attēliem
        cv2.drawContours(result_only_contours, [approx], -1, (0,0,0), thickness=1)
        cv2.drawContours(result_overlap, [approx], -1, (0,255,0), thickness=5)

        poly_functions[i] = {}
        if (not JSON_EXISTS or OVERWRITE_JSON) or INTERNAL_CALL:
            if i == outer_contour_id:
                poly_functions[i]["is_outer"] = True
            else:
                poly_functions[i]["is_outer"] = False

        if i == outer_contour_id:
            verbose_print("==== Ārējā kontūra ====")
        else:
            verbose_print(f"==== {i}. šķērslis ====")

        # Saglabā stūra punktus, uzzīmē tos uz attēla
        last_corner_point = np.array([-1, -1])
        for j, point in enumerate(approx_flat):
            corner_point =  np.array(point) #[0]

            # lai izvairītos no horizontālas vai vertikālas taisnes
            if np.any(corner_point == last_corner_point):
                corner_point = corner_point + 1

            #corners.append(corner_point)
            last_corner_point = corner_point

            if not JSON_EXISTS or OVERWRITE_JSON or INTERNAL_CALL:
                # Iegūst taisnes vienādojumu
                x1, y1 = corner_point
                x2, y2 = approx_flat[(j+1) % len(approx_flat)]
                
                if x2-x1 == 0 or y2-y1 == 0:
                    x1 += 1
                    y1 += 1
                
                A = (y2-y1) / (x2-x1)
                B = -1
                C = y1 - A * x1
            
                poly_functions[i][j] = {
                    "y": f"lambda x: int({A}*x + {C})",
                    "y_can":f"labmda x, y: {A}*x + {B}*y + {C} == 0",
                    "x_start": x1.tolist(),
                    "x_end": x2.tolist()
                }
                y_str = f"{A:.3f}*x {B:.3f}*y + {C:.3f} = 0" if C>=0 else f"{A:.3f}*x {B:.3f}*y {C:.3f} = 0"
                verbose_print(f"{j+1}.\t{y_str}")

            cv2.circle(result_only_contours, corner_point, 7, (0, 0, 255), -1)

    verbose_print("Izpilde veiksmīga")

    # Izvada un saglabā rezutlātus
    if VISUALIZE and not INTERNAL_CALL:
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Oriģinālā karte")

        axs[1].imshow(cv2.cvtColor(result_only_contours, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Tikai kontūras")

        axs[2].imshow(cv2.cvtColor(result_overlap, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Pārklājums")

        fig.suptitle(f"Approksimēti daudzstūri pie α = {ALPHA}")
        plt.show()

    if OUTPUT_PATH and not INTERNAL_CALL:
        contour_img_path = os.path.join(OUTPUT_PATH,filename+"_polygons"+ext)
        try:
            cv2.imwrite(contour_img_path, cv2.cvtColor(result_only_contours, cv2.COLOR_BGR2RGB))
        except:
            print(f"Neizdevās saglabāt {contour_img_path}")

        overlap_img_path = os.path.join(OUTPUT_PATH,filename+"_overlap"+ext)
        try:
            cv2.imwrite(overlap_img_path, cv2.cvtColor(result_overlap, cv2.COLOR_BGR2RGB))
        except:
            print(f"Neizdevās saglabāt {overlap_img_path}")

    if not JSON_EXISTS or OVERWRITE_JSON and not INTERNAL_CALL:
        json_str = json.dumps(poly_functions, indent=4)
        with open(json_file_name, "w") as json_file:
            json_file.write(json_str)
        
        print(f"Fails \"{json_file_name}\" saglabāts veiksmīgi!")

    verbose_print("Programmas izpilde pabeigta")
    
    X_pixels = len(image[0])
    Y_pixels = len(image)
    return poly_functions, X_pixels, Y_pixels

if __name__ == "__main__":
    # Komandrindas argumentu apstrāde
    parser = argparse.ArgumentParser(description="2D vides daudzstūru aproksimācija")
    parser.add_argument("--input", "-i", help="Ceļš uz ieejas failu", type=str)
    parser.add_argument("--alpha", "-a", help="Daudzstūru aproksimācijas precizitāte (zemāka -> precīzāk).", default=0.01, type=float)
    parser.add_argument("--output", "-o", help="Izejas attēlu atrašanās vieta.", type=str)
    parser.add_argument("--overwrite_json", "-j", action="store_true", help="Pārrakstīt JSON failu. in")
    parser.add_argument("--verbose", "-v", action="store_true", help="Rādīt pilnu programmas izvadi.")
    parser.add_argument("--draw", "-d", action="store_true", help="Vizualizēt rezultātu.")
    parser.add_argument("--internal_call", action="store_true", help="Paredzēts, kad izsauc no citas programmas.")
    args = parser.parse_args()

    INPUT_FILE = args.input
    ALPHA = args.alpha
    OUTPUT_PATH = args.output
    VERBOSE_MODE = args.verbose
    VISUALIZE = args.draw
    OVERWRITE_JSON = args.overwrite_json
    INTERNAL_CALL = args.internal_call

    if INPUT_FILE == None:
        raise Exception("Norādi ieejas failu!")
    
    if not VISUALIZE and not OUTPUT_PATH and not INTERNAL_CALL:
        VISUALIZE = True

    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Fails \"{INPUT_FILE}\" netika atrasts!")

    if ALPHA <= 0:
        raise Exception(f"Vērtībai alpha jābūt lielākai par 0. Tika ievadīts {ALPHA}")

    if OUTPUT_PATH and not os.path.exists(OUTPUT_PATH):
        raise Exception(f"Izejas mape \"{OUTPUT_PATH}\" netika atrasta.")
    
    main(INPUT_FILE, ALPHA, OUTPUT_PATH, VERBOSE_MODE, VISUALIZE, OVERWRITE_JSON, INTERNAL_CALL)