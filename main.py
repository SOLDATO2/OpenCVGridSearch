import cv2
import numpy as np
import time

# Precisamos do scikit-image para a função de SSIM
from skimage.metrics import structural_similarity as ssim

def ssim_score(imgA, imgB):
    """
    Calcula o SSIM (Structural Similarity Index) entre duas imagens coloridas.
    Faz a média do SSIM de cada canal (B, G, R).
    """
    if imgA.shape != imgB.shape:
        # Ajustar tamanho se necessário
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    bA, gA, rA = cv2.split(imgA)
    bB, gB, rB = cv2.split(imgB)

    # Converter para float (necessário para o SSIM)
    bA = bA.astype(np.float32)
    gA = gA.astype(np.float32)
    rA = rA.astype(np.float32)
    bB = bB.astype(np.float32)
    gB = gB.astype(np.float32)
    rB = rB.astype(np.float32)

    ssim_b = ssim(bA, bB, data_range=(bA.max() - bA.min()))
    ssim_g = ssim(gA, gB, data_range=(gA.max() - gA.min()))
    ssim_r = ssim(rA, rB, data_range=(rA.max() - rA.min()))
    ssim_mean = (ssim_b + ssim_g + ssim_r) / 3.0
    
    return ssim_mean

def warp_image(image, flow):
    """
    Aplica o fluxo 'flow' (shape: HxWx2) na imagem de entrada (HxWx3).
    O fluxo indica o deslocamento [dx, dy] em cada pixel.
    """
    h, w = flow.shape[:2]
    
    # Criar o mapa de coordenadas (remap)
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            flow_map[y, x, 0] = x + flow[y, x, 0]
            flow_map[y, x, 1] = y + flow[y, x, 1]

    # Aplicar remap com interpolação bilinear
    warped = cv2.remap(
        image, 
        flow_map, 
        None, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )
    return warped

def find_best_params(im1_path, im2_path, im3_path, output_path, time_budget=0.015):
    """
    - Carrega im1, im2 e im3.
    - Faz uma grid search sobre uma lista grande de parâmetros do Farneback.
    - Para cada combinação, mede o tempo de execução do fluxo 1->3.
    - Se estiver dentro do time_budget, faz warp im1->im3 e compara com im2 via SSIM.
    - Salva o warp de maior SSIM (<= time_budget).
    - Agora, exibindo a cada iteração quantas combinações restam.
    """
    # 1. Ler as imagens
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    im3 = cv2.imread(im3_path)
    
    if im1 is None or im2 is None or im3 is None:
        print("Erro ao carregar as imagens. Verifique os caminhos.")
        return

    # Converter para cinza somente para o fluxo
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # Grid de parâmetros
    # -----------------------------------------------------
    pyr_scales = [0.2, 0.3, 0.5, 0.7, 0.8]  
    levels_list = [1, 2, 3, 5, 8, 10]       
    winsizes    = [5, 9, 15, 21, 25]
    iterations = [1, 2]       
    poly_ns     = [5, 7, 9]                
    poly_sigmas = [1.1, 1.2, 1.3, 1.5, 1.8]

    best_ssim_value = -float("inf")
    best_params = None
    best_warp = None

    # Cálculo do total de combinações
    total_tests = (len(pyr_scales) * len(levels_list) *
                   len(winsizes) * len(iterations)  * len(poly_ns)    *
                   len(poly_sigmas))

    print(f"Total de combinações: {total_tests}")

    tested_count = 0
    feasible_count = 0

    for pyr_scale in pyr_scales:
        for levels in levels_list:
            for winsize in winsizes:
                for iteration in iterations:
                    for poly_n in poly_ns:
                        for poly_sigma in poly_sigmas:
                            tested_count += 1
                            
                            # Exibir quantas restam
                            remaining = total_tests - tested_count
                            print(f"Testando combinação {tested_count}/{total_tests} | Restantes: {remaining}")

                            # 2. Calcular Optical Flow Farneback
                            start_time = time.time()
                            flow = cv2.calcOpticalFlowFarneback(
                                gray1,
                                gray3,
                                None,
                                pyr_scale=pyr_scale,
                                levels=levels,
                                winsize=winsize,
                                iterations=iteration,  # fixado
                                poly_n=poly_n,
                                poly_sigma=poly_sigma,
                                flags=0
                            )
                            end_time = time.time()
                            flow_time = end_time - start_time

                            # 3. Verificar tempo
                            if flow_time <= time_budget:
                                feasible_count += 1
                                # 4. Warpar im1 -> im3
                                warped_im1 = warp_image(im1, flow)

                                # 5. Calcular SSIM com im2
                                current_ssim = ssim_score(warped_im1, im2)

                                # 6. Atualiza melhor
                                if current_ssim > best_ssim_value:
                                    best_ssim_value = current_ssim
                                    best_params = {
                                        "pyr_scale": pyr_scale,
                                        "levels": levels,
                                        "winsize": winsize,
                                        "iterations": 1,
                                        "poly_n": poly_n,
                                        "poly_sigma": poly_sigma,
                                        "time": flow_time
                                    }
                                    best_warp = warped_im1.copy()
                            else:
                                # Fora do limite de tempo
                                pass

    print(f"Total de combinações testadas: {tested_count}")
    print(f"Combinações dentro do tempo limite: {feasible_count}")

    if best_warp is not None:
        cv2.imwrite(output_path, best_warp)
        print("Melhor warp salvo em:", output_path)
        print("Melhores parâmetros encontrados (<= 0.015s):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"SSIM obtido: {best_ssim_value:.4f} (quanto mais próximo de 1, melhor)")
    else:
        print(f"Nenhuma combinação rodou abaixo de {time_budget:.4f} s.")

if __name__ == "__main__":
    find_best_params(
        im1_path="im1.png",
        im2_path="im2.png",
        im3_path="im3.png",
        output_path="best_warp_ssim.png",
        time_budget=0.015  # 15 ms
    )
