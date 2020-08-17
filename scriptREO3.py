# NOME: ANA CAROLINA FARIA MARTINS
print('a)Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados com a imagem original;')
import cv2
from matplotlib import pyplot as plt
import numpy as np
nome_arquivo = "106.jpeg.jpeg"
img_bgr = cv2.imread(nome_arquivo,1) # Carrega imagem (0 - Binária e Escala de Cinza; 1 - Colorida (BGR))
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB) #transformar em rgb
#Filtros - média
img_fm_1 = cv2.blur(img_rgb,(3,3)) #tamanho do kernel
img_fm_2 = cv2.blur(img_rgb,(5,5))
img_fm_3 = cv2.blur(img_rgb,(7,7))
img_fm_4 = cv2.blur(img_rgb,(9,9))
img_fm_5 = cv2.blur(img_rgb,(11,11))
# Apresentar imagens no matplotlib
plt.figure('Filtros')
plt.subplot(3,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(3,3,2)
plt.imshow(img_fm_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3")

plt.subplot(3,3,3)
plt.imshow(img_fm_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("5x5")

plt.subplot(3,3,4)
plt.imshow(img_fm_3)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("7x7")

plt.subplot(3,3,5)
plt.imshow(img_fm_3)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("9x9")

plt.subplot(3,3,6)
plt.imshow(img_fm_3)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("11x11")
plt.show()
print('b)Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os resultados entre si e com a imagem original.')
# Filtros
# Imagem filtro Média
img_fm_11 = cv2.blur(img_rgb,(9,9))
img_fm_22 = cv2.blur(img_rgb,(11,11))
# Imagem filtro Gaussiano
img_fg_1 = cv2.GaussianBlur(img_rgb,(9,9),0) #media ponderada #quanto vai ser o peso na média ponderada. O determina automaticamente
img_fg_2 = cv2.GaussianBlur(img_rgb,(11,11),0) #Média Ponderada
# Os vizinhos mais próximos tem peso maior. 3L e 3C

# Imagem filtro mediana
img_fmed_1 = cv2.medianBlur(img_rgb,9)
img_fmed_2 = cv2.medianBlur(img_rgb,11)
#gera imagem mais agradável

# Filtro bilateral
# identifica melhor as bordas
img_fb_1 = cv2.bilateralFilter(img_rgb,9,9,3)
img_fb_2 = cv2.bilateralFilter(img_rgb,11,11,5)


########################################################################################################################
# Apresentar imagens no matplotlib
plt.figure('Filtro Média')
plt.subplot(5,5,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(5,5,2)
plt.imshow(img_fm_11)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3 - Filtro média")

plt.subplot(5,5,3)
plt.imshow(img_fm_22)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("20x20 - Filtro média")

plt.subplot(5,5,4)
plt.imshow(img_fg_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3 - Filtro Gaussiana")

plt.subplot(5,5,5)
plt.imshow(img_fg_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("20x20 - Filtro Gaussiana")

plt.subplot(5,5,6)
plt.imshow(img_fmed_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3 - Filtro mediana")

plt.subplot(5,5,7)
plt.imshow(img_fmed_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("20x20 - Filtro mediana")

plt.subplot(5,5,8)
plt.imshow(img_fb_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3 - Filtro binário")

plt.subplot(5,5,9)
plt.imshow(img_fb_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("20x20 - Filtro binário")

plt.show()
print('c)Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o reconhecimento de contornos, identifique e salve os objetos de interesse. Além disso, acesse as bibliotecas Opencv e Scikit-Image, verifique as variáveis que podem ser mensuradas e extrai as informações pertinentes (crie e salve uma tabela com estes dados). Apresente todas as imagens obtidas ao longo deste processo.')
from skimage.measure import label, regionprops
r,g,b = cv2.split(img_rgb)
r = cv2.bilateralFilter(r,15,15,13)
histR = cv2.calcHist([r], [0], None, [256], [0, 256])
# Limiarização - Thresholding
(L, img_limiar_inv) = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Obtendo imagem segmentada
img_segmentada = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_inv)


# Objetos
mascara = img_limiar_inv.copy()
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
n_sementes = len(cnts)
sementes = np.zeros((n_sementes,1))
eixo_menor = np.zeros((n_sementes,1))
eixo_maior = np.zeros((n_sementes,1))
razao = np.zeros((n_sementes,1))
area_1 = np.zeros((n_sementes,1))
#Dados grãos
for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar_inv[y:y+h,x:x+w]
	obj_rgb = img_segmentada[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('s'+str(i+1)+'.png',obj_bgr)
	cv2.imwrite('sb'+str(i+1)+'.png',obj)

	regiao = regionprops(obj) #https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
	print('Semente: ', str(i+1))
	sementes[i,0] = i+1
	print('Dimensão da Imagem: ', np.shape(obj))

	print('Medidas Físicas')
	print('Centroide: ', regiao[0].centroid)
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	eixo_menor[i,0] = regiao[0].minor_axis_length
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	eixo_maior[i,0] = regiao[0].major_axis_length
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
	razao[i,0] = regiao[0].major_axis_length / regiao[0].minor_axis_length
	area = cv2.contourArea(c)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True))
	perimetro = cv2.arcLength(c,True)
	area_1 [i,0] = area
	print('Medidas de Cor')
	min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_r, ' - Posição: ', min_loc_r)
	print('Valor Máximo no R: ', max_val_r, ' - Posição: ', max_loc_r)
	med_val_r = cv2.mean(obj_rgb[:,:,0], mask=obj)
	print('Média no Vermelho: ', med_val_r)

	min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_g, ' - Posição: ', min_loc_g)
	print('Valor Máximo no G: ', max_val_g, ' - Posição: ', max_loc_g)
	med_val_g = cv2.mean(obj_rgb[:,:,1], mask=obj)
	print('Média no Verde: ', med_val_g)

	min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no B: ', min_val_b, ' - Posição: ', min_loc_b)
	print('Valor Máximo no B: ', max_val_b, ' - Posição: ', max_loc_b)
	med_val_b = cv2.mean(obj_rgb[:,:,2], mask=obj)
	print('Média no Azul: ', med_val_b)
	print('-'*50)

print('-'*50)

seg = img_segmentada.copy()
cv2.drawContours(seg,cnts,-1,(0,255,0),2)

# Apresentando as imagens
seg = img_segmentada.copy()
cv2.drawContours(seg,cnts,-1,(0,255,0),2)

plt.figure('Sementes')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('Grãos com contorno')

plt.subplot(1,2,2)
plt.imshow(obj_rgb) #ultima semente executada será imprimida
plt.xticks([])
plt.yticks([])
plt.title('Último Grão')
plt.show()

print('d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse')
img_f = 's82.png'
s82 = cv2.imread(img_f,1)
S82rgb = cv2.cvtColor(s82, cv2.COLOR_BGR2RGB)

img_f2 = 'sb82.png'
sb82 = cv2.imread(img_f2,0)

#Criando histogramas com as máscaras
# Histograma do canal informativo

hist_R1 = cv2.calcHist([S82rgb], [0], sb82, [256],[0,256])
hist_G1 = cv2.calcHist([S82rgb], [1], sb82, [256],[0,256])
hist_B1 = cv2.calcHist([S82rgb], [2], sb82, [256],[0,256])

# Apresentando a imagen do grão colorido
plt.figure('Letra D - Histograma utilizando mascara')
plt.subplot(3, 3, 2)
plt.imshow(s82)  # Imagem do objeto colorida
plt.title('Imagem colorida s82')
plt.xticks([])
plt.yticks([])

# Plotando a imagem do grão em cada canal separadamente
plt.subplot(3, 3, 4)
plt.imshow(S82rgb[:, :, 0], cmap='gray')  # Plotando o canal R
plt.title('Imagem do canal R')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 5)
plt.imshow(S82rgb[:, :, 1], cmap='gray')  # Plotando o canal G
plt.title('Imagem do canal G')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 6)
plt.imshow(S82rgb[:, :, 2], cmap='gray')  # Plotando  canal B
plt.title('Imagem do canal B')
plt.xticks([])
plt.yticks([])

# Plotando o histograma de cada canal separadamente
plt.subplot(3, 3, 7)
plt.plot(hist_R1, color="red")  # Obtendo o histograma do canal R "vermelho"
plt.title("Histograma - R com mascara")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 8)
plt.plot(hist_G1, color="green")  # Obtendo o histograma do canal R "vermelho"
plt.title("Histograma - G com mascara")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 9)
plt.plot(hist_B1, color="blue")
plt.title("Histograma - B com mascara")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

''''
# Histograma de cada grão de arroz


img_seg3 = img_segmentada.copy()

for (i, p) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(p)
    img_bi = img_limiar_inv[y:y + h, x:x + w]
    obj_rgb = img_segmentada[y:y + h, x:x + w]
    Hist = True

    if Hist:
        histR3 = cv2.calcHist([obj_rgb], [0], img_bi, [256], [0, 256])
        histG3 = cv2.calcHist([obj_rgb], [1], img_bi, [256], [0, 256])
        histB3 = cv2.calcHist([obj_rgb], [2], img_bi, [256], [0, 256])

        # Apresentando a imagen do grão
        plt.figure('Histograma utilizando mascara')
        plt.subplot(3, 3, 2)
        plt.imshow(obj_rgb)
        plt.title('Objeto: (i + 1)')

        # Plotando a imagem do grão em cada canal separadamente
        plt.subplot(3, 3, 4)
        plt.imshow(obj_rgb[:, :, 0], cmap='gray')
        plt.title('Objeto: (i + 1)')

        plt.subplot(3, 3, 5)
        plt.imshow(obj_rgb[:, :, 1], cmap='gray')
        plt.title('Objeto: (i + 1)')

        plt.subplot(3, 3, 6)
        plt.imshow(obj_rgb[:, :, 2], cmap='gray')
        plt.title('Objeto: (i + 1)')

        # Plotando o histograma de cada canal separadamente
        plt.subplot(3, 3, 7)
        plt.plot(histR3, color="red")
        plt.title("Histograma - R com mascara")
        plt.xlim([0, 256])
        plt.xlabel("Valores Pixels")
        plt.ylabel("Número de Pixels")

        plt.subplot(3, 3, 8)
        plt.plot(histG3, color="green")
        plt.title("Histograma - G com mascara")
        plt.xlim([0, 256])
        plt.xlabel("Valores Pixels")
        plt.ylabel("Número de Pixels")

        plt.subplot(3, 3, 9)
        plt.plot(histB3, color="blue")
        plt.title("Histograma - B com mascara")
        plt.xlim([0, 256])
        plt.xlabel("Valores Pixels")
        plt.ylabel("Número de Pixels")
        plt.show()
    else:
        pass
'''
print('e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens obtidas neste processo.')

print('Dimensão: ',np.shape(img_rgb))
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])


# Formatação da imagem para uma matriz de dados
pixel_values = img_rgb.reshape((-1, 3))
# Conversão para Decimal
pixel_values = np.float32(pixel_values)

print('Dimensão Matriz: ',pixel_values.shape)
# K-means
# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# Número de Grupos (k)
k = 2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))
# flatten the labels array
labels = labels.flatten()

print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))


# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))
hist = np.concatenate((val_unicos,contagens),axis=1)
print('Histograma')
print(hist)
print('-'*80)
print('Centroides Decimais')
print(centers)
print('-'*80)
# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)

print('Centroides uint8')
print(centers)


# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels]

print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:])


# Reformatar a matriz na imagem de formato original
img_segmentada2 = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_or_01 = original_01.reshape((-1, 3))
matriz_or_01[labels != 0] = [0, 0, 0]
img_final_01 = matriz_or_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels != 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)

# Apresentar Imagem
plt.figure('Imagens')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('ORIGINAL')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada2)
plt.title('ROTULOS')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])
plt.show()

print('f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as imagens obtidas neste processo')
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
r2,g2,b2 = cv2.split(img_rgb)

lf, mascara2 = cv2.threshold(r2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_dist = ndimage.distance_transform_edt(mascara2)
# Calcula a distância euclidiana até o zero


#Matriz booleana com os picos da imagem
max_local = peak_local_max(img_dist, indices=False, min_distance= 23,labels=mascara2)

print('Número de Picos')
print(np.unique(max_local,return_counts=True))

#MARCAÇÃO DOS PICOS
marcadores,n_marcadores = ndimage.label(max_local, structure=np.ones((3,3)))

print('Marcadores')
print(np.unique(marcadores,return_counts=True))
img_ws = watershed(-img_dist, marcadores, mask=mascara2)
print("Número de sementes: ", len(np.unique(img_ws)) - 1)

print('Imagem Segmentada - Watershed')
img_final = np.copy(img_rgb)
img_final[img_ws != 30] = [0,0,0] # Acessando o 30ºgrão de arroz

plt.figure('Segmentação Watershed')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('ORIGINAL')

plt.subplot(2,3,2)
plt.imshow(r2,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('R')

plt.subplot(2,3,3)
plt.imshow(mascara2,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Mascara')

plt.subplot(2,3,4)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Distância')

plt.subplot(2,3,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Arroz')

plt.subplot(2,3,6)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('SELEÇÃO')

plt.show()
print('g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed) e identifique as potencialidades de cada delas.')
plt.figure('Comparando os resultados')
plt.subplot(1,3,1)
plt.imshow(img_segmentada)
plt.xticks([])
plt.yticks([])
plt.title('OTSU')

plt.subplot(1,3,2)
plt.imshow(img_final_01)
plt.title('K-MEANS')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img_ws)
plt.xticks([])
plt.yticks([])
plt.title('Wathershed')
plt.show()
print('-'*50)

