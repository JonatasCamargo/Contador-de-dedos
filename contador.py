import cv2
import mediapipe as mp

#integrar com webcam
video = cv2.VideoCapture(0)

#variavel responsável pelas consfigurações do mediapipe
hand = mp.solutions.hands

#variavel para dar o parametro de número máixmo de mãos que o sisteam irá reconhecer
Hand = hand.Hands(max_num_hands=1) 

#variavel responsável por desenhar as ligações entre os pontos na mão
mpDraw = mp.solutions.drawing_utils

#iteração feita para começar a usar a webcam
while True:
    check,img = video.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #conversão de imagem para RGB
    results = Hand.process(imgRGB) #processar a imagem RGB com mediapipe
    handsPoints = results.multi_hand_landmarks #estraindo as coordenadas, na onde estão os pontos da mão
    h, w, _ = img.shape
    pontos = []
    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points,hand.HAND_CONNECTIONS)
            #podemos enumerar esses pontos da seguinte forma
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                #cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                pontos.append((cx,cy))

            #este arraw vai conter os pontos superiores de cada dedo, exceto o dedão
            dedos = [8,12,16,20]
            contador = 0
            if pontos:
                if pontos[4][0] < pontos[2][0]:
                    contador += 1
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1

        #Inserir informação dentro da imagem
        cv2.rectangle(img, (80, 10), (200,110), (255, 0, 0), -1)
        cv2.putText(img,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)        

    cv2.imshow("Imagem",img) #nome da webcam
    cv2.waitKey(1) #delay de 1seg para abrir a webcam

