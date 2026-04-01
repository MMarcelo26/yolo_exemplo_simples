# TODO
- [ ] documentação da Captura do OpenCV 
- [ ] Pesquisar Documentação YOLO para identificação de objetos
  - [ ] Conexão além da rede local
- [ ] Procurar apps semelhantes ao IPCam para conexão por rede independente
- [ ] ler Objective Video Quality Assessment Documento Whatsapp

link repo coco yolo database
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml


## Resumo dos Parâmetros-Chave 

### Funções para Identificação de formas

- **`cv2.GaussianBlur(imagem, kernel_size, desvio_padrao)`**
  - **Descrição:** Suaviza a imagem para reduzir ruídos.
  - **`kernel_size`:** Tamanho da matriz de desfoque (ímpar, ex: `(5, 5)`, `(9, 9)`). Quanto maior, mais suave fica a imagem.
  - **`desvio_padrao`:** Geralmente `0` para cálculo automático.

- **`cv2.HoughCircles(imagem, método, dp, minDist, param1, param2, minRadius, maxRadius)`**
  - **Descrição:** Detecta círculos usando a Transformada de Hough.
  - **`método`:** Geralmente `cv2.HOUGH_GRADIENT`.
  - **`dp`:** Razão inversa do acumulador (resolução da detecção). `1` = mesma resolução da imagem.
  - **`minDist`:** Distância mínima entre centros de círculos detectados (ex: `50`).
  - **`param1`:** Limiar superior para detecção de bordas (ex: `50`).
  - **`param2`:** Limiar para detecção de círculos (menor = mais círculos detectados, ex: `30` a `100`).
  - **`minRadius` / `maxRadius`:** Limites para o raio dos círculos detectados (ex: `10` a `1000`).

- **`cv2.adaptiveThreshold(imagem, valor_máximo, método, tipo, blockSize, C)`**
  - **Descrição:** Binariza a imagem com limiar adaptativo.
  - **`método`:** Geralmente `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`.
  - **`tipo`:** Geralmente `cv2.THRESH_BINARY_INV` (inverte os valores).
  - **`blockSize`:** Tamanho da vizinhança para cálculo do limiar (ímpar, ex: `11`).
  - **`C`:** Constante subtraída da média (ex: `2` a `5`).

- **`cv2.findContours(imagem, modo_recuperação, método_aproximação)`**
  - **Descrição:** Encontra contornos na imagem.
  - **`modo_recuperação`:** Geralmente `cv2.RETR_EXTERNAL` (recupera apenas contornos externos).
  - **`método_aproximação`:** Geralmente `cv2.CHAIN_APPROX_SIMPLE` (comprime contornos).

- **`cv2.contourArea(contorno)`**
  - **Descrição:** Calcula a área do contorno em pixels. Usado para filtrar contornos pequenos (ruídos).
  - **Valor típico:** `500` a `2000` (ajuste conforme sua necessidade).

- **`cv2.approxPolyDP(contorno, epsilon, fechado)`**
  - **Descrição:** Simplifica contornos para polígonos.
  - **`epsilon`:** Precisão da aproximação (ex: `0.02` a `0.05` do perímetro). Quanto maior, mais simplificado fica o contorno.
  - **`fechado`:** Geralmente `True` para fechar o polígono.

- **`cv2.arcLength(contorno, fechado)`**
  - **Descrição:** Calcula o perímetro do contorno, usado para definir `epsilon` no `approxPolyDP`.

- **`cv2.boundingRect(contorno)`**
  - **Descrição:** Retorna o retângulo delimitador de um contorno, usado para verificar proporções (ex: quadrado vs. retângulo).


---
- ### Funções Específicas para Detecção de Cor Laranja


- **`cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`**
  - **Descrição:** Converte a imagem do espaço de cor **BGR** (padrão do OpenCV) para **HSV** (Hue, Saturation, Value).
  - **Por que usar HSV?**
    - O espaço de cor **HSV** é mais intuitivo para detecção de cores, pois separa a **matiz (Hue)** da intensidade e saturação.
    - Facilita a definição de faixas de cor (ex: laranja) com base apenas no **Hue**, independentemente da iluminação.



- **`cv2.inRange(hsv, laranja_claro, laranja_escuro)`**
  - **Descrição:** Cria uma **máscara binária** que isola pixels dentro de uma faixa de cor específica no espaço HSV.
  - **Parâmetros:**
    - `hsv`: Imagem no espaço de cor HSV.
    - `laranja_claro`: Limite inferior da cor laranja (`[H, S, V]`).
    - `laranja_escuro`: Limite superior da cor laranja (`[H, S, V]`).
  - **Resultado:**
    - Retorna uma imagem binária (preto e branco) onde:
      - **Branco**: Pixels dentro da faixa de cor laranja.
      - **Preto**: Pixels fora da faixa de cor laranja.
  - **Exemplo de valores para laranja:**
    ```python
    laranja_claro = np.array([5, 100, 100])   # H=5, S=100, V=100
    laranja_escuro = np.array([22, 255, 255]) # H=22, S=255, V=255
    ```


- **`cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]`**
  - **Descrição:** Encontra **contornos** em uma imagem binária (máscara).
  - **Parâmetros:**
    - `mascara`: Imagem binária gerada pelo `cv2.inRange`.
    - `cv2.RETR_TREE`: Modo de recuperação que cria uma **hierarquia completa** de contornos (inclui contornos internos e externos).
    - `cv2.CHAIN_APPROX_SIMPLE`: Comprime contornos horizontais, verticais e diagonais, economizando memória.
  - **`[-2]`:**
    - Garante compatibilidade entre versões do OpenCV.
    - Em versões antigas, `cv2.findContours` retorna 2 valores: `(contornos, hierarquia)`.
    - Em versões novas, retorna 3 valores: `(contornos, hierarquia, outros)`.
    - `[-2]` sempre retorna os **contornos**, independentemente da versão.


- **`cv2.contourArea(contorno)`**
  - **Descrição:** Calcula a **área** de um contorno (em pixels).
  - **Uso no código:**
    - Filtra contornos muito pequenos (ruídos) usando um limite (ex: `area > 1000`).
    - Ajuste o valor do limite conforme o tamanho do objeto e a distância da câmera.


- **`cv2.boundingRect(contorno)`**
  - **Descrição:** Retorna as coordenadas do **retângulo delimitador** de um contorno.
  - **Resultado:**
    - Retorna uma tupla `(x, y, largura, altura)`, onde:
      - `(x, y)`: Coordenadas do canto superior esquerdo do retângulo.
      - `largura` e `altura`: Dimensões do retângulo.
  - **Uso no código:**
    - Usado para desenhar um retângulo ao redor do objeto laranja detectado.



- **`cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)`**
  - **Descrição:** Desenha um **retângulo** na imagem.
  - **Parâmetros:**
    - `frame`: Imagem onde o retângulo será desenhado.
    - `(x, y)`: Canto superior esquerdo do retângulo.
    - `(x + w, y + h)`: Canto inferior direito do retângulo.
    - `(0, 165, 255)`: Cor do retângulo em **BGR** (azul=0, verde=165, vermelho=255 → laranja).
    - `2`: Espessura da linha do retângulo (em pixels).


- **`cv2.putText(frame, "Cartao Laranja", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)`**
  - **Descrição:** Adiciona **texto** à imagem.
  - **Parâmetros:**
    - `frame`: Imagem onde o texto será desenhado.
    - `"Cartao Laranja"`: Texto a ser exibido.
    - `(x, y - 10)`: Posição do texto (10 pixels acima do retângulo).
    - `cv2.FONT_HERSHEY_SIMPLEX`: Tipo da fonte.
    - `0.6`: Escala do texto.
    - `(0, 165, 255)`: Cor do texto em **BGR** (laranja).
    - `2`: Espessura do texto.

