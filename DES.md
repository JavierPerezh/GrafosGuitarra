# G08: Generador Algorítmico de Tablaturas Óptimas

## 1. Visión General del Proyecto

Este proyecto consiste en el desarrollo de una herramienta computacional que traduce automáticamente partituras musicales estructuradas a tablaturas de guitarra ergonómicamente optimizadas. Para lograrlo, el sistema abstrae la redundancia del diapasón y la partitura como un problema de optimización en **Teoría de Grafos**.

El modelo matemático central transforma la secuencia musical en un **Grafo Dirigido Acíclico (DAG)**[cite: 1] por capas temporales. Dado que el movimiento humano tiene inercia y memoria muscular, el grafo se transforma posteriormente en un **Grafo de Estados Compuestos (Grafo Dual)**[cite: 1, 2]. Finalmente, utilizando el algoritmo heurístico de búsqueda multiobjetivo **NAMOA***[cite: 1, 3], el sistema calcula el conjunto de tablaturas Pareto-óptimas (minimizando el esfuerzo físico y optimizando la fluidez y el timbre acústico)[cite: 1], ofreciéndolas al usuario como alternativas.

Este repositorio documenta la arquitectura técnica, las estructuras de datos y el flujo de los algoritmos para facilitar la comprensión y colaboración de cualquier desarrollador.

---

## 2. Arquitectura del Sistema

La herramienta opera secuencialmente a través de cuatro módulos principales[cite: 3].

### Módulo 1: Ingestión y Parseo de Datos
*   **Función:** Consume la secuencia de eventos musicales discretos.
*   **Entrada:** Formatos musicales estructurados como archivos MIDI, MusicXML[cite: 1, 3] (o a futuro, datos pre-procesados de motores OMR de terceros)[cite: 1].
*   **Salida:** Una secuencia lineal o temporal de conjuntos de frecuencias (notas o acordes) en instantes discretos de tiempo ($t_1, t_2, \dots, t_m$)[cite: 2].

### Módulo 2: Generador del DAG Base y Filtrado Anatómico
*   **Función:** Construye el espacio de estados inicial mapeando frecuencias a posiciones físicas.
*   **Estructura Base:** Instancia un Grafo Dirigido Acíclico (DAG) por capas[cite: 1, 2]. Cada capa $L_i$ representa el instante de tiempo $t_i$[cite: 2].
*   **Lógica de Nodos:** Cada nodo en la capa $L_i$ es una *tupla* que representa un estado completo y válido de la mano[cite: 2]. Si es un acorde de $k$ notas, el nodo contiene las posiciones específicas $(s, f, d)$ (cuerda, traste, dedo) y la posición base de la mano ($p_{\text{mano}}$)[cite: 2].
*   **Reglas de Viabilidad (Poda):** Antes de crear un nodo, el módulo aplica restricciones intra-capa:
    *   **Exclusión de cuerda:** $\forall x \neq y, \, s_x \neq s_y$ (no pueden coexistir dos notas en la misma cuerda en trastes diferentes)[cite: 2].
    *   **Apertura límite ($D_{\text{máx}}$):** La diferencia entre el traste más agudo y el más grave ($\max(f) - \min(f)$) no puede superar el límite anatómico humano elástico (usualmente 4 a 5 trastes)[cite: 2].

### Módulo 3: Constructor del Grafo Dual de Estados Compuestos
*   **Función:** Otorga memoria inercial al sistema, convirtiendo el DAG estático de primer orden en un grafo capaz de evaluar trayectorias de segundo orden.
*   **Transformación de Nodos:** Cada arista $e = (u, v)$ del DAG original (que conecta la capa $t_{i-1}$ con $t_i$) se convierte en un nuevo vértice compuesto $\hat{v}_{u,v}$ en el grafo dual $\hat{G}$[cite: 2].
*   **Conectividad:** Se establece una arista dirigida entre $\hat{v}_{u,v}$ y $\hat{v}_{x,y}$ si y solo si $v = x$, conectando la trayectoria $t_{i-1} \to t_i$ con $t_i \to t_{i+1}$[cite: 2].
*   **Ponderación Multiobjetivo:** Cada arista en el grafo dual recibe un vector de costos tridimensional $\mathbf{W} = [W_{\text{bio}}, W_{\text{timbre}}, W_{\text{fluidez}}]^\top$[cite: 3].
    *   **$W_{\text{bio}}$:** Penaliza aceleraciones longitudinales bruscas de la mano ($\alpha \cdot |p_{\text{mano}}(t_i) - p_{\text{mano}}(t_{i-1})|^\gamma$) y desplazamientos transversales de cuerdas[cite: 3].
    *   **$W_{\text{timbre}}$:** Penaliza saltos entre cuerdas heterogéneas que rompen la continuidad tímbrica de líneas melódicas[cite: 3].
    *   **$W_{\text{fluidez}}$:** Bonifica transiciones que posibilitan ligados articulados (\textit{hammer-ons, pull-offs})[cite: 3].

### Módulo 4: Motor de Búsqueda y Optimización de Pareto
*   **Función:** Encuentra y extrae los caminos óptimos (Frontera de Pareto).
*   **Algoritmo Principal:** Implementación de **NAMOA*** (\textit{A New Approach to Multi-Objective A*})[cite: 1, 3].
*   **Mecánica Base:** Utiliza listas de etiquetas no dominadas ($G_{\text{open}}$ y $G_{\text{closed}}$) por nodo, ordenadas lexicográficamente utilizando una heurística vectorial admisible $\mathbf{h}(n)$[cite: 3].
*   **Control de Crecimiento:** Ejecuta subrutinas de *Early Dominance Checking* para podar expansiones (rutas generadas) que sean dominadas en el sentido de Pareto por otra ruta en el mismo estado[cite: 3].
*   **Salida:** Devuelve un conjunto reconstruido de rutas no dominadas desde $t_1$ hasta $t_m$[cite: 3], que corresponden a las opciones de tablaturas ergonómicas presentadas al usuario[cite: 1].

---

## 3. Estructuras de Datos y Modelo de Clases (Propuesta para Código)

Al escribir código para este proyecto (en C++, Python, o Java), deberás estructurar los datos basándote en la matemática del artículo formal.

### A. La Estructura de un Evento Musical
*   Un `MusicalEvent` representa un instante de tiempo $t_i$. Contiene una lista de frecuencias musicales requeridas simultáneamente.

### B. La Estructura del Nodo Original (Postura Base)
*   **Clase `PositionTuple`:** Almacena la `cuerda` ($s$), `traste` ($f$) y el `dedo` ($d$) asignado.
*   **Clase `BaseNode`:** Almacena una lista de objetos `PositionTuple` (que deben pasar las validaciones de exclusión de cuerda y límite $D_{\text{máx}}$)[cite: 2], la posición central de la mano ($p_{\text{mano}}$)[cite: 2] y el ID de la capa temporal ($t_i$).

### C. La Estructura del Grafo de Estados Compuestos (El modelo activo)
*   **Clase `CompoundNode` ($\hat{v}$):**
    *   Posee dos referencias a `BaseNode`: el `origen` ($t_{i-1}$) y el `destino` ($t_i$)[cite: 2].
    *   Contiene un estado interno de memoria de costos acumulados y un puntero de retroceso al predecesor para la reconstrucción algorítmica.
*   **Estructura de Aristas:**
    *   Una arista contiene el vector multidimensional de pesos $\mathbf{W} = [W_{\text{bio}}, W_{\text{timbre}}, W_{\text{fluidez}}]^\top$ calculados a través de funciones estáticas de evaluación de esfuerzo, acústica y articulación[cite: 3].

### D. La Cola de NAMOA*
*   Requiere la implementación de un Priority Queue (\texttt{OPEN})[cite: 3] donde la función de comparación no es un escalar univariante, sino una evaluación lexicográfica o comparador basado en dominancia estricta de Pareto[cite: 3] aplicada a $\mathbf{f}(l) = \mathbf{g}(l) + \mathbf{h}(n)$[cite: 3].

---

## 4. Guía para Nuevos Desarrolladores

Si te unes al equipo de desarrollo de G08, considera las siguientes pautas:

1.  **Respeta las Restricciones Anatómicas:** El éxito del grafo yace en la poda temprana. Si un desarrollador omite los axiomas de viabilidad intra-capa (como el $\Delta \text{Traste} \le D_{\text{máx}}$)[cite: 2], el espacio de búsqueda combinatorio ($\mathcal{O}(n^m)$) estallará[cite: 2] y agotará la RAM disponible.
2.  **No implementes OMR (Optical Music Recognition):** Según el documento de diseño y recomendaciones académicas, procesar imágenes de partituras (pdf/jpg) es un campo de investigación de *Computer Vision* de altísima complejidad. Reutiliza librerías externas para convertir la imagen a MusicXML (si es estrictamente necesario), o ingresa la partitura mediante formatos ya estructurados (MIDI/JSON). **Concéntrate exclusivamente en el motor algorítmico y el procesamiento sobre los grafos compuestos.**
3.  **Mantén la Vectorización Multiobjetivo:** No colapses la función de costo biomecánico, tímbrico y de fluidez en un único escalar mediante promedios o sumas ponderadas. NAMOA* y el marco teórico dependen estrictamente de la dominancia de Pareto con vectores para encontrar regiones no convexas[cite: 1, 3].
4.  **Optimiza la Función Heurística $\mathbf{h}(n)$:** NAMOA* requiere una heurística admisible (optimista). Debes colaborar en afinar la fase de búsqueda "hacia atrás" que precalcula este vector heurístico[cite: 3] para garantizar la máxima velocidad de procesamiento en partituras densas.
