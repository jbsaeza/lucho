# Ejercicio 1

def invertir_string(string):
    string_inv = ""
    for i in range(len(string)):
        string_inv += string[-i-1]
    return string_inv

# Ejercicio 2

def maximo_producto_absoluto(lista):
    max1 = 0
    max2 = 0
    for i in range(len(lista)):
        if max1 < abs(lista[i]):
            max1 = abs(lista[i])
        elif max2 < abs(lista[i]):
            max2 = abs(lista[i])
    return max1 * max2

# Ejercicio 3

def calcular_raiz(number):
    #Busqueda binaria
    if number == 1:
        return 1
    raiz = number//2
    while True:
        if number < (raiz+1)**2 and raiz**2 <= number:
            return raiz
        elif raiz**2 < number:
            raiz += raiz//2
        else:
            raiz = raiz//2

# Ejercicio 4

def caminos_desde_todos(grafo):
    # BFS visto en clases
    visited, queue = list(), [0]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            for v in range(len(grafo[vertex])):
                if grafo[vertex][v] == 1 and v not in visited:
                    queue.append(v)
    return len(visited) == len(grafo)