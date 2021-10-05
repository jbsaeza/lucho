# Ejecicio 1

def calcular_ocurrencias(palabras):
    words = {}
    for word in palabras:
        if word not in words.keys():
            words[word] = 1
        else:
            words[word] += 1
    ocurrencias = []
    for word in words:
        ocurrencias.append((word, words[word]))
    return ocurrencias


palabras = ['avión', 'perro', 'gato', 'avión', 'edificio', 'gato']
ocurrencias = calcular_ocurrencias(palabras)
print(ocurrencias)

# Ejercicio 2