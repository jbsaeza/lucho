# Ejercicio 1

def asignar_colores(lineas, combinaciones):
    n_lineas = len(lineas)
    vecinos = [[] for i in range(n_lineas)]
    colores = [-1 for i in range(n_lineas)]
    # asignar estaciones a su linea de forma optima, asi saber en orden de 1 a que linea corresponde cada estación
    estaciones = {}
    i = 0
    while i < n_lineas:
        j = 0
        while j < len(lineas[i]):
            estaciones[str(lineas[i][j])] = i
            j += 1
        i += 1
    for comb in combinaciones:
        # Ver q lineas son las q corresponden al par
        linea_a = estaciones[str(comb[0])]
        linea_b = estaciones[str(comb[1])]
        # Ver si la linea tiene color asignado
        if colores[linea_a] != -1:
            # Linea a tiene color, linea b no tiene color
            if colores[linea_b] == -1:
                for i in range(n_lineas):
                    # Asignar color distinto a la otra linea
                    if i != colores[linea_a]:
                        # Asignar color distinto a otros vecinos
                        if i not in [colores[vecino] for vecino in vecinos[linea_b]]:
                            colores[linea_b] = i
                            break
        else:
            # Linea a no tiene color, linea b tiene color
            if colores[linea_b] != -1:
                for i in range(n_lineas):
                    # Asignar color distinto a la otra linea
                    if i != colores[linea_b]:
                        # Asignar color distinto a otros vecinos
                        if i not in [colores[vecino] for vecino in vecinos[linea_a]]:
                            colores[linea_a] = i
                            break
            # Linea a no tiene color, linea b no tiene color
            else:
                for i in range(n_lineas):
                    # Asignar color distinto a otros vecinos
                    if i not in [colores[vecino] for vecino in vecinos[linea_a]]:
                        colores[linea_a] = i
                        break
                for i in range(n_lineas):
                    # Asignar color distinto a la otra linea
                    if i != colores[linea_a]:
                        # Asignar color distinto a otros vecinos
                        if i not in [colores[vecino] for vecino in vecinos[linea_b]]:
                            colores[linea_b] = i
                            break
        # Agregar a vecinos
        if linea_b not in vecinos[linea_a]:
            vecinos[linea_a].append(linea_b)
        if linea_a not in vecinos[linea_b]:
            vecinos[linea_b].append(linea_a)

    return colores


# lineas = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
# combinaciones = {(0, 10), (1, 6)}

# colores = asignar_colores(lineas, combinaciones)
# print(colores)

# Ejercicio 2

def encontrar_ocurrencias(sopa, texto):
    posiciones = []
    m = len(sopa)
    for i in range(m):
        for j in range(m):
            if sopa[i][j] == texto[0]:
                # DFS
                ocs = enc_oc_dfs(sopa, texto, posiciones, [i, j], 0, [], [])
                if ocs != []:
                    for oc in ocs:
                        posiciones.append(oc)

    return posiciones


def enc_oc_dfs(sopa, texto, posiciones, posición, letra_a_buscar, ocurrencia, ocurrencias):
    oc_agregar = ocurrencia.copy()
    # Agregar a posiciones
    if texto[letra_a_buscar] in sopa[posición[0]][posición[1]]:
        oc_agregar.append((posición[0], posición[1]))
    # Si la letra agregada es la ultima, retornar
    if letra_a_buscar+1 == len(texto):
        if oc_agregar not in ocurrencias:
            ocurrencias.append(oc_agregar)
        return
    # Revisar proxima letra en las distintas direcciones
    letra_sig = texto[letra_a_buscar+1]
    # Revisar arriba
    if posición[1]-1 >= 0 and sopa[posición[0]][posición[1]-1] == letra_sig and (posición[0], posición[1]-1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0], posición[1]-1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar abajo
    if posición[1]+1 <= len(sopa)-1 and sopa[posición[0]][posición[1]+1] == letra_sig and (posición[0], posición[1]+1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0], posición[1]+1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar izquierda
    if posición[0]-1 >= 0 and sopa[posición[0]-1][posición[1]] == letra_sig and (posición[0]-1, posición[1]) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]-1, posición[1]], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar derecha
    if posición[0]+1 <= len(sopa)-1 and sopa[posición[0]+1][posición[1]] == letra_sig and (posición[0]+1, posición[1]) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]+1, posición[1]], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar arriba-izquierda
    if posición[1]-1 >= 0 and posición[0]-1 >= 0 and sopa[posición[0]-1][posición[1]-1] == letra_sig and (posición[0]-1, posición[1]-1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]-1, posición[1]-1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar arriba-derecha
    if posición[1]-1 >= 0 and posición[0]+1 <= len(sopa)-1 and sopa[posición[0]+1][posición[1]-1] == letra_sig and (posición[0]+1, posición[1]-1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]+1, posición[1]-1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar abajo-izquierda
    if posición[1]+1 <= len(sopa)-1 and posición[0]-1 >= 0 and sopa[posición[0]-1][posición[1]+1] == letra_sig and (posición[0]-1, posición[1]+1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]-1, posición[1]+1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Revisar abajo-derecha
    if posición[1]+1 <= len(sopa)-1 and posición[0]+1 <= len(sopa)-1 and sopa[posición[0]+1][posición[1]+1] == letra_sig and (posición[0]+1, posición[1]+1) not in oc_agregar:
        enc_oc_dfs(sopa, texto, posiciones, [
            posición[0]+1, posición[1]+1], letra_a_buscar+1, oc_agregar, ocurrencias)
    # Retornar ocurrencias encontradas
    return ocurrencias


# sopa = ["TAKXX", "AOEYF", "FCHTB", "GFKAR", "POSFD"]
# texto = "TAKA"
# sopa = ["LAMXB", "AOEYF", "FCHTB", "GFKAR", "POSFD"]
# texto = "HOLA"
# posiciones = encontrar_ocurrencias(sopa, texto)
# print(*posiciones, sep="\n")

# # Ejercicio 3


def programar_evaluaciones(evaluaciones):
    return (1, 1)


# evaluaciones = [("Tarea4", 9, 15),
#                 ("Control1", 2, 2),
#                 ("I1", 5, 18),
#                 ("Control3", 7, 1),
#                 ("Control2", 4, 25),
#                 ("Taller1", 2, 20),
#                 ("Tarea2", 5, 8),
#                 ("Tarea3", 7, 10),
#                 ("Taller2", 4, 12),
#                 ("Tarea1", 3, 5)]

# orden, nota = programar_evaluaciones(evaluaciones)
# print(orden)
# print(nota)


# Ejercicio 4
def search_relation(rel_dic, start, to_search):
    if start in rel_dic:
        for r in rel_dic[start]:
            if r in rel_dic and to_search in rel_dic[r]:
                return True
            else:
                return search_relation(rel_dic, r, to_search)
    return False


def ordenes_vacunacion(relaciones):
    rel = {}
    personas = []
    for r in relaciones:
        if str(r[0]) not in rel:
            rel[str(r[0])] = [str(r[1])]
        else:
            rel[str(r[0])].append(str(r[1]))
        if r[0] not in personas:
            personas.append(str(r[0]))
        if r[1] not in personas:
            personas.append(str(r[1]))
    ordenes = []
    ordenar(personas, rel, 0, ordenes, [])
    return ordenes


def ordenar(personas, rel_dic, p_actual, ordenes, orden):
    # Condicion de termino
    if p_actual == len(personas):
        ordenes.append(orden)
        return
    if orden == []:
        orden.append(personas[p_actual])
        ordenar(personas, rel_dic, p_actual+1, ordenes, orden)
    else:
        # agregar por insertion sort
        value = personas[p_actual]
        if value not in orden:
            orden.append(value)
            largo_o = len(orden)
            test_slot = largo_o - 2
            while test_slot > -1 and value in rel_dic and (orden[test_slot] in rel_dic[value] or search_relation(rel_dic, value, orden[test_slot])):
                orden[test_slot + 1] = orden[test_slot]
                test_slot += -1
            if not test_slot == largo_o - 1:
                orden[test_slot + 1] = value
            # Ver que el elemento siguiente sea mayor o no que el elemento agregado
            if test_slot > -1:
                if (orden[test_slot] not in rel_dic or value not in rel_dic[orden[test_slot]]) and not search_relation(rel_dic, orden[test_slot], value):
                    # Aqui se generan multiples ordenes
                    while test_slot > -1 and (value not in rel_dic or orden[test_slot] not in rel_dic[value]) and not search_relation(rel_dic, orden[test_slot], value):
                        orden_n = orden.copy()
                        orden_n[test_slot + 1] = orden_n[test_slot]
                        test_slot += -1
                        orden_n[test_slot + 1] = value
                        ordenar(personas, rel_dic,
                                p_actual+1, ordenes, orden_n)
        return ordenar(personas, rel_dic, p_actual+1, ordenes, orden)


# relaciones = [(0, 6), (1, 2), (1, 4), (1, 6), (3, 0),
#               (3, 4), (5, 1), (7, 0), (7, 1)]
# resultado = ordenes_vacunacion(relaciones)
# print(resultado)

# relaciones = [(0,6),(1,2),(1,4),(1,6),(3,0),(3,4),(5,1),(6,3),(7,0),(7,1)]
# resultado = ordenes_vacunacion(relaciones)
# print(resultado)


def main():
    # Ejercicio 1
    # lineas = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    # combinaciones = {(0, 10), (1, 6)}

    # colores = asignar_colores(lineas, combinaciones)
    # print(colores)

    # Ejercicio 2
    # sopa = ["TAKXX", "AOEYF", "FCHTB", "GFKAR", "POSFD"]
    # texto = "TAKA"
    # sopa = ["LAMXB", "AOEYF", "FCHTB", "GFKAR", "POSFD"]
    # texto = "HOLA"
    # posiciones = encontrar_ocurrencias(sopa, texto)
    # print(*posiciones, sep="\n")

    # Ejercicio 3, no resuelto
    # evaluaciones = [("Tarea4", 9, 15),
    #                 ("Control1", 2, 2),
    #                 ("I1", 5, 18),
    #                 ("Control3", 7, 1),
    #                 ("Control2", 4, 25),
    #                 ("Taller1", 2, 20),
    #                 ("Tarea2", 5, 8),
    #                 ("Tarea3", 7, 10),
    #                 ("Taller2", 4, 12),
    #                 ("Tarea1", 3, 5)]

    # orden, nota = programar_evaluaciones(evaluaciones)
    # print(orden)
    # print(nota)

    # Ejercicio 4, no se llego al resultado
    # relaciones = [(0, 6), (1, 2), (1, 4), (1, 6), (3, 0),
    #               (3, 4), (5, 1), (7, 0), (7, 1)]
    # resultado = ordenes_vacunacion(relaciones)
    # print(resultado)

    # relaciones = [(0,6),(1,2),(1,4),(1,6),(3,0),(3,4),(5,1),(6,3),(7,0),(7,1)]
    # resultado = ordenes_vacunacion(relaciones)
    # print(resultado)

    return
