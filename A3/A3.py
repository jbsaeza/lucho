import sqlite3

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

# Mision 1

cursor.execute(
    "CREATE TABLE empleados(id INTEGER, nombre TEXT, edad INTEGER, sueldo FLOAT, PRIMARY KEY(id))")
cursor.execute(
    "CREATE TABLE trabaja_en(id_empleado INTEGER, id_depto INTEGER, porcentaje_tiempo INTEGER, PRIMARY KEY(id_empleado, id_depto), FOREIGN KEY (id_empleado) REFERENCES empleados, FOREIGN KEY (id_depto) REFERENCES departamentos)")
cursor.execute(
    "CREATE TABLE departamentos(id INTEGER, nombre TEXT, presupuesto FLOAT, id_jefe INTEGER, PRIMARY KEY(id), FOREIGN KEY (id_jefe) REFERENCES empleados)")

with open("empleado.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        row = line.strip().split(",")
        cursor.execute(
            "INSERT INTO empleados VALUES ({}, '{}', {}, {})".format(row[0], row[1], row[2], row[3]))

with open("trabaja_en.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        row = line.strip().split(",")
        cursor.execute(
            "INSERT INTO trabaja_en VALUES ({}, {}, {})".format(row[0], row[1], row[2]))

with open("departamento.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        row = line.strip().split(",")
        cursor.execute(
            "INSERT INTO departamentos VALUES ({}, '{}', {}, {})".format(row[0], row[1], row[2], row[3]))


# Mision 2

# C1
cursor.execute(
    "SELECT DISTINCT E.nombre, E.edad FROM empleados E, trabaja_en T, departamentos D WHERE E.id==T.id_empleado AND T.id_depto==D.id AND (D.nombre=='Software' OR D.nombre=='Hardware')"
)

# C2
cursor.execute(
    "SELECT E.nombre, D.presupuesto FROM empleados E, departamentos D WHERE E.id==D.id_jefe AND D.presupuesto == (SELECT MAX(D.presupuesto) FROM departamentos D)"
)

# C3
cursor.execute(
    "SELECT E.nombre, E.sueldo FROM empleados E, (SELECT E.nombre, E.id, MAX (D.presupuesto) max_pres FROM empleados E, trabaja_en T, departamentos D WHERE E.id==T.id_empleado AND T.id_depto==D.id GROUP BY E.nombre) AS X WHERE E.id == X.id AND E.sueldo > X.max_pres"
)

# C4
cursor.execute(
    "SELECT X.nombre, X.c_emp FROM (SELECT D.nombre, COUNT(E.nombre) AS c_emp FROM empleados E, trabaja_en T, departamentos D WHERE E.id==T.id_empleado AND T.id_depto==D.id AND T.porcentaje_tiempo == 100 GROUP BY D.nombre) AS X WHERE X.c_emp >= 20"
)

# print(cursor.fetchall())
