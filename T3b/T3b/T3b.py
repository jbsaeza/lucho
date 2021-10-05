import sqlite3

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

# 1
# Para todos los niveles, con excepción de JR, encuentre el nivel y el promedio de edad de los estudiantes del nivel.
cursor.execute(
    "SELECT nivel, AVG (E.edad) FROM Estudiantes E GROUP BY E.nivel HAVING E.nivel != 'JR'"
)

# print(cursor.fetchall())

# 2
# Encuentre los nombres de todos los alumnos que no estan inscritos en ningún curso.
cursor.execute(
    "SELECT E.nombre FROM Estudiantes E WHERE E.num NOT IN (SELECT I.num_est FROM Inscritos I)"
)

# print(cursor.fetchall())

# 3
# Encuentre los nombres de los profesores para los cuales la suma de estudiantes de todos los cursos que dictan es menor que 5 (considere solo aquellos cursos con al menos 1 estudiante inscrito).

cursor.execute(
    "SELECT X.nombre FROM (SELECT COUNT (I.num_est) AS c_est, P.nombre AS nombre FROM Inscritos I, Profesores P, Cursos C WHERE I.nombre_curso == C.nombre AND C.id_profesor == P.id GROUP BY P.nombre) AS X WHERE X.c_est < 5"
)

# print(cursor.fetchall())

# 4
# Encuentre los nombres de todos los cursos que tienen catedra en la sala R128 o tienen cinco o mas estudiantes.

cursor.execute(
    "SELECT C.nombre FROM Cursos C WHERE C.sala == 'R128' UNION SELECT Y.nombre FROM (SELECT COUNT(I.num_est) AS c_est, I.nombre_curso AS nombre FROM Inscritos I GROUP BY I.nombre_curso) AS Y WHERE Y.c_est > 4"
)

# print(cursor.fetchall())


connection.close()
