import json
import sqlite3
import urllib.request as net
import ssl
import bs4

connection = sqlite3.connect('data.db')
cursor = connection.cursor()


# M0
# Código obtenido de la pauta del taller 3a, https://github.com/IIC2115/Syllabus/blob/main/Pautas/Talleres/T3a_pauta.ipynb
# Se utiliza este código ya que cumple con lo pedido en esta mision y fue entregado por el cuerpo docente.

# Se crean las 3 tablas de entidades mas 2 tablas de relaciones entre entidades
cursor.execute(
    "CREATE TABLE Movies(mid INTEGER PRIMARY KEY, title TEXT, year INTEGER)")
cursor.execute("CREATE TABLE Actors(aid INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("CREATE TABLE Genres(gid INTEGER PRIMARY KEY, genre TEXT)")
cursor.execute("CREATE TABLE ActorsMovies(actor_id INTEGER, movie_id INTEGER, FOREIGN KEY (actor_id) REFERENCES Actors, FOREIGN KEY (movie_id) REFERENCES Movies)")
cursor.execute("CREATE TABLE GenresMovies(genre_id INTEGER, movie_id INTEGER, FOREIGN KEY (genre_id) REFERENCES Genres, FOREIGN KEY (movie_id) REFERENCES Movies)")

# Variables de indice y almacenamiento de generos y actores unicos
mid = 1
aid = 1
gid = 1
genres = {}
actors = {}

with open("movies.json", encoding="utf8") as movies_file:
    movies = json.load(movies_file)
    for movie in movies:

        title = movie["title"]
        year = movie["year"]

        if len(title) > 1 and int(year):

            cursor.execute("INSERT INTO Movies VALUES (?,?,?)",
                           (mid, title, year))

            for genre in movie["genres"]:
                # Se utiliza el diccionario genres para saber si ya se había agregado el genero y se almacena el gid
                if not genre.islower() and not genre.isupper() and len(genre) > 1:
                    if genre not in genres:
                        genres[genre] = gid
                        cursor.execute(
                            "INSERT INTO Genres VALUES (?,?)", (gid, genre))
                        gid += 1
                    cursor.execute("INSERT INTO GenresMovies VALUES (?,?)",
                                   (genres[genre], mid))

            for actor in movie["cast"]:
                if not actor.islower() and not actor.isupper() and len(actor) > 1 and actor.lower() != "the":
                    if actor not in actors:
                        # Analogo a genre
                        actors[actor] = aid
                        cursor.execute(
                            "INSERT INTO Actors VALUES (?,?)", (aid, actor))
                        aid += 1
                    cursor.execute("INSERT INTO ActorsMovies VALUES (?,?)",
                                   (actors[actor], mid))

            mid += 1


# M1

# Código obtenido del material de clases/capitulo 3/Parte a, https://github.com/IIC2115/Syllabus/blob/main/Pautas/Talleres/T3a_pauta.ipynb
# Se utiliza este código con el propósito de obtener el código fuente de la pagina de wikipedia.
class WebDownloader:

    def __init__(self, link):
        self.user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        self.url = link

    def getHtmlAsString(self):
        headers = {'User-Agent': self.user_agent}
        request = net.Request(self.url, None, headers)
        gcontext = ssl.SSLContext()
        response = net.urlopen(request, context=gcontext)
        return response.read().decode('utf-8')


url = "https://en.wikipedia.org/wiki/"

cursor.execute("ALTER TABLE Genres ADD Description TEXT")
for genre in genres:
    if genre == "Found Footage":
        wd = WebDownloader(url+"Found_footage_(film_technique)")
    elif genre == "Martial Arts":
        wd = WebDownloader(url+'Martial_arts_film')
    elif genre == "Live Action":
        wd = WebDownloader(url+genre.replace(" ", "_"))
    elif genre == "Performance":
        wd = WebDownloader(url+genre)
    elif " " in genre:
        wd = WebDownloader(url+genre.replace(" ", "_")+"_film")
    else:
        wd = WebDownloader(url+genre+"_film")
    sourceCode = wd.getHtmlAsString()
    soup = bs4.BeautifulSoup(sourceCode)
    if genre == "Horror":
        for node in soup.findAll('p', limit=5):
            text = str(u''.join(node.findAll(text=True)).encode('utf-8'))[2:-1]
            if len(text) > 3:
                cursor.execute(
                    "UPDATE Genres SET Description = ? WHERE genre == ?", (text, genre))
                break
    else:
        for node in soup.findAll('p', limit=3):
            text = str(u''.join(node.findAll(text=True)).encode('utf-8'))[2:-1]
            if len(text) > 3:
                cursor.execute(
                    "UPDATE Genres SET Description = ? WHERE genre == ?", (text, genre))
                break

# Este webscraping demora bastante
cursor.execute("ALTER TABLE Actors ADD Description TEXT")
for actor in actors:
    names = ""
    for name in actor.split(" "):
        names += "_" + name
    wd = WebDownloader(url+names)
    try:
        sourceCode = wd.getHtmlAsString()
    except:
        pass
    soup = bs4.BeautifulSoup(sourceCode)
    for node in soup.findAll('p', limit=3):
        text = str(u''.join(node.findAll(text=True)).encode('utf-8'))[2:-1]
        if "may refer to" in text:
            break
        elif len(text) > 3:
            cursor.execute(
                "UPDATE Actors SET Description = ? WHERE name == ?", (text, actor))
            break

# M2

# 1
cursor.execute(
    "SELECT count(M.title) as n_movies, M.year FROM Movies M GROUP BY M.year ORDER BY n_movies LIMIT 3")
# print(cursor.fetchall())


# 2
cursor.execute(
    "SELECT A.name, (MAX(M.year) - MIN(M.year) + 1) as career FROM Movies M, Actors A, ActorsMovies AM WHERE M.mid == AM.movie_id AND A.aid == AM.actor_id GROUP BY A.name ORDER BY career DESC LIMIT 5")
# print(cursor.fetchall())

# 3
participacion_actor_genero = "SELECT A.name as name, G.genre as genre, G.gid as genre_id, count(M.mid) as n_movies FROM Movies M, Genres G, Actors A, GenresMovies GM, ActorsMovies AM WHERE M.mid == GM.movie_id AND G.gid == GM.genre_id AND A.aid == AM.actor_id AND M.mid == AM.movie_id GROUP BY A.name, G.genre"
top_actors_genre = "SELECT PAG.genre, GROUP_CONCAT(PAG.name) as top_actors, PAG.genre_id FROM ({}) AS PAG WHERE (SELECT count(*) FROM ({}) AS PAGS WHERE PAGS.genre == PAG.genre AND PAGS.n_movies > PAG.n_movies) < 3 GROUP BY PAG.genre".format(
    participacion_actor_genero, participacion_actor_genero)
cursor.execute(
    "SELECT G.genre, TOP_AG.top_actors FROM Movies M, Genres G, GenresMovies GM, ({}) as TOP_AG WHERE M.mid == GM.movie_id AND G.gid == GM.genre_id AND TOP_AG.genre_id == G.gid GROUP BY G.genre ORDER BY count(GM.movie_id) DESC".format(top_actors_genre))
# print(cursor.fetchall())
