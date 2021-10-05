import json


class Movie:

    instances = []

    def __init__(self, title, year):
        self.title = title
        self.genres = []
        self.cast = []
        self.year = year
        Movie.instances.append(self)

class Actor:

    instances = []
    names = []

    def __init__(self, name):
        self.name = name
        Actor.instances.append(self)
        Actor.names.append(name)

class Genre:

    instances = []
    names = []

    def __init__(self, name):
        self.name = name
        Genre.instances.append(self)
        Genre.names.append(name)


#movie: {title, year, cast, genres}
with open('movies.json', encoding = 'utf8') as movies_file:
    movies = json.load(movies_file)
    for movie in movies:
        m = Movie(movie['title'], movie['year'])
        for genre in movie['genres']:
            if genre not in Genre.names:
                Genre(genre)
            m.genres.append(genre)
        for actor in movie['cast']:
            if actor not in Actor.names:
                Actor(actor)
            m.cast.append(actor)

def top5_populars_genres():
    genres = {}
    for movie in Movie.instances:
        for genre in movie.genres:
            if genre not in genres.keys():
                genres[genre] = 1
            else:
                genres[genre] += 1
    genres = [(key, genres[key]) for key in genres.keys()]
    genres.sort(key=(lambda x: x[1]))
    print(genres[-5:])

def top3_years_most_movies():
    years = {}
    for movie in Movie.instances:
        if movie.year not in years.keys():
            years[movie.year] = 1
        else:
            years[movie.year] += 1
    years = [(key, years[key]) for key in years.keys()]
    years.sort(key=(lambda x: x[1]))
    print(years[-3:])

def top5_oldest_actors():
    top_actors = {}
    for movie in Movie.instances:
        for actor in movie.cast:
            if actor not in top_actors.keys():
                top_actors[actor] = {}
                top_actors[actor]['min'] = movie.year
                top_actors[actor]['max'] = movie.year
            elif top_actors[actor]['max'] < movie.year:
                top_actors[actor]['max'] = movie.year
            elif movie.year < top_actors[actor]['min']:
                top_actors[actor]['min'] = movie.year
    top_actors = [(key, top_actors[key]['min'], top_actors[key]['max']) for key in top_actors.keys()]
    top_actors.sort(key=(lambda x: x[2] - x[1]))
    print(top_actors[-5:])

def most_freq_cast():
    casts = []
    for movie in Movie.instances:
        if 2 <= len(movie.cast):
            added = False
            for cast in casts:
                if all([actor in cast[0] for actor in movie.cast]):
                    cast[1] += 1
                    added = True
                    break
            if not added:
                casts.append([movie.cast, 1])
    casts.sort(key=(lambda x: x[1]))
    print(casts[-1])

top5_populars_genres()
top3_years_most_movies()
top5_oldest_actors()
most_freq_cast()