#%%
import json
import pandas as pd
import numpy as np

file_dir = 'C:/Users/esrampur/Documents/BootCamp Berkley/Movie-ETL/Movies-ETL/'

f'{file_dir}wikipedia.movies.json'

# %%

with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

len(wiki_movies_raw)

# %%
# First 5 records
wiki_movies_raw[:5]

# %%
# Last 5 records
wiki_movies_raw[-5:]

# %%
# Some records in the middle
wiki_movies_raw[3600:3605]

# %%

kaggle_metadata = pd.read_csv(f'{file_dir}/the-movies-dataset/movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}/the-movies-dataset/ratings.csv')

MetaDataDF = pd.DataFrame(kaggle_metadata)

MetaDataDF.sample(n=5)

# %%
wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()

# %%
wiki_movies_df.columns.tolist()

# %%
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
wiki_movies

# %%
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

# %%
x = 'global value'

def foo():
    x = 'local value'
    print(x)

foo()
print(x)


# %%
my_list = [1,2,3]

def append_four(x):
    x.append(4)


append_four(my_list)
print(my_list)    

# %%

# %%
square = lambda x: x * x
square(5)

# %%
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie

#%%

wiki_movies_df[wiki_movies_df['Arabic'].notnull()]

# %%
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']

# %%
wiki_movies_df['Arabic'].head()

# %%
sorted(wiki_movies_df.columns.tolist())

#%%
wiki_movies_df[wiki_movies_df['Russian'].notnull()]

# %%
wiki_movies_df[wiki_movies_df.isna()]['url'].count()

# %%
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}

    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key] ## copy the corresponding key value into alt-title
            movie.pop(key)
            
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)

    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie
# %%
clean_movies = [clean_movie(movie) for movie in wiki_movies]

wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())

# %%
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))

# %%
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))

# %%
wiki_movies_df.head()

# %%
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

# %%
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

print(len(wiki_movies_df.columns.tolist()))

# %%
print(len(wiki_movies_df))

# %%
wiki_movies_df.dtypes

# %%
box_office = wiki_movies_df['Box office'].dropna() 

#%%
box_office.head()

# %%
def is_not_a_string(x):
    return type(x) != str

box_office[box_office.map(is_not_a_string)]

# %%
box_office.head()

# %%
box_office[box_office.map(lambda x: type(x) != str)]

# %%
some_list = ['One','Two','Three']
'Mississippi'.join(some_list)

# %%
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
box_office

# %%
import re
form_one = r'\$\d+\.?\d*\s*[mb]illion'
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

# %%
form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()

# %%
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

#%%
matches_form_one 


# %%
box_office[~matches_form_one & ~matches_form_two]

#%%
box_office[~matches_form_one & ~matches_form_two]

# %%
form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'

#%%
form_one = r'\$\s*\d+(?:[,\.]\d{3})+(?!\s[mb]illion)'
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

#%%
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

# %%
box_office

# %%
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'

# %%
box_office.str.extract(f'({form_one}|{form_two})')

# %%
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

# %%
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# %%
wiki_movies_df.drop('Box office', axis=1, inplace=True)

# %%
wiki_movies_df.head()

# %%
budget = wiki_movies_df['Budget'].dropna()

budget.head()

#%%
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
budget.tail()

# %%
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
budget.tail()

# %%
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

#%%

wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

# %%
wiki_movies_df['budget'].head()

# %%
wiki_movies_df.drop('Budget', axis=1, inplace=True)

#%%
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# %%
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'

# %%
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

#%%
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

#%%
wiki_movies_df['Running time']

#%%
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

running_time[727]


# %%
running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# %%
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# %%
running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]
running_time[727]

# %%
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

# %%
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

# %%
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

# %%
wiki_movies_df.drop('Running time', axis=1, inplace=True)


#%%
kaggle_metadata = pd.read_csv(f'{file_dir}/the-movies-dataset/movies_metadata.csv')

# %%
kaggle_metadata.dtypes

#%%
kaggle_metadata['adult'].value_counts()

# %%
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

#%%
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

kaggle_metadata.head()


# %%
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

#%%
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


#%%
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
#%%
ratings.info(null_counts=True)

#%%
pd.to_datetime(ratings['timestamp'], unit='s')

#%%
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

#%%
ratings['rating'].plot(kind='hist')
ratings['rating'].describe()

# %%
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# %%
movies_df[['title_wiki','title_kaggle']]
 

#%%
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]

# %%
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# %%
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# %%
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# %%
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# %%
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


#%%
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


#%%
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

#%%
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

# %%
movies_df['Language'].value_counts()

# %%
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

# %%
movies_df['original_language'].value_counts(dropna=False)

# %%
movies_df[['Production company(s)','production_companies']]

#%%
movies_df.head()

# %%
movies_df.drop(columns=['title_wiki','release_date','Language','Production company(s)'], inplace=True)

# %%
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

movies_df

# %%
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)

#%%
#movies_df['video'].value_counts(dropna=True)

# %%
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]

# %%
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)

# %%
# First, we need to use a groupby on the “movieId” and “rating” columns and take the count for each group.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()

# %%
# Then we’ll rename the “userId” column to “count.”
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1) 

#%%
rating_counts

# %%
## Now the magical part. We can pivot this data so that movieId is the index, the columns 
## will be all the rating values, and the rows will be the counts for each rating value.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating', values='count')


# %%
rating_counts

# %%
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

# %%
## This time, we need to use a left merge, since we want to keep everything in movies_df:
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

# %%
movies_with_ratings_df.head()

# %%
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

#%%
movies_with_ratings_df.head()

# %%
## 8.5.1: Connect Pandas and SQL
## Now that we’ve extracted and transformed our data, it’s time to load it into a SQL database. 
# We’re going to create a new database and use the built-in to_sql() method in Pandas 
# to create a table for our merged movie data. We’ll also import the raw ratings data into its own table.

#%%
#Import Modules
from sqlalchemy import create_engine
import psycopg2

# %%
## Create the Database Engine
## The database engine needs to know how to connect to the database. To do that, we make a connection string. 
## For PostgreSQL, the connection string will look like the following:
db_password = '11223344'

db_string = f"postgresql://postgres:{db_password}@localhost:53084/movie_data"

#%%
# This is all the information that SQLAlchemy needs to create a database engine.
# SQLAlchemy handles connections to different SQL databases and manages the conversion between data types. 
# The way it handles all the communication and conversion is by creating a database engine.

# Create the database engine with the following:
engine = create_engine(db_string)

# %%
# Import the Movie Data
# To save the movies_df DataFrame to a SQL table, 
# we only have to specify the name of the table and the engine in the to_sql() method.
movies_df.to_sql(name='movies', con=engine)

# %%
import time

rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}the-movies-dataset/ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')
