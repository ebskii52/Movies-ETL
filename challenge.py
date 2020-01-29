#%%
import json
import pandas as pd
import numpy as np
import re
#Import Modules
from sqlalchemy import create_engine
import psycopg2

# %%
## Create the Database Engine
## The database engine needs to know how to connect to the database. To do that, we make a connection string. 
## For PostgreSQL, the connection string will look like the following:
from config import db_password

db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data"

#%%
# This is all the information that SQLAlchemy needs to create a database engine.
# SQLAlchemy handles connections to different SQL databases and manages the conversion between data types. 
# The way it handles all the communication and conversion is by creating a database engine.

# Create the database engine with the following:
engine = create_engine(db_string)

file_dir = 'C:/Users/esrampur/Documents/BootCamp Berkley/Movie-ETL/Movies-ETL/'

f'{file_dir}wikipedia.movies.json'


#%%
with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_source= json.load(file)


#%%
####################################################################################################################
### Clean and Transform Kaggle Meta Data From movies_metadata.csv
####################################################################################################################
kaggle_data = pd.read_csv(f'{file_dir}/the-movies-dataset/movies_metadata.csv')

#%%
####################################################################################################################
### Clean and Transform Ratings Data From ratings.csv
####################################################################################################################
MovieLensrating = pd.read_csv(f'{file_dir}/the-movies-dataset/ratings.csv')

#%%
########################################################################################################
## Module Challenge
########################################################################################################

def DataSource(wiki_movies_raw, kaggle_metadata, ratings):
    try:
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

        wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                    and 'imdb_link' in movie
                    and 'No. of episodes' not in movie]

        clean_movies = [clean_movie(movie) for movie in wiki_movies]

        wiki_movies_df = pd.DataFrame(clean_movies)
        sorted(wiki_movies_df.columns.tolist())

        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')

        wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

        wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
        wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

        box_office = wiki_movies_df['Box office'].dropna() 

        def is_not_a_string(x):
            return type(x) != str

        box_office[box_office.map(is_not_a_string)]

        box_office[box_office.map(lambda x: type(x) != str)]

        box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

        form_one = r'\$\d+\.?\d*\s*[mb]illion'
        box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

        form_two = r'\$\d{1,3}(?:,\d{3})+'
        box_office.str.contains(form_two, flags=re.IGNORECASE).sum()

        matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
        matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

        box_office[~matches_form_one & ~matches_form_two]

        form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
        form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

        box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
        box_office.str.extract(f'({form_one}|{form_two})')

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

        wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

        wiki_movies_df.drop('Box office', axis=1, inplace=True)

        budget = wiki_movies_df['Budget'].dropna()

        budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

        budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

        wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

        wiki_movies_df.drop('Budget', axis=1, inplace=True)

        release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


        date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
        date_form_two = r'\d{4}.[01]\d.[123]\d'
        date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
        date_form_four = r'\d{4}'

        release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

        wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

        running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

        running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]

        running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()

        running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

        running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

        running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

        wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

        wiki_movies_df.drop('Running time', axis=1, inplace=True)

        #kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
        #kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
        kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
        kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
        kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
        kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
        #kaggle_metadata['budget'] = pd.to_numeric(kaggle_metadata['budget'], errors='coerce')
        kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
        kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
        kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

        
        ratings.info(null_counts=True)
        pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        # First, we need to use a groupby on the “movieId” and “rating” columns and take the count for each group.
        rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()
        # Then we’ll rename the “userId” column to “count.”
        rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1) 
        ## Now the magical part. We can pivot this data so that movieId is the index, the columns 
        ## will be all the rating values, and the rows will be the counts for each rating value.
        rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating', values='count')
        rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

        ## Loading the Tables into SQL
        wiki_movies_df.to_sql(name='movies', con=engine)
        kaggle_metadata.to_sql(name='kaggledata', con=engine)
        rating_counts.to_sql(name='ratings', con=engine)

    except:
        print("There exist a problem in data evaluation")
        pass

#%%

DataSource(wiki_movies_source, kaggle_data,MovieLensrating )
