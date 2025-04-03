"""
utility functions for mapping between different movie ID formats.
"""
import os
import csv

def map_tmdb_to_movielens(tmdb_favorite_movies, data_path="data"):
    """
    map TMDB movie IDs to MovieLens movie IDs.
    
    Args:
        tmdb_favorite_movies (list): List of TMDB movie IDs
        data_path (str): path to directory containing links.csv
    
    Returns:
        list: mapped MovieLens movie IDs
    """
    if not tmdb_favorite_movies:
        print("No TMDB favorite movies provided")
        return []
    
    print(f"mapping TMDB ids to MovieLens ids: {tmdb_favorite_movies}")
    
    # find links.csv file
    possible_data_dirs = [
        data_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', data_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', data_path)
    ]
    
    # find first directory that contains links.csv
    links_path = None
    for possible_dir in possible_data_dirs:
        temp_path = os.path.join(possible_dir, 'links.csv')
        if os.path.exists(temp_path):
            links_path = temp_path
            print(f"Found links.csv at: {links_path}")
            break
    
    if not links_path:
        print(f"ERROR: Could not find links.csv in any of the expected locations.")
        return []
    
    # mapping dict
    tmdb_to_movielens = {}
    
    # read links.csv and create mapping
    try:
        with open(links_path, 'r') as csvfile:
            
            csvreader = csv.reader(csvfile)
            header = next(csvreader, None)
            
            # get column indexes
            tmdb_idx = 2  
            ml_idx = 0    
            
            if header:
                # find column indexes from header
                for i, col in enumerate(header):
                    if col.lower() == 'tmdbid':
                        tmdb_idx = i
                    elif col.lower() == 'movieid':
                        ml_idx = i
            
            # read and create mapping
            for row in csvreader:
                if len(row) > max(tmdb_idx, ml_idx):
                    ml_id = row[ml_idx].strip()
                    tmdb_id = row[tmdb_idx].strip() if row[tmdb_idx].strip() else None
                    
                    # handle floating point format for tmdbId
                    if tmdb_id and '.' in tmdb_id:
                        tmdb_id = tmdb_id.split('.')[0]
                    
                    # only add valid tmdb IDs
                    if tmdb_id and tmdb_id.strip():
                        tmdb_to_movielens[tmdb_id.strip()] = ml_id
        
        # map favorite movies
        favorite_movies_ml = []
        mapped_ids = []
        
        for tmdb_id in tmdb_favorite_movies:
            # convert to string and remove decimal points
            tmdb_str = str(tmdb_id).split('.')[0].strip()
            
            if tmdb_str in tmdb_to_movielens:
                ml_id = tmdb_to_movielens[tmdb_str]
                favorite_movies_ml.append(int(ml_id)) 
                mapped_ids.append((tmdb_str, ml_id))
            else:
                # try an integer comparison if string comparison failed
                try:
                    tmdb_int = int(tmdb_str)
                    # look for matching int values
                    for key, value in tmdb_to_movielens.items():
                        try:
                            if int(key) == tmdb_int:
                                favorite_movies_ml.append(int(value))
                                mapped_ids.append((tmdb_str, value))
                                break
                        except ValueError:
                            continue
                except ValueError:
                    pass
        
        print(f"Successfully mapped {len(mapped_ids)} out of {len(tmdb_favorite_movies)} IDs")
        
        # if at least one successful mapping, return it
        if len(favorite_movies_ml) > 0:
            return favorite_movies_ml
        
        # otherwise use hardcoded popular MovieLens idsfallback
        print("using fallback MovieLens ids no mapping  successful")
        return [1, 2, 3, 4, 5]  # Return integers instead of strings
    
    except Exception as e:
        print(f"error mapping TMDB to MovieLens ids: {e}")
        import traceback
        traceback.print_exc()
        return []


def map_movielens_to_tmdb(movielens_ids, data_path="data"):
    """
    Map MovieLens movie IDs back to TMDB movie IDs.
    
    Args:
        movielens_ids (list): List of MovieLens movie IDs
        data_path (str): path to directory containing links.csv
    
    Returns:
        list: mapped TMDB movie IDs
    """
    if not movielens_ids:
        print("no MovieLens ids provided for mapping to tmbb")
        return []
        
    print(f"Mapping MovieLens IDs to TMDB IDs: {movielens_ids}")
    
    # find links.csv file
    possible_data_dirs = [
        data_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', data_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', data_path)
    ]
    
    # find the first directory that contains links.csv
    links_path = None
    for possible_dir in possible_data_dirs:
        temp_path = os.path.join(possible_dir, 'links.csv')
        if os.path.exists(temp_path):
            links_path = temp_path
            print(f"Found links.csv at: {links_path}")
            break
    
    if not links_path:
        print(f"couldnt find links.csv")
        return []
    
    
    movielens_to_tmdb = {}
    

    try:
        with open(links_path, 'r') as csvfile:
            
            csvreader = csv.reader(csvfile)
            header = next(csvreader, None)
            
            
            tmdb_idx = 2 
            ml_idx = 0    
            
            if header:
               
                for i, col in enumerate(header):
                    if col.lower() == 'tmdbid':
                        tmdb_idx = i
                    elif col.lower() == 'movieid':
                        ml_idx = i
            
            
            for row in csvreader:
                if len(row) > max(tmdb_idx, ml_idx):
                    ml_id = row[ml_idx].strip()
                    tmdb_id = row[tmdb_idx].strip() if row[tmdb_idx].strip() else None
                    #clean tmdb id
                    if tmdb_id and '.' in tmdb_id:
                        tmdb_id = tmdb_id.split('.')[0]
                    
                    if tmdb_id and tmdb_id.strip():
                        movielens_to_tmdb[ml_id] = tmdb_id
        
        # map MovieLens ids to TMDB ids
        tmdb_ids = []
        mapped_ids = []
        
        for ml_id in movielens_ids:
            # convert to string
            ml_str = str(ml_id).strip()
            
            if ml_str in movielens_to_tmdb:
                tmdb_id = movielens_to_tmdb[ml_str]
                tmdb_ids.append(tmdb_id)
                mapped_ids.append((ml_str, tmdb_id))
            else:
                # try integer comparison if string comparison fails
                try:
                    ml_int = int(ml_str)
                    # look for matching int values
                    for key, value in movielens_to_tmdb.items():
                        try:
                            if int(key) == ml_int:
                                tmdb_ids.append(value)
                                mapped_ids.append((ml_str, value))
                                break
                        except ValueError:
                            continue
                except ValueError:
                    pass
        
        print(f"Successfully mapped {len(mapped_ids)} out of {len(movielens_ids)} IDs")
        
        if len(tmdb_ids) > 0:
            return tmdb_ids
        
        print(" no valid TMDB ids were mapped from MovieLens ids")
        # provide fallback recs
        print("Using fallback TMDB IDs")
        return ["299534", "299536", "24428", "299537", "10138"]
            
    except Exception as e:
        print(f"Error mapping MovieLens to TMDB IDs: {e}")
        import traceback
        traceback.print_exc()
        return []