def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#import gen_population as pop
import facility_ABM
import pandas as pd
import geopandas as gpd


"""
run_model.py
code to run the facility placement abm

"""

import numpy as np 
import pandas as pd 
import geopandas as gpd
import argparse

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import shapely as shp
from shapely.geometry import Point
from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
from copy import deepcopy

from facility_ABM import *
from multiprocessing import Pool,freeze_support
from pathlib import Path

EPSG_LON_LAT = 4326 # the EPSG which represents pairs of lon lat points


def get_parser():
    parser = argparse.ArgumentParser(
        usage=f"python {__name__}.py"    
    )
    parser.add_argument(
        "--n_facilities",
        type=int,
        default=1070
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1
    )
    parser.add_argument(
        "--legal_states_only",
        #type=bool,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--pop_size",
        type=int,
        default=-1#if set to -1, the full population will be used
    )
    parser.add_argument(
        "--move_type",
        type=str,
        default='pres_weighted', # 'random', 'pres_weighted'
    )
    return parser


def main(
    n_facilities: int = 1070,
    num_steps: int = 4,
    legal_states_only: bool = True,
    out_path: Path = Path("./"),
    pop_size: int = -1,
    beta: int = 1,
    move_type: str = 'random',  # 'pres_weighted'
    min_portion: float = 0, # Threshold for a weighted county to be included
):
    """
    GLOBAL VARIABLES
    """
    COUNTY_SHP_FILE = "data/UScounties/UScounties.shp"
    BANNED_STATES = ['Arkansas','Alabama','Idaho','Kentucky','Louisiana','Kentucky','Mississippi','Missouri','Oklahoma','South Dakota','Tenesee','Texas','West Virginia','Wisconsin']
    EPSG = 4
    if beta <= 0:
        EXTREMUM = "MAX"
    else:
        EXTREMUM = "MIN"
    """"
    init the model for a single facility
    """
    print("loading pop point data...")
    df_pop = pd.read_csv("data/simulated_pop_points.csv")
    print(df_pop)
    gdf_pop = gpd.GeoDataFrame(df_pop,geometry = gpd.points_from_xy(df_pop.lon,df_pop.lat)).rename(columns = {"Unnamed: 0":"index"}).set_crs(EPSG_LON_LAT)#.rename(columns ={" Unnamed: 0",'index'})#generate initial facility placement
    print(f"population size: {gdf_pop.size}")
    full_pop_size = gdf_pop.shape[0]

    #load population size
    if (pop_size == -1) or (pop_size > full_pop_size):
        print("setting population size to full:{}".format(full_pop_size))
        pop_size = full_pop_size
    gdf_pop = gdf_pop.sample(pop_size)#cut the sample size down
    print("loaded population points containing {} points".format(pop_size))

    #load state boundaries
    if move_type == 'random':
        bounds = load_state_bounds(COUNTY_SHP_FILE)
    elif move_type == 'pres_weighted':
        bounds = load_county_bounds(COUNTY_SHP_FILE)
        weights = pres_vote_weight(min_portion=min_portion)

    if legal_states_only:
        df_state_legal = bounds[~df_state['STATE_NAME'].isin(BANNED_STATES)]
    else:
        df_state_legal = bounds
    #pull out polygon
    # us_border = df_state.dissolve().geometry.values[0]#extract shapley polygons from dataframe
    
    #cut down to subset of states with legalized facilites
    us_legal_border = df_state_legal.dissolve().geometry.values[0]#extract shapley polygon from dataframe
    if move_type == 'pres_weighted':
        fac_placements_df = random_points_weighted(df_state_legal, n_facilities, weights).reset_index()
    else:
        fac_placements_df = random_points_within_polygon(us_legal_border,n_facilities).reset_index()#

    print(fac_placements_df)
    #calculate the distance to the nearest facility
    fac_pop_dist_df = calc_facility_distance(gdf_pop,fac_placements_df)
    objective_function_val = objective_function(fac_pop_dist_df,'distance','nearest_fac', beta)
    
    for i in range(num_steps):#for each timestep  
        print("step: {}".format(i))
        #generate a test facility placement by moving an agent
        if move_type == 'random':
            test_fac_placements_df = move_agents(fac_placements_df,us_legal_border,1)
        elif move_type == 'pres_weighted':
            test_fac_placements_df = move_agents_weighted(fac_placements_df,df_state_legal,1, weights)
        #calculate the distances for this new facility placement
        test_fac_pop_dist_df = calc_facility_distance(gdf_pop,test_fac_placements_df)
        #calculate the objective function for this new facility placement 
        test_objective_function_val = objective_function(test_fac_pop_dist_df,'distance','nearest_fac', beta)
        print("objective_function: {}".format(objective_function_val))
    
        #evaluate the agent bahvaior - if it is better than the original, we can do something with that
        if EXTREMUM == "MIN":#if we are minimizing the objective function
            if  test_objective_function_val < objective_function_val:#if the new facility placement has a lower objectve function
                fac_placements_df = test_fac_placements_df#use the new facility placement instead of the old one
                print("Replacing old facility list with score {} with new facility list with score {}".format(objective_function_val,test_objective_function_val))
                objective_function_val = test_objective_function_val
    
        elif EXTREMUM == "MAX":
            if  test_objective_function_val > objective_function_val:#if the new facility placement has a higer objectve function
                fac_placements_df = test_fac_placements_df#use the new facility placement instead of the old one
                print("Replacing old facility list with score {} with new facility list with score {}".format(objective_function_val,test_objective_function_val))
                objective_function_val = test_objective_function_val
    
    if legal_states_only:
        states = "legal"
    else:
        states = "all"
    out = out_path / f"{states}_{num_steps}steps_{move_type}40.parq"
    print(out)
    fac_placements_df.to_parquet(out)
    fac_placements_df.plot()
    plt.savefig(out_path / "fac_placements_df.png")
    print(objective_function_val)
    

if __name__=="__main__":
    __spec__ = None
    freeze_support()
    main(**vars(get_parser().parse_args()))







