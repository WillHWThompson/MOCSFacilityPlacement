def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
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

from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
from copy import deepcopy

from facility_ABM import *
from multiprocessing import Pool,freeze_support
from pathlib import Path

n_facilities = 1070
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
        default=2
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
        "--out_path",
        type=str,
        default=-"./"#if set to -1, the full population will be used
    )
    return parser


def main(
    n_facilities: int = 1070,
    num_steps: int = 4,
    legal_states_only: bool = True,
    out_path: Path = Path("./"),
    pop_size: int = -1 
):
    """
    GLOBAL VARIABLES
    """
    COUNTY_SHP_FILE = "data/UScounties/UScounties.shp"
    BANNED_STATES = ['Arkansas','Alabama','Idaho','Kentucky','Louisiana','Kentucky','Mississippi','Missouri','Oklahoma','South Dakota','Tenesee','Texas','West Virginia','Wisconsin']
    EPSG = 4
    EXTREMUM = "MIN"
    """"
    init the model for a single facility
    """
    print("loading pop point data...")
    df_pop = pd.read_csv("data/simulated_pop_points.csv")
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
    df_state = load_state_bounds(COUNTY_SHP_FILE)
    if legal_states_only:
        df_state_legal = df_state[~df_state['STATE_NAME'].isin(BANNED_STATES)]
    else:
        df_state_legal = df_state
    #pull out polygon
    # us_border = df_state.dissolve().geometry.values[0]#extract shapley polygons from dataframe
    
    #cut down to subset of states with legalized facilites
    us_legal_border = df_state_legal.dissolve().geometry.values[0]#extract shapley polygon from dataframe
    fac_placements_df = random_points_within_polygon(us_legal_border,n_facilities).reset_index()#
    
    #calculate the distance to the nearest facility
    fac_pop_dist_df = calc_facility_distance(gdf_pop,fac_placements_df)
    objective_function_val = objective_function_ind(fac_pop_dist_df,'distance','nearest_fac')

    objective_function_reindex = objective_function_val.reindex(range(n_facilities),fill_value = np.nan)
    
    objective_function_list = []
    for i in range(num_steps):#for each timestep  
        print("step: {}".format(i))
        #generate a test facility placement by moving an agent
        test_fac_placements_df = move_agents(fac_placements_df,us_legal_border,10)
        #calculate the distances for this new facility placement
        test_fac_pop_dist_df = calc_facility_distance(gdf_pop,test_fac_placements_df)
        #calculate the objective function for this new facility placement 
        test_objective_function_val = objective_function_ind(test_fac_pop_dist_df,'distance','nearest_fac')
        test_objective_function_reindex = test_objective_function_val.reindex(range(n_facilities),fill_value = np.nan)



        
        to_move = test_objective_function_reindex > objective_function_reindex
        fac_placements_df.loc[to_move] = test_fac_placements_df[to_move] 
        #evaluate the agent bahvaior - if it is better than the original, we can do something with that
        if EXTREMUM == "MIN":#if we are minimizing the objective function
            to_move = test_objective_function_reindex < objective_function_reindex
            print("moving {} agents".format(to_move.astype(int).sum()))
            fac_placements_df.loc[to_move] = test_fac_placements_df[to_move] 

        elif EXTREMUM == "MAX":
            to_move = test_objective_function_reindex > objective_function_reindex
            print("moving {} agents".format(to_move.astype(int).sum()))
            fac_placements_df.loc[to_move] = test_fac_placements_df[to_move] 

       #record keeping 
        objective_function_val.loc[to_move] = test_objective_function_val.loc[to_move]
        total_objective_function = objective_function_val.sum()
        print("total objective_function: {}".format(total_objective_function))
        objective_function_list.append({"time_step":i,"total_objective_function":total_objective_function})


    
    if legal_states_only:
        states = "legal"
    else:
        states = "all"
    file_name =  f"{states}_{num_steps}steps_placement.parq"
    out_file_name = os.path.join(out_path,file_name)
    fac_placements_df.to_parquet(out_file_name)
    
    
    objective_function_df = pd.DataFrame(objective_function_list)

    df_file_name =  f"{states}_{num_steps}steps_placement.csv"
    df_out_file_name = os.path.join(out_path,df_file_name)
    objective_function_df.to_csv(df_out_file_name)
    
    

if __name__=="__main__":
    __spec__ = None
    freeze_support()
    main(**vars(get_parser().parse_args()))







