def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import gen_population as pop
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

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import shapely as shp
from shapely.geometry import Point

from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
from copy import deepcopy

from facility_ABM import *
from multiprocessing import Pool,freeze_support





def main():
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
    #load county dat    a 
    df_pop = pd.read_csv("data/simulated_pop_points.csv")
    gdf_pop = gpd.GeoDataFrame(df_pop,geometry = gpd.points_from_xy(df_pop.lon,df_pop.lat)).rename(columns = {"Unnamed: 0":"index"})#.rename(columns ={" Unnamed: 0",'index'})#generate initial facility placement
    #load state boundaries
    df_state = pop.load_state_bounds(COUNTY_SHP_FILE)
    df_state_legal = df_state[~df_state['STATE_NAME'].isin(BANNED_STATES)]
    #pull out polygon
    us_border = df_state.dissolve().geometry.values[0]#extract shapley polygons from dataframe
    
    #cut down to subset of states with legalized facilites
    
    
    us_legal_border = df_state.dissolve().geometry.values[0]#extract shapley polygon from dataframe
    fac_placements_df = pop.random_points_within_polygon(us_border,100).reset_index()#
    
    #calculate the distance to the nearest facility
    fac_pop_dist_df = pop.calc_facility_distance(gdf_pop,fac_placements_df)
    objective_function_val = pop.objective_function(fac_pop_dist_df,'distance','nearest_fac')
    
    
    
    num_steps = 1
    
    for i in range(num_steps):#for each timestep    
        print("step: {}".format(i))
        #generate a test facility placement by moving an agent
        test_fac_placements_df = move_agents(fac_placements_df,us_border,1)
        #calculate the distances for this new facility placement
        test_fac_pop_dist_df = calc_facility_distance(gdf_pop,test_fac_placements_df)
        #calculate the objective function for this new facility placement 
        test_objective_function_val = objective_function(test_fac_pop_dist_df,'distance','nearest_fac')
        print("objective_function: {}".format(objective_function_val))
    
        #evaluate the agent bahvaior - if it is better than the original, we can do something with that
        if EXTREMUM == "MIN":#if we are minimizing the objective function
            if  test_objective_function_val < objective_function_val:#if the new facility placement has a lower objectve function
                fac_placements_df = test_fac_placements_df#use the new facility placement instead of the old one
                print("Replacing old facility list with score {} with new facility list with score {}".format(objective_function_val,test_objective_function_val))
    
        elif EXTREMUM == "MAX":
            if  test_objective_function_val > objective_function_val:#if the new facility placement has a higer objectve function
                fac_placements_df = test_fac_placements_df#use the new facility placement instead of the old one
                print("Replacing old facility list with score {} with new facility list with score {}".format(objective_function_val,test_objective_function_val))
    
    


if __name__=="__main__":
    __spec__ = None
    freeze_support()
    main()







