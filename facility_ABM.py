"""
facility_abm.py

file contains all the functions to run the facility placement agent based model
"""
import numpy as np 
import pandas as pd 
import geopandas as gpd
import shapely as shp
import time
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
from copy import deepcopy
from multiprocessing import Pool,freeze_support
from functools import partial



"""

"""
def parallelize(data,func,num_processes=8):

    data_split = np.array_split(data,num_processes)
    pool = Pool(num_processes)

    #data = pd.concat(pool.map(func,data_split))
    data = pd.concat(pool.map(func,data_split))

    pool.close()
    pool.join()
    return data

def run_on_subset(func,df2,geom1_col,data_subset):
    applied_data = data_subset.apply(func,df2 = df2,geom1_col = geom1_col,axis=1)
    return applied_data


def paralellize_on_rows(data,func,df2,geom1_col,num_processes=8):
    return parallelize(data,partial(run_on_subset,func,df2,geom1_col),num_processes)



"""load_state_bounds(): given a shp file of county locations, return a GeoDataFrame with the states as rows
    input:
        COUNTY_SHP_FILE<str>: the path to a shp file containing us county data
        EPSG: the ESPG id of the coordinate reference system to use
"""
def load_state_bounds(COUNTY_SHP_FILE,EPSG=4326):
    df_county = gpd.read_file(COUNTY_SHP_FILE)
    df_county = df_county.to_crs(f"EPSG:{EPSG}")
    df_county = df_county[~df_county['STATE_NAME'].isin(['Alaska','Hawaii'])]#remove Alaska and Hawai
    df_state = df_county.dissolve(by = "STATE_NAME")['geometry'].reset_index()#[['STATE_NAME','geometry']]#aggregate by stateI
    return df_state

"""
random_points_within_polygon(); raturns a list of n points within a given polygon
    inputs: 
        polygon<Shapley.geometry.polygon>: a polygon from which to generate points inaside of 
        number<int>: the number of points to generate
   Output:
        points<list{Shapley.geometry.Points}: a list of points randomly distributed within the polygon
"""
def random_points_within_polygon(polygon,number):
    points = []

    #generate points within the polygons bounding box
    minx,miny,maxx,maxy = polygon.bounds
    while len(points) <  number:
        pnt = Point(np.random.uniform(minx,maxx),np.random.uniform(miny,maxy))
        #check if points fall within a polygon
        if polygon.contains(pnt):
            points.append(pnt)
    return gpd.GeoDataFrame(geometry=points)

"""
nearest():calculate the nearest neighbor between two rows of dataframe
inputs:
    row<GeoPandas.Series>: a row of a GeoDataFrame, each row should contain one individual in the population
    df2<GeoDataFrame>: a seperate dataframe with facility placements in it
"""
def nearest(row,df2,geom1_col = 'geometry'):
    #find the geometry that is closest
     geom_union = df2.unary_union#create a multipoint with all facility placements
     return nearest_points(row[geom1_col],geom_union)[1]#find the nearest facility to each ind in the population

def get_distance(row,geom1_col,geom2_col):
    return row[geom1_col].distance(row[geom2_col])


"""
calc_facility_distance(): takes in a df of population and facility placements, returns a df the nearest facility to each individual and the distance
inpukt:
        df1<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each individual in the population
        df2<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each facility
        geom1_col<str>: the column in <df1> that contains the point objects
        geom2_col<str>: the column in <df2> that contains the point objects
returns: 
        concat_df<GeoDataFrame>: a df with the position of each indiviudal, its nearest facility and the distance
"""
def calc_facility_distance(df1,df2,geom1_col='geometry',geom2_col="geometry"):
    #calculate the nearest facilty to each member of the population
     
    #start_time = time.time()
    #nearest_fac_series = df1.apply(nearest,df2=df2,geom1_col=geom1_col,axis = 1)
    #time1 = time.time()-start_time
    
    start_time = time.time()
    nearest_fac_series = paralellize_on_rows(df1,nearest,df2=df2,geom1_col=geom1_col)
    time2 = time.time()-start_time

    print("time1:{},time2:{}".format(time1,time2))

    #reformat nearest facility data into a df
    nearest_fac_df = nearest_fac_series.reset_index().rename(columns = {0:"nearest_fac"}).set_geometry("nearest_fac")
    #calculate the distance between each individual and the nearest facility
    distance_df=df1['geometry'].distance(nearest_fac_df).reset_index().rename(columns = {0:"distance"})
    #join all data and return
    concat_df = pd.concat([df1,nearest_fac_df,distance_df],axis = 1).drop("index",axis = 1)
    return concat_df

"""
calculate the objective function(dist^beta) for each facility. return the result as a dataframe
"""
def objective_function(df,value_col,groupby_col,beta=1):
    df['weighted_dist'] = df[value_col]**beta#take the distance to an exponential
    df = df.groupby(df[groupby_col].to_wkt()).agg(total_fac_weighted_dist = ('weighted_dist',sum))
    return df['total_fac_weighted_dist'].sum()

"""
move_agents: generate a new facility location df with a random subset of agents moved to new locations
    inputs:
        my_fac_placement<GeoDataFrame>: a geodataframe with the locations of our facilties

        num_replacements<int>: the number of agents to move
    output:
        fac_placement_df_test<GeoDataFrame>: a df with the location of our facilites with a subset moved to new random locations
"""
def move_agents(my_fac_placements_df,border,num_replacements):
    fac_placements_df_test = deepcopy(my_fac_placements_df)#deepcopy the facility placement list
    num_fac = my_fac_placements_df.shape[0]#get the number of facilities
    #get new trial facility locations
    new_points = random_points_within_polygon(border,num_replacements)
    #pick facilities to move
    inds_to_change = np.random.choice(np.arange(num_fac),num_replacements)
    #iterate through the list of possibilites, change as needed
    for i,ind in enumerate(inds_to_change):
        fac_placements_df_test.loc[ind,'geometry'] = new_points.geometry.values[i]
    return fac_placements_df_test
    


