import numpy as np 
import pandas as pd 
import geopandas as gpd
import shapely as shp
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
from copy import deepcopy

"""
GLOBAL VARIABLES
"""
COUNTY_SHP_FILE = "data/UScounties/UScounties.shp"
BANNED_STATES = ['Arkansas','Alabama','Idaho','Kentucky','Louisiana','Kentucky','Mississippi','Missouri','Oklahoma','South Dakota','Tenesee','Texas','West Virginia','Wisconsin']
EPSG = 4


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
    output:
        points<list{Shapley.geometry.Points}: a list of points randomly distributed within the polygon
"""
def random_points_within_polygon(polygon,number):
    points = []
    minx,miny,maxx,maxy = polygon.bounds
    while len(points) <  number:
        pnt = Point(np.random.uniform(minx,maxx),np.random.uniform(miny,maxy))
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
input:
        df1<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each individual in the population
        df2<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each facility
        geom1_col<str>: the column in <df1> that contains the point objects
        geom2_col<str>: the column in <df2> that contains the point objects
returns: 
        concat_df<GeoDataFrame>: a df with the position of each indiviudal, its nearest facility and the distance
"""
def calc_facility_distance(df1,df2,geom1_col='geometry',geom2_col="geometry"):
    #calculate the nearest facilty to each member of the population
    nearest_fac_series = gdf_pop.apply(nearest,df2=df2,geom1_col=geom1_col,axis = 1)
    #reformat nearest facility data into a df
    nearest_fac_df = nearest_fac_series.reset_index().rename(columns = {0:"nearest_fac"}).set_geometry("nearest_fac")
    #calculate the distance between each individual and the nearest facility
    distance_df=gdf_pop['geometry'].distance(nearest_fac_df).reset_index().rename(columns = {0:"distance"})
    #join all data and return
    concat_df = pd.concat([gdf_pop,nearest_fac_df,distance_df],axis = 1).drop("index",axis = 1)
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
def move_agents(my_fac_placements_df,num_replacements):
    fac_placements_df_test = deepcopy(fac_placements_df)#deepcopy the facility placement list
    num_fac = fac_placements_df.shape[0]#get the number of facilities
    #get new trial facility locations
    new_points = random_points_within_polygon(us_border,num_replacements)
    #pick facilities to move
    inds_to_change = np.random.choice(np.arange(num_fac),num_replacements)
    #iterate through the list of possibilites, change as needed
    for i,ind in enumerate(inds_to_change):
        print(f"i:f{i}")
        fac_placements_df_test.loc[ind,'geometry'] = new_points.geometry.values[i]
    return fac_placements_df_test
    

""""
init the model for a single facility
"""
#load county data 
df_pop = pd.read_csv("data/simulated_pop_points.csv")
gdf_pop = gpd.GeoDataFrame(df_pop,geometry = gpd.points_from_xy(df_pop.lon,df_pop.lat)).rename(columns = {"Unnamed: 0":"index"})#.rename(columns ={" Unnamed: 0",'index'})#generate initial facility placement
#load state boundaries
df_state = load_state_bounds(COUNTY_SHP_FILE)
df_state_legal = df_state[~df_state['STATE_NAME'].isin(BANNED_STATES)]
#pull out polygon
us_border = df_state.dissolve().geometry.values[0]#extract shapley polygons from dataframe
#cut down to subset of states with legalized facilites
us_legal_border = df_state.dissolve().geometry.values[0]#extract shapley polygon from dataframe
fac_placements_df = random_points_within_polygon(us_border,100).reset_index()#

#calculate the distance to the nearest facility
fac_pop_dist_df = calc_facility_distance(gdf_pop,fac_placements_df)
objective_function_val = objective_function(fac_pop_dist_df,'distance','nearest_fac')



    



    


















