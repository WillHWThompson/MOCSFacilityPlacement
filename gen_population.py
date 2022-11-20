import numpy as np 
import pandas as pd 
import geopandas as gpd
import shapely as shp
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from shapely.ops import nearest_points
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

def objective_function(fac_pop_dist_df,value_col,groupby_col)
    
    


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


fac_pop_dist_df = calc_facility_distance(gdf_pop,fac_placements_df)












