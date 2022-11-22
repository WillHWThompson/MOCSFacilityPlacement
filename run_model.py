import gen_population as pop
import facility_ABM
import pandas as pd
import geopandas as gpd


"""
GLOBAL VARIABLES
"""
COUNTY_SHP_FILE = "data/UScounties/UScounties.shp"
BANNED_STATES = ['Arkansas','Alabama','Idaho','Kentucky','Louisiana','Kentucky','Mississippi','Missouri','Oklahoma','South Dakota','Tenesee','Texas','West Virginia','Wisconsin']
EPSG = 4

""""
init the model for a single facility
"""
#load county data 
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



