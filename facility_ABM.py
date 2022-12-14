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
import swifter
from sklearn.neighbors import BallTree, KDTree
from matplotlib import pyplot as plt

BANNED_STATES = ['Arkansas','Alabama','Idaho','Kentucky','Louisiana','Kentucky','Mississippi','Missouri','Oklahoma','South Dakota','Tenesee','Texas','West Virginia','Wisconsin']
BANNED_STATES_UPPER = [x.upper() for x in BANNED_STATES]

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

def load_county_bounds(COUNTY_SHP_FILE,EPSG=4326, legal_only=False):
    df_county = gpd.read_file(COUNTY_SHP_FILE)
    df_county = df_county.to_crs(f"EPSG:{EPSG}")
    df_county = df_county[~df_county['STATE_NAME'].isin(['Alaska','Hawaii'])]#remove Alaska and Hawai
    if legal_only:
        df_county = df_county[~df_county['STATE_NAME'].isin(BANNED_STATES)]
    df_county = df_county.dissolve(by = "FIPS")['geometry'].reset_index()#[['STATE_NAME','geometry']]#aggregate by stateI
    df_county['FIPS'] = pd.to_numeric(df_county['FIPS'])
    return df_county


"""
random_points_within_polygon(); returns a list of n points within a given polygon
    inputs: 
        polygon<Shapley.geometry.polygon>: a polygon from which to generate points inaside of 
        number<int>: the number of points to generate
   Output:
        points<list{Shapley.geometry.Points}: a list of points randomly distributed within the polygon
"""
def random_points_within_polygon(polygon,number,EPSG=4326):
    points = []
    #generate points within the polygons bounding box
    minx,miny,maxx,maxy = polygon.bounds
    while len(points) <  number:
        pnt = Point(np.random.uniform(minx,maxx),np.random.uniform(miny,maxy))
        #check if points fall within a polygon
        if polygon.contains(pnt):
            points.append(pnt)
    gdf =  gpd.GeoDataFrame(geometry=points).set_crs(EPSG)
    return gdf

def random_points_weighted(border_df,number,weights_df,EPSG=4326):
    points = []
    while len(points) <  number:
        county_fips = np.random.choice(weights_df.index, p=weights_df.weight)
        bdf = border_df[border_df.FIPS == county_fips]
        if len(bdf) == 0:
            print(f"MISSING county: {county_fips}")
            continue
        border = bdf.dissolve().geometry.values[0]
        #get new trial facility locations
        minx,miny,maxx,maxy = border.bounds
        while True:
            pnt = Point(np.random.uniform(minx,maxx),np.random.uniform(miny,maxy))
            #check if points fall within a polygon
            if border.contains(pnt):
                points.append(pnt)
                break
        
    gdf =  gpd.GeoDataFrame(geometry=points).set_crs(EPSG)
    return gdf

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
add a lot and lon column to a GeoDataFrame with a geometry row with Shapley Point objects
input: 
    df<GeoDataFrame>: a datafarme with a column with Point objects
output:<numpy array> the x and y coordinates of those points
"""
def get_coords_from_shape(df,geom_row="geometry"):
    return np.stack(df[geom_row].apply(lambda coord:[coord.x,coord.y]).tolist(),axis = 0)

# """
# calc_facility_distance(): takes in a df of population and facility placements, returns a df the nearest facility to each individual and the distance
# inpukt:
#         df1<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each individual in the population
#         df2<GeoDataFrame>: a datafame with Shapley.geometry.Point objects for each facility
#         geom1_col<str>: the column in <df1> that contains the point objects
#         geom2_col<str>: the column in <df2> that contains the point objects
# returns: 
#         concat_df<GeoDataFrame>: a df with the position of each indiviudal, its nearest facility and the distance
# """
# def calc_facility_distance(df1,df2,geom1_col='geometry',geom2_col="geometry"):
#     #calculate the nearest facilty to each member of the population
     
#     # start_time = time.time()
#     # nearest_fac_series = df1.apply(nearest,df2=df2,geom1_col=geom1_col,axis = 1)
#     # time1 = time.time()-start_time
    
#     start_time = time.time()
#     nearest_fac_series = paralellize_on_rows(df1,nearest,df2=df2,geom1_col=geom1_col)
#     time2 = time.time()-start_time

#     start_time = time.time()
#     nearest_fac_series = df1.swifter.apply(nearest,df2=df2,geom1_col=geom1_col,axis = 1)
#     time3 = time.time()-start_time

#     print("time1:{},time2:{},time3:{}".format(time1,time2,time3))

#     #reformat nearest facility data into a df
#     nearest_fac_df = nearest_fac_series.reset_index().rename(columns = {0:"nearest_fac"}).set_geometry("nearest_fac")
#     #calculate the distance between each individual and the nearest facility
#     distance_df=df1['geometry'].distance(nearest_fac_df).reset_index().rename(columns = {0:"distance"})
#     #join all data and return
#     concat_df = pd.concat([df1,nearest_fac_df,distance_df],axis = 1).drop("index",axis = 1)
#     return concat_df



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

def calc_facility_distance(df1,df2,DISTANCE_EPSG = 3857):

    #convert coordinates to a flat projection for mercator
    df1 = df1.to_crs(DISTANCE_EPSG)
    df2 = df2.to_crs(DISTANCE_EPSG)
     
    lat_lon_arr = np.stack(get_coords_from_shape(df2),axis = 0)#generate lat long pairs from Point objects
    x= lat_lon_arr[:,0]
    y = lat_lon_arr[:,1]
    tree = KDTree(np.c_[x,y])#create search tree
    #get population coordinates
    pop_coords = np.stack(get_coords_from_shape(df1),axis=0)

    dd,ii = tree.query(pop_coords,k=1)#find the nearest facilty 

    #merged nerest neighbor info with facility info
    df1['distance'] = dd
    df1['nearest_fac'] = ii
    df_fac_pop_merged = df1.merge(df2,left_on = 'nearest_fac',right_on = "index",suffixes=('_pop','_fac'))
    df_fac_pop_merged.drop(['index_pop','index_fac'],axis =1,inplace=True)
    df_fac_pop_merged.rename(columns={'geometry_pop':'geometry','geometry_fac':'nearest_fac'})
    return df_fac_pop_merged


"""
calculate the objective function(dist^beta) for each facility. return the result as a dataframe
"""
def objective_function(df,value_col,groupby_col,beta=1):
    df['weighted_dist'] = df[value_col]**beta#take the distance to an exponential
    #df = df.groupby(df[groupby_col].to_wkt()).agg(total_fac_weighted_dist = ('weighted_dist',sum))
    df = df.groupby(df[groupby_col]).agg(total_fac_weighted_dist = ('weighted_dist',sum))
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
    

def pres_vote_weight(min_portion=.4, legal_only=False):
    pres = pd.read_csv('./data/countypres-2020.csv')
    pres = pres.groupby(['county_fips', 'candidate'], as_index=False).agg(state=('state','max'), state_po=('state_po','max'),
                                                          county_name=('county_name', 'max'), party=('party', 'max'),
                                                          candidatevotes=('candidatevotes','sum'), totalvotes=('totalvotes', 'max')
                                                         )
    # Correcting for missing FIPS for D.C. (https://dc.postcodebase.com/county/11001)
    pres.loc[pres['state_po']=='DC', 'county_fips'] = 11001
    # The below FIPS is not valid for Missouri. Kansas City, MO, is mostly in Jackson County
    # Further investigation is needed to determine what the below data point is
    pres = pres[pres.county_fips != 36000]
    # Filtering out a Rhode Island federal precinct which is missing an FIPS
    pres = pres[~pres['county_fips'].isna()]
    aia = pd.read_csv('./data/bas21_codes_aia.csv')
    aia['ex_FIPS'] = aia['State FIPS'].astype('str') + aia['County FIPS'].astype('str')
    aia['ex_FIPS'] = pd.to_numeric(aia['ex_FIPS'])
    aia.set_index(['ex_FIPS'], inplace=True)
    pres = pres[~pres['state'].isin(['ALASKA','HAWAII'])]

    if legal_only:
        pres = pres[~pres['state'].isin(BANNED_STATES_UPPER)]
    dems = pres[pres.party == 'DEMOCRAT']
    dems['portion'] = dems['candidatevotes'] / dems['totalvotes']
    dems.set_index(['county_fips'], inplace=True)
    
    # Removing Native American counties from weighting dataframe
    dems = dems.join(aia, how='left')
    dems = dems[dems['BAS ID'].isna()]
    print(len(dems))
    dems = dems[dems['portion'] >= min_portion]
    #dems['weight'] = dems['portion'] / sum(dems['portion'])
    dems['weight'] = dems['candidatevotes'] / sum(dems['candidatevotes'])
    print(len(dems))
    #dems = dems[['weight']]
    return dems

# np.random.choice. Pass in list of weights along with list of choices
def move_agents_weighted(my_fac_placements_df,border_df,num_replacements, weights_df):
    county_fips = np.random.choice(weights_df.index, p=weights_df.weight)
    border_df = border_df[border_df.FIPS == county_fips]
    if len(border_df) == 0:
        print(f"county: {county_fips}")
        return my_fac_placements_df
    border = border_df.dissolve().geometry.values[0]
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

def plot_facility_placement(df_pop):
    COUNTY_SHP_FILE = "data/UScounties/UScounties.shp"
    bounds = load_state_bounds(COUNTY_SHP_FILE)

    BACKGROUND_COLOR = '#EEEEEE'
    LINE_COLOR ="#A0A19F"
    DOT_COLOR = "#F08104"
    
    
    fig,ax = plt.subplots(figsize = (10,15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.grid(False)
    
    fig.patch.set_facecolor(BACKGROUND_COLOR)#set black background
    ax.set_facecolor(BACKGROUND_COLOR)#set black foreground
    
        #df_merged.plot(column = df_pop_label,ax = ax,legend=True,cmap ='magma',norm=matplotlib.colors.LogNorm(vmin=1,vmax=pop_max),cax=cax)
        #geo_df_up_to_year.plot(ax = ax,markersize = 0.3,c='white',alpha = 0.4)
    #res = gpd.read_parquet('all_1000steps_placement.parq')
    ax.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
    bounds.plot(ax = ax,facecolor = "none",edgecolor = LINE_COLOR,lw = 0.5)
    df_pop.plot(ax = ax,color = DOT_COLOR,markersize = 1)
    plt.show()