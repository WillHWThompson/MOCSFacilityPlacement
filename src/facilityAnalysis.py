import pandas
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import seaborn as sns
import os
import urllib
import shutil
import copy
from shapely.geometry import Point, Polygon
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi

from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import voronoi_regions_from_coords, points_to_coords

from shapely.ops import unary_union
plt.style.use('seaborn')




"""
get_fac_file(): checks if a filename exitst, if it does not downloads a csv of facility placements from a url and saves it as a csv, returns the file as a df
inputs: 
    filename<str>: the filename at which to save the file
    fac_url<str>: the url from which to download the file 
output: 
    df<pandas DataFrame>: a dataframe created from the file
"""
def get_fac_file(filename="../../data/facility_data/business-locations-us.txt",url="https://pdodds.w3.uvm.edu/permanent-share/business-locations-us.txt"):
    if not os.path.exists(filename):#check if you already have the file
        print("you do not have the file {} downloaded.\ndownloading file from {}...\nthis make take a few minutes".format(filename,url))
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("download complete")
    print("opening file")
    df = pd.read_csv(filename,sep='\t')
    return df

"""
make_county_pop_df(): Read in data from a county shape file and a county pop file and merge them to create one file with historical county populations
input: 
    COUNTY_SHP_FILE<str>: the path a a shp file with US county boundaies and FIPS designations for each county 
    COUNTY_POP_FILE<str>: the path to a file with county populations and and FIPS designations
    EPSG<int>: the EPSG number for the cooridnate reference system to use
    AREA_ESPG: the ESPG number to use for area calculation
output: 
    df_merged<GeoDataFrame>: a geodataframe containing the the borders of each county as Shapley Polygons and along with all the counthy information
"""
def make_county_pop_df(COUNTY_SHP_FILE,COUNTY_POP_FILE,EPSG=4326,area_epsg = 6933):

    df_cbsa = gpd.read_file(COUNTY_SHP_FILE)
    df_cbsa = df_cbsa.to_crs(f"EPSG:{EPSG}")
    #df = pd.read_csv('../../data/post_office_data/historical_county_populations.csv')
    df = pd.read_csv(COUNTY_POP_FILE)
    #df_cbsa = gpd.read_file('../../data/post_office_data/UScounties/UScounties.shp')
    #df_cbsa = gpd.read_file('/Users/willthompson/Library/CloudStorage/OneDrive-UniversityofVermont/FacPlaceData/tl_2022_us_county/tl_2022_us_county.shp')
    df_cbsa = df_cbsa.to_crs(EPSG)

    df_cbsa.CNTY_FIPS = df_cbsa.CNTY_FIPS.astype('int64')
    df_cbsa.FIPS = df_cbsa.FIPS.astype('int64')
    df_merged = df_cbsa.merge(df,left_on = 'FIPS',right_on = 'cty_fips')
    df_merged = df_merged[~df_merged['STATE_NAME'].isin(['Alaska','Hawaii'])]#remove Alaska and Hawaii
    df_merged.drop(['cty_fips'],axis = 1)
    df_merged = df_merged.replace(-np.inf,0)
    df_merged['county_area'] = df_merged.geometry.to_crs(area_epsg).area#add the area in m^2 to the dataframe
    df_merged['county_area_km'] = df_merged['county_area']/1e6
    df_merged = df_merged.to_crs(epsg=EPSG)
    #create a single boundary for the whole US
    df_merged['country'] = 'USA'
    df_merged['geometry_county'] = df_merged['geometry']#store geometry in extra column for purposes of sjoin
    #clean dataframe and remove extraneous columns
    df_merged = df_merged.drop(['STATE_FIPS','CNTY_FIPS','cty_fips'],axis = 1)#remove extranous columns
    df_merged = df_merged.rename(columns={"NAME":"county","STATE_NAME":"state"})
    return df_merged


    """ 
   make_boundary(): takes in a geopandas geoDataFrame with and returns the outer boundary of all the geometries in the dataframe. 
   parameters:
    geo_df<GeoDataFrame>: a data frame with a geometry column 
    EPSG<int>: the EPSG designation for the CRS to use
   returns: 
    boundary_shape<shapely MultiPolygon>: the outer shape of the combined geometires in the dataframe
"""
def make_boundary(geo_df,EPSG):
    boundary = geo_df.dissolve(by = 'country')
    boundary = boundary.set_crs(epsg=EPSG)
    #convert us shape into a single polygon for voroni cell boundary
    boundary_shape = unary_union(boundary.geometry)
    return boundary_shape


"""
create_facility_geo_df(): take a df with columns for lat and lon and transform it into a GeoDataFrame with a geometry feature for each row
inputs:
    fac_df<DataFrame>: a pandas DataFrame with latitude and longitude columns
    boundary_shape<Shapely Polygon>: a polygon of the outer boundary to consider. Should be a shapley polygon. 
    lat_col<str>: the column in the df used for lat
    lon_col<str>: the column in the def use for longitude
    crs_to_use<int>: the ESPG CRS to use for the datafame
"""
def create_facility_geo_df(fac_df,boundary_shape,lat_col,lon_col,crs_to_use=4326):
    print("creating facility points from lat lon pairs...")
    #convert lat lon pairs to data farme
    geometry = [Point(x,y) for y,x in zip(fac_df[lat_col],fac_df[lon_col])]

    #specify the CRS of the dataframe 
    crs_to_use = 4326
    #crs_dict =  {"EPSG":"{}".format(crs_to_use)}
    crs_dict =  "EPSG:{}".format(crs_to_use)

    print("initializing GeoDataFrame...")
    geo_df = gpd.GeoDataFrame(copy.deepcopy(fac_df),
                                crs=crs_dict,
                                geometry=geometry).to_crs(crs_to_use)
    #filter points outside of boundary for purposes of making voronoi cells
    print("filtering locations within boundaries")
    #geo_df = geo_df.to_crs(boundary_shape.crs)
    geo_df_mask = geo_df.within(boundary_shape)
    geo_df = geo_df.loc[geo_df_mask]

    return geo_df

"""
calculate_voronoi_tessellation(): takes in a GeoDatafame with facility placements and a geodatafarme with a border and returns a dataframe containing the voronoi tessaltion of each facility within the boundary
input: 
    fac_pos_df_to_voronoi<GeoDataFrame>: a dataframe that contains a row called 'geomtery' filled with Shapley Point objects representing the placement of each facility
    border_df<GeoDataFrame>: a dataframecontaining the borders within which to calculate the voronoi tessellation, should contain a geometry row of Shapley Polygons, the unary union of this row will be treated as the border of the tessellation
    voronoi_espg<int>: the ESPG code designating the coordinate reference system to use for the voronoi tesselation. The default is a mercator projection with ESPG=3395
output: 
    voronoi_df<GeoDataFrame>: a dataframe, each row is one voronoi cell for the tesselation join with the facility information from fac_pos_df_to_voronoi
"""
def calculate_voronoi_tessellation(fac_pos_df_to_voronoi,border_df,voronoi_epsg=3395):
    #convert to CRS for voronoi tess calc(MERCATOR CRS)
    border_df = border_df.to_crs(voronoi_epsg)#convert columns to mercator projection
    fac_pos_df_to_voronoi.to_crs(voronoi_epsg,inplace=True)
    #convert us shape into a single polygon for voroni cell boundary
    print("filtering locations by boundary...")
    boundary_union = unary_union(border_df.to_crs(voronoi_epsg).dissolve().geometry)#get the border to use for tesselation
    fac_pos_df_to_voronoi = fac_pos_df_to_voronoi[fac_pos_df_to_voronoi.within(boundary_union)]#fliter df to make sure all points are in boundaires
    print("calculating voronoi tesselation")
    fac_coords = points_to_coords(fac_pos_df_to_voronoi.geometry)#generate points for voronoi tesselation
    poly_shapes, pts = voronoi_regions_from_coords(fac_coords, boundary_union)#generate voronoi tesselation

    print("calculating voronoi_cell county overlap...")
    #create a dataframe from the voronoi cells
    voronoi_df = gpd.GeoDataFrame([poly_shapes,pts]).T
    voronoi_df.columns = ['geometry','voronoi_cell_index']
    voronoi_df['voronoi_cell_index'] = voronoi_df['voronoi_cell_index'].apply(lambda x:x[0]).astype(int)
    #clean and format the dataframe
    voronoi_df['geometry_voronoi'] = voronoi_df['geometry']
    voronoi_df = voronoi_df.set_crs(voronoi_epsg)

    voronoi_df = gpd.sjoin(fac_pos_df_to_voronoi,voronoi_df)#join voronoi df with original facility df
    voronoi_df = voronoi_df.set_crs(epsg=voronoi_epsg)
    voronoi_df = voronoi_df.drop(voronoi_df.columns[voronoi_df.columns.str.contains('index')],axis =1)#drop extraneous index columns
    voronoi_df = voronoi_df.rename(columns={'geometry':'geometry_location'})#raname the columns for sjoin with boundary_df
    voronoi_df['geometry'] = voronoi_df['geometry_voronoi']
    voronoi_df = voronoi_df.set_geometry('geometry')
    voronoi_df.dropna()
    return voronoi_df


"""
calc_voronoi_cell_population(): given a GeoDataFrame with Voronoi cell information estimate the population in each way. The intersection between the voronoi cell and each of the population regions(counties, censsus blocks census tracts) in intersects is calculated. The population living in this intersection is just the percent of the population region consiting of this intersection multiplied by the population of the region. 
the population of a voronoi cell is calculated as the sum of the population of the intersections.
input: 
    voronoi_df<GeoDataFrame>: a dataframe with a geometry column consisting of Shapley Polygons 
    border_df<GeoDataFrame>: a dataframe with a geometry column consisting of Shappley Polygons which represent population regions. A column of this dataframe must contain the population of each region
    population_col<str>: the name of the column in border_df which contains the population of each regoin 
    cell_index_col<str>: the name of the column in voronoi_df which provides a unique index for each cell, could be a facility id. 

"""
def calc_voronoi_cell_population(voronoi_df,border_df,population_col,cell_index_col,voronoi_epsg=3395,area_epsg=6933):
    print("joining vornoi tesslation to boundary data")
    border_df = border_df.to_crs(voronoi_epsg)
    border_df['boundary_geometry'] = border_df.geometry#create another copy of border geometry for merged df
    voronoi_county_merged = gpd.sjoin(voronoi_df,border_df)#identify which counties each voronoi cell overlaps with

    #estimate the population of each voronoi cell
    voronoi_county_merged_intersection = voronoi_county_merged.boundary_geometry.intersection(voronoi_county_merged)#calculate the intersections between voronoi cells and counties
    intersection_area = voronoi_county_merged_intersection.to_crs(area_epsg).area#get the area of each intersection
    border_area = voronoi_county_merged.set_geometry('boundary_geometry').to_crs(area_epsg).area#calcualte the area contained by each boundary(ie. county)
    intersection_percent = intersection_area/border_area#calculate the intersection as a fraction of the county area

    voronoi_county_merged['intersection_percent'] = intersection_percent
    voronoi_county_merged['intersection_pop'] = voronoi_cell_population = voronoi_county_merged[population_col]*voronoi_county_merged['intersection_percent']#calculate the population of each intersection by multiplying the population of each county by the percent of the county overlapped
    voronoi_cell_population = voronoi_county_merged.groupby(cell_index_col).agg(cell_pop =('intersection_pop','sum'))#calculate the total popuation of each voronoi cell

    return voronoi_cell_population

"""
calculate_voronoi_pop_density_and_fac_density():given a dataframe with facility locations, this use voronoi cells to calculate the cell density and population density
inputs:
    fac_pos_df_to_voronoi<GeoDataFrame>: a dataframe that contains a row called 'geomtery' filled with Shapley Point objects representing the placement of each facility
    border_df<GeoDataFrame>: a dataframecontaining the borders within which to calculate the voronoi tessellation, should contain a geometry row of Shapley Polygons, the unary union of this row will be treated as the border of the tessellation
    voronoi_espg<int>: the ESPG code designating the coordinate reference system to use for the voronoi tesselation. The default is a mercator projection with ESPG=3395
    population_col<str>: the name of the column in border_df which contains the population of each regoin 
    cell_index_col<str>: the name of the column in voronoi_df which provides a unique index for each cell, could be a facility id. 
    voronoi_epsg<int>: the EPSG to use for the voronoi, should be Mercator, 3395
    area_epsg<int>: the calculation used to return the area of each cell, ensure this is an equal area projection like 6933
"""
def calc_voronoi_pop_density_and_fac_density(fac_pos_df_to_voronoi,border_df,cell_index_col,population_col,voronoi_epsg=3395,area_epsg=6933,fac_density_col = 'fac_density',pop_density_col = 'pop_density'):
    #calculate voronoi tesselation from a list of facility densities
    print("calculating voronoi tesselllation")
    voronoi_df = calculate_voronoi_tessellation(fac_pos_df_to_voronoi,border_df,voronoi_epsg)
    print("estimating cell population")
    #estimate the population of each voronoi cell
    voronoi_population_data = calc_voronoi_cell_population(voronoi_df=voronoi_df,border_df=border_df,population_col=population_col,cell_index_col=cell_index_col,voronoi_epsg=voronoi_epsg)
    print("print calculating facility and population densities")
    #join voronoi cell data with population estimates
    voronoi_cell_pop_df = voronoi_df.set_index(cell_index_col).join(voronoi_population_data)
    voronoi_cell_pop_df['cell_area'] = voronoi_cell_pop_df.to_crs(area_epsg).area*1e-6#convert to equal area projection, get area in m^2 convert to km^2 
    #calculate the population and facility densities
    voronoi_cell_pop_df[fac_density_col] = 1/voronoi_cell_pop_df['cell_area']#the facility density is 1/area of the cell
    voronoi_cell_pop_df[pop_density_col] = voronoi_cell_pop_df['cell_pop']/voronoi_cell_pop_df['cell_area']#the popuation density is the cell population/area
    return voronoi_cell_pop_df



"""
Simple method of calculating densities, uses the county borders
the facility density is the number of facilites in a county and the area is the area of a county 
inputs:
    geo_df<GeoDataFrame>: a dataframe with facility locations as Shapley Points
    df_merged<GeoDataFrame>: a dataframe with the borders of counties as Shapley Polygons as well as the population of each county
    groupby_columns<str>: the groupby will use these coulmns to groupby, one should be a column assignign a unique designator to each county(like a FIPS)
    pop_col<str>: the column in the dataframe which stores the population of each county
    count_col<str>: this column is used to count the numeber of facilities, can be any column in the dataframe 
    border_geometry_col<str>: the borders of each county are stored here
    voronoi_epsg<int>: both dataframes are converted to this EPSG for the spatial join
    area_epsg<int>: <df_merged> is converted to this epsg to calculate the area
"""
def count_facs_by_county(geo_df,df_merged,groupby_columns,pop_col,count_col,border_geometry_col= 'geometry',voronoi_epsg=3395,area_epsg=6933):
    #convert to the same CRS
    geo_df  = geo_df.to_crs(voronoi_epsg)
    df_merged = df_merged.to_crs(voronoi_epsg)
    
    #get the area of a of each region 
    df_merged['region_area'] = df_merged['geometry'].to_crs(area_epsg).area

    #merge border and population df with county locations
    facilities_counties_merged = gpd.sjoin(df_merged,geo_df)

    #perform spatial join and aggregate to get counts
    groupby_df = facilities_counties_merged.groupby(groupby_columns).agg(
        pop = (pop_col,'median'),
        fac_count = (count_col,'count'),
        area = ('region_area','median')
    ).reset_index()

    return groupby_df


"""
calc_county_pop_density_and_fac_density():given a dataframe of county information and population and a dataframe of facility locations, perform a spatial join and calculate the facility density and population density using the border of each county
input: 
    geo_df<GeoDataFrame>: a dataframe with facility locations as Shapley Points
    df_merged<GeoDataFrame>: a dataframe with the borders of counties as Shapley Polygons as well as the population of each county
    groupby_columns<str>: the groupby will use these coulmns to groupby, one should be a column assignign a unique designator to each county(like a FIPS)
    pop_col<str>: the column in the dataframe which stores the population of each county
    count_col<str>: this column is used to count the numeber of facilities, can be any column in the dataframe 
    border_geometry_col<str>: the borders of each county are stored here
    voronoi_epsg<int>: both dataframes are converted to this EPSG for the spatial join
    area_epsg<int>: <df_merged> is converted to this epsg to calculate the area
"""
def calc_county_pop_density_and_fac_density(geo_df,df_merged,**kwargs):
    #merge the county info dataframe with the facilyt info, calculate the number of clinics per county
    df_by_county = count_facs_by_county(geo_df=geo_df,
                            df_merged=df_merged,
                            **kwargs
                            )
    #calculate facility density
    df_by_county['area'] = df_by_county['area']*1e-6
    df_by_county['fac_density'] = df_by_county['fac_count']/df_by_county['area']
    df_by_county['pop_density'] = df_by_county['pop']/df_by_county['area']
    return df_by_county

"""
rma_reg(): returns a reduced major axis regression of the input data
inputs:
    pop_density_arr<numpy_array>: an array with data for the x axis of our regression, used for population density data 
    fac_desnity_arr<numpy array>: an array with data for the y axis of our regression, usef for facility density data
returns: 
    reg_dict<dict>
"""
def rma_reg(pop_density_arr,fac_density_arr):
    #log transform data
    pop_density_arr = np.log(pop_density_arr)
    po_density_arr = np.log(fac_density_arr)
    #run a scipy linear regression
    reg = LinearRegression().fit(pop_density_arr.reshape(-1,1),po_density_arr.reshape(-1,1))
    po_density_predicted = reg.predict(pop_density_arr.reshape(-1,1))#predict values from linreg
    reg_score = reg.score(pop_density_arr.reshape(-1,1),po_density_arr.reshape(-1,1))#score our predicition 
    print("Score: {}\tCoeff:{}".format(reg_score,reg.coef_[0][0]))
    #calculate standard deviations for RMA fit
    pop_density_std = np.std(pop_density_arr) 
    po_density_std = np.std(po_density_arr) 
    major_axis_regression_exponent = po_density_std/pop_density_std

    intercept = np.mean(po_density_arr)-major_axis_regression_exponent*np.mean(pop_density_arr)


    reg_dict = {'reg':reg,'pop_density':pop_density_arr,'po_density':po_density_arr,'po_density_predicted':po_density_predicted,'score':reg_score,'coef':reg.coef_[0][0],'intercept':reg.intercept_[0],'major_axis_regression_exponent':major_axis_regression_exponent,'major_axis_regression_intercept':intercept}
    return reg_dict

def make_scaling_plot(my_reg,my_ax,title):
    sns.histplot(x = my_reg.x_emp,y = my_reg.y_emp,ax = my_ax,cmap = 'cividis')

    linear_y_vals = my_reg.x_emp+my_reg.intercept
    two_thirds_y_vals = (2/3)*my_reg.x_emp+my_reg.intercept
    my_ax.plot(my_reg.x_emp,my_reg.predict(my_reg.x_emp),label = 'empirical',c = 'C0')
    my_ax.plot(my_reg.x_emp,my_reg.predict(my_reg.x_emp),label = 'alpha={},R^2={}'.format(round(my_reg.beta,3),round(my_reg.rho**2,3)),c = 'C2')
    my_ax.plot(my_reg.x_emp,linear_y_vals,label = 'alpha=1',c="grey")
    my_ax.plot(my_reg.x_emp,two_thirds_y_vals,label = 'alpha=2/3',c="grey")
    my_ax.set(xlabel = 'Log Population Density',ylabel = "Log Facility Density",title=title)
