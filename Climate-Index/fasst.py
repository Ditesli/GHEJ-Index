
print('[.] > importing packages ...')
import time as ti
t0 = ti.time()
import argparse
from typing import Iterable, Collection, List, Dict, Tuple
from dataclasses import dataclass, field
import itertools
from pathlib import Path
import re as re
import subprocess
import os

import json
import numpy as np
import pandas as pd

from pym import read_mym
from read_fasst_data import read_fasst_data
from calc_exposure import calc_exposure

t1 = ti.time()
print(f"    > packages imported in t = {t1 - t0:1.2f} seconds.")
print("[.] > done.")

# Define version information
__version__ = "0.5.0"

# Define module global values
NO_DATA = -9.99 * 10**4
HOME = Path(__file__).absolute().parent.parent
OUTPUT_FOLDER = HOME.parent / "outputlib"
class FASSTDataError(Exception):
    pass


@dataclass
class TimeLine:
    start: int = 2010
    end: int = 2100
    step: int = 1
    baseyear: int = 2010
    line: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.line = np.arange(self.start, self.end + self.step, self.step)


@dataclass
class DataLibrary:
    project: str
    scenario: str
    type: str
    config_file: Path
    config: dict = field(init=False)

    def __post_init__(self):
        with open(HOME / self.config_file) as config:
            self.config = json.load(config)

    def get_datapath(self, name: str, category: str) -> Path:
        if not (data_source := self.config.get(category)):
            raise FASSTDataError(f"data source {category} not available in {self.config_file}")
        
        folder_tags = {"local-lib": "folder", "1click-lib": "folder.1click"}
        location = folder_tags.get(self.type, "1click-lib")
        path_parts = [self.parse(part) for part in data_source.get(location)]
        path = Path(*path_parts)
        filename: str = data_source["data"].get(name)
        if (path is not None and filename is not None):
            datapath = path / filename
        else:
            raise FASSTDataError(
                f"cannot create filepath from path [{path}] and filename [{name}] for [{category}]"
                )
             
        return datapath
    
    def parse(self, part: str) -> str:
        parameters = {"$project": self.project, "$scenario": self.scenario}
        parsed_part = parameters.get(part)
        return parsed_part if parsed_part is not None else part


@dataclass
class Country:
    """Store country level data."""
    name: str
    gdp: pd.Series
    pop: pd.Series
    historical_emissions: pd.Series


class Region:
    """hoi"""
    def __init__(self, name: str, region_countries: Collection[str], emissions: pd.DataFrame, all_countries: List[Country], time: TimeLine):
        self.name = name
        self.countries = region_countries
        self.number_of_countries = len(region_countries)
        self.emissions = emissions
        self.gdp = self.sum_countries(all_countries, 'gdp')
        self.pop = self.sum_countries(all_countries, 'pop')
        self.time = time

    def sum_countries(self, all_countries: List[Country], attr):
        # > it seems we can improve here by just passing the countries in the region
        quantity = np.array([getattr(country, attr) for country in all_countries
                             if country.name in self.countries])
        return np.sum(quantity, axis=0)

    def emission_intensity(self, year: int) -> pd.Series:
        # > check if an extrapolation outside the [time] range is required
        if year in self.time.line:
            intensity = self.emissions[year] / self.gdp[self.time.line == self.time.end]
        elif year > self.time.end:
            # Assume gdp and emissions to remain constant after `time.end`
            # TODO: check why gdp (and pop) isn't a pd.DataFrame
            intensity = self.emissions[time.end]/self.gdp[self.time.line == self.time.end]
        else:
            raise ValueError(f"Cannot calculate emission intensity for year {year}")
        
        return intensity


def simple_downscaling(region: Region):
    """simple method downscaling. Returns an array of emission levels for all
       countries in a region. Each row of [emissions] represents a separate
       country."""
    emissions = {name: region.emissions/region.number_of_countries for name in region.countries}                    
    return emissions
# ............................................................................ #

def convergence_downscaling(region: Region, countries: List[Country], gases: List[str], time: TimeLine):
    """downscale emissions based on the convergence method of Van Vuuren et al., (2007)."""
    # > for now define here the base year `t_0` and convergence year `t_c`
    t_0 = 2010
    t_c = 2150
    # t_c = 2018
    # > store `emissions` of all `gases` for all `countries_in_region`
    emissions = []
    # TODO: check whether we want to include this in `region`
    countries_in_region = [c for c in countries if c.name in region.countries]
    # The emissions can be expressed as a function of a preliminary `emission_intensity`:
    # > E_{C,t} = E_{C,0}\frac{POP_{C,t}}{POP_{C,0}}
    #             \frac{GDP_{pcC,t}}{GDP_{pcP,0}}\frac{EI_{C,t}}{EI_{C,0}}
    # > where the `emission_intensity` can be written as: EI_{t} = E_{t}/GDP_{t}
    for country in countries_in_region:
        # > compute the known historic emission intensity for the [country]
        historical_emission_intensity = country.historical_emissions[gases]/country.gdp[t_0]
        
        # Calculate the emission intensity growth rate [d_emission_intensity_dt]
        # > EI_{\_grC} = (\frac{EI_{R,CY}}{EI_{C,BY}})^{1/(CY-BY)}
        d_emission_intensity_dt = (region.emission_intensity(t_c)[gases]/historical_emission_intensity) ** (1/(t_c - t_0))
        
        # Calculate the preliminary emission intensity  
        # > EI^*_{C,t} = EI^*_{C,t-1}EI_{\_grC}, which can be rewritten as
        #   EI^*_{C,t} = EI^*_{C,BY} * EI_{\_grC}^{t - BY}
        emission_intensity = pd.concat([historical_emission_intensity] * time.line.size, axis=1)
        emission_intensity = emission_intensity * d_emission_intensity_dt.values[:,None] ** (time.line - t_0)
        emission_intensity.columns = time.line

        # > adding separate DataFrames per country to a list and later concatenating is much faster 
        #   than an assignment to a subset of an `emissions` DataFrame
        country_emissions = emission_intensity.loc[gases,:] * country.gdp
        country_emissions.index = pd.MultiIndex.from_product(
            [[country.name], gases], names=["country", "gas"]
            )
        emissions.append(country_emissions)
    
    # Generate single DataFrame
    emissions = pd.concat(emissions)
    # Calculate the [remaining_emissions] as the difference between the
    # regional emissions and the sum of the country emissions
    # > Diff_{R,t} = E_{R,t} - \sum_{C}E^*_{C,t}
    sum_of_emissions = emissions.groupby(level=1).sum()
    remaining_emissions = region.emissions.loc[gases,:] - sum_of_emissions

    # Calculate the emissions share to allocate the [remaining_emissions] to the countries
    # > E_{\_shC,t} = \frac{E^*_{C,t}}{\sum_{CinR}E^*_{C,t}}
    emissions_share = emissions/sum_of_emissions

    # Update emissions to final emission level by distributing the [remaining_emissions]
    # > E_{C,t} = E^*_{C,t} + E^*_{C,t}Diff_{R}E_{\_shC,t}

    # Trying to see what the results look like if we do not try to add remaining emissions
    emissions = emissions + emissions_share * remaining_emissions
    
    # overwrite emissions to only get CEDS values downscaled, comment immediately once not needed anymore
    # emissions = region.emissions.loc[gases,:]

    return emissions
# ............................................................................ #

def region_downscaling(regions: List[Region], countries, gases, time: TimeLine, downscale):
    """
    downscale data from region to country level, through a specified `downscale`
    method. This downscaling method maps the regional emissions onto the
    countries of that `region`.
    """
    countries_emissions = pd.DataFrame()
    for region in regions:
        countries_emissions = pd.concat(
            [countries_emissions, downscale(region, countries, gases, time)]
            )
        print(f"    > region {region.name} downscaled.")
    return countries_emissions.sort_index(axis=0)
# ............................................................................ #

def map_time(time: TimeLine, raw_time: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
    """Interpolate raw time-dependent data to `TimeLine` time."""
    # > Transform the `raw_data`` input, such that we have a single looping index. This relies on
    #   the fact that the last dimension [raw_data.shape[-1]] is the time dimension.
    # TODO: when updating read_mym also update this function, as it switches dimension order
    target_dimension = (*raw_data.shape[:-1], time.line.shape[0])
    linearized_data = raw_data.reshape(-1, raw_data.shape[-1])
    
    # > interpolate data for all non-time dimensions
    interpolant_dimension = (linearized_data.shape[0], time.line.shape[0])
    data = np.empty(interpolant_dimension)
    for idx in range(linearized_data.shape[0]):
        data[idx,:] = np.interp(time.line, raw_time, linearized_data[idx,:])
    
    data = data.reshape(target_dimension)

    return data
# ............................................................................ #

def retrieve_region_names():
    """
    Retrieve two lists, that contain the [fasst_region_names]  in alphabetical
    order and the [image_region_names] in region number order.
    """
    # > in region number order
    image_region_names = ["CAN","USA","MEX","RCAM","BRA","RSAM","NAF","WAF","EAF",
                        "SAF","WEU","CEU","TUR","UKR","STAN","RUS","ME","INDIA",
                        "KOR","CHN","SEAS","INDO","JAP","OCE","RSAS","RSAF"]
    image_region_names = [region.lower() for region in image_region_names]

    # > in alphabetic order
    fasst_region_names = ["ARG","AUS","AUT","BGR","BLX","BRA","CAN","CHE","CHL",
                        "CHN","COR","EAF","EGY","ESP","FIN","FRA","GBR","GOLF",
                        "GRC","HUN","IDN","ITA","JPN","KAZ","MEME","MEX","MON",
                        "MYS","NDE","NOA","NOR","NZL","PAC","PHL","POL","RCAM",
                        "RCEU","RCZ","RFA","RIS","ROM","RSA","RSAM","RSAS",
                        "RSEA","RUE","RUS","SAF","SWE","THA","TUR","TWN","UKR",
                        "USA","VNM","WAF"]
    fasst_region_names = [region.lower() for region in fasst_region_names]

    return fasst_region_names, image_region_names


def generate_image_regions(country_region_map: pd.DataFrame, region_names: List[str], emissions: pd.DataFrame, all_countries: List[Country], time: TimeLine) -> List[Region]:
    """Generate IMAGE regions, including emissions and countries"""
    regions = []
    for idx, name in enumerate(region_names):
        # TODO: check if this needs to be list[str], using the tolist method
        # the 'region' column contains the corresponding regions
        countries_in_region = country_region_map[country_region_map["region"] == name].index.values
        emissions_in_region = emissions.xs(idx, level="region")
        regions.append(Region(name, countries_in_region, emissions_in_region, all_countries, time))
    
    return regions


def generate_fasst_regions(country_region_map: pd.DataFrame, region_names: List[str], country_emissions: pd.DataFrame, all_countries: List[Country], time: TimeLine) -> List[Region]:
    """Generate FASST regions by upscaling emissions from country to region level."""
    # TODO: Merge this function with the image function?
    regions = []
    for name in region_names:
        countries_in_region = country_region_map[country_region_map["region"] == name].index.values
        emissions_in_region = country_emissions.loc[countries_in_region,:].groupby(level=1).sum()
        regions.append(Region(name, countries_in_region, emissions_in_region, all_countries, time))
    
    return regions
# ............................................................................ #

def generate_countries(country_names: Iterable[str], gdps: pd.DataFrame, pops: pd.DataFrame, emissions: Dict[str, pd.Series]) -> List[Country]:
    """Generate `countries`, including emissions and socioeconomic data."""
    countries = []
    gases = [gas for gas in emissions.keys()]
    for name in country_names:
        # > emissions[gas] is a pd.Series, such that emissions[gas].loc[name] yields an np.ndarray
        all_emissions_by_country = pd.Series({gas: emissions[gas].loc[name] for gas in gases})
        this_country = Country(name, gdps.loc[name], pops.loc[name], all_emissions_by_country)
        countries.append(this_country)

    return countries


def read_socioeconomic_data(scenario_root: str, filepath: Path, time: TimeLine) -> pd.DataFrame:
    """Read socio-economic data from csv file, indexed by iso country code."""
    data = pd.read_csv(filepath)
    select_scenario = data["scenario"].str.lower() == scenario_root
    data = data[select_scenario].set_index("iso").sort_index(axis="index")

    # gdp and pop data columns are formatted as 'XTTTT', e.g. X2010
    time_columns = [column for column in data.columns if column[0] == "X"]
    raw_time = np.array([int(column[1:]) for column in time_columns])
    mapped_data = pd.DataFrame(
        map_time(time, raw_time, data.loc[:, time_columns].values),
        index=data.index, columns=time.line
        ) 
    
    return mapped_data


def read_gdp_pop(scenario_root: str, pop_filepath: Path, gdp_filepath: Path, time: TimeLine):
    """
    Read [gdp] and population [pop] scenario data. 
    """
    # > define very small number [epsilon] for zero comparison
    epsilon = 10**(-7)
    # > read csv files into dataframe
    gdp = read_socioeconomic_data(scenario_root, gdp_filepath, time)
    pop = read_socioeconomic_data(scenario_root, pop_filepath, time)
    
    # > generate error for countries that have zero-valued data arrays
    novalue_countries = [country for country in gdp.index
                         if np.sum(gdp.loc[country]) < epsilon
                         or np.sum(pop.loc[country]) < epsilon]
    
    if novalue_countries:
        raise ValueError(f"Countries with zero gdp and/or pop encountered: {novalue_countries}")
    
    if set(gdp.index) != set(pop.index):
        raise ValueError("The sets of countries for the gdp and population data are not the same.")
    
    gdp = preprocess_gdp_pop(gdp)
    pop = preprocess_gdp_pop(pop)
    
    return gdp, pop


def preprocess_gdp_pop(data: pd.DataFrame) -> pd.DataFrame:
    """contract countries for compatibility with historic_emissions and fasst/image definitions."""
    # > sdn = sdn + ssd; srb = srb + srb (kosovo); isr = isr + pse
    mapping = {"srb (kosovo)": "srb", "ssd": "sdn", "pse": "isr", "gum": "usa", "grl": "dnk"}

    for country in mapping.keys():
        data.loc[mapping[country]] += data.loc[country]
        data = data.drop(country)

    return data


def read_mapping(filepath: Path, sheet_name: str, index: str) -> pd.DataFrame:
    """Read region mapping from excel sheet.""" 
    region_mapping = pd.read_excel(filepath, sheet_name=sheet_name)
    # > cast the array as lower case string, otherwise we might end op with the object dtype
    region_mapping = region_mapping.apply(lambda x: x.astype(str).str.lower()).set_index(index)

    return region_mapping


def read_mappings(data_library: DataLibrary) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read excel files with country-to-region mappings"""
    models = ["fasst","image"]
    filepaths = {model: data_library.get_datapath(model, "mappings") for model in models}

    fasst_region_definition = read_mapping(filepaths["fasst"], "REGION_LEGEND", "ISO 3")
    image_region_definition = read_mapping(filepaths["image"], "regional definition", "ISO Code")

    # > in the `fasst_region_definition` ["mne","srb"] is represented as a single country "scg",
    #   we split them up for consistency. The mapping contains column headers for empty columns.
    fasst_region_definition = fasst_region_definition.iloc[:,0:2].drop(labels='scg', axis=0) 
    additional_rows = pd.DataFrame(
        [['montenegro','rceu'],
         ['serbia','rceu'],
         ['sint maarten','rcam'],
         ['curacao','rcam']],
        index=['mne','srb','sxm','cuw'], columns=fasst_region_definition.columns
        )

    columns = {"fasst": {"FASST REGION": "region"}, "image": {"Native Region Code": "region"}}
    fasst_region_definition = (
        pd.concat([fasst_region_definition, additional_rows])
          .rename(columns=columns["fasst"])
          .sort_index(axis=0)
        )
    # > because the "rue" region is missing, assign he countries ["arm", "aze", "geo"] to "rue".
    # TODO: check this later
    fasst_region_definition.loc[["arm","aze","geo"],"region"] = "rue"

    image_region_definition = (image_region_definition.rename(columns=columns["image"])
        .sort_index(axis=0)
        )

    return fasst_region_definition, image_region_definition


def define_fasst_to_image_mapping():
    """
    Define the region-to-region method fasst to image region mapping. This mapping
    can be checked against the region country mapping for fasst and image.
    """

    mapping = {"can": ["can"], "usa": ["usa"], "mex": ["mex"], "rcam": ["rcam"], 
               "bra": ["bra"], "rsam": ["arg","chl","rsam"], "naf": ["egy","noa"], 
               "waf": ["waf"], "eaf": ["eaf"], "saf": ["rsa"], 
               "weu": ["aut","blx","che","esp","fin","fra","gbr","grc","ita","nor","rfa","swe"],
               "ceu": ["bgr","hun","pol","rceu","rcz","rom"], "tur": ["tur"],
               "ukr": ["ukr"], "stan": ["kaz","ris"], "rus": ["rus","rue"], 
               "me": ["golf","meme"], "india": ["nde"], "kor": ["cor"],
               "chn": ["chn","mon","twn"], "seas": ["mys","phl","rsea","tha","vnm"],
               "indo": ["idn"], "jap": ["jpn"], "oce": ["aus","nzl","pac"],
               "rsas": ["rsas"], "rsaf": ["saf"]}

    return mapping
# ............................................................................ #

def read_historic_emissions(gases: List[str], time: TimeLine, data_library: DataLibrary) -> Dict[str, pd.Series]:
    """
    Reads country level historic [emissions] for all [gases] from the files and
    folder specified in [config]. Aggregates [emissions] to a shared country set
    for all input data. Return `emissions` for countries in `baseyear`.
    """
    # > define very small number `epsilon` for zero comparison
    epsilon = 10**(-7)

    all_emissions: dict[str, pd.DataFrame] = {}
    emissions: dict[str, pd.Series] = {}

    ceds_gases = [gas for gas in gases if not gas == "n2o"]  
    for gas in ceds_gases:
        filepath = data_library.get_datapath(gas, "historic_emissions")
        try:
            all_emissions[gas] = pd.read_csv(filepath)
        except IOError:
            raise FASSTDataError(f"Oh dear, something went wrong trying to read {filepath}")

        time_string = f"X{time.baseyear}"
        # > make a copy to prevent "chained assignment".
        emissions[gas] = all_emissions[gas].set_index("country").loc[:,time_string].copy()
        # > contract countries for compatibility with fasst, image and socio-economic data
        emissions[gas].loc["srb"] = emissions[gas].loc["srb"] + emissions[gas].loc["srb (kosovo)"]
        emissions[gas].loc["sdn"] = emissions[gas].loc["sdn"] + emissions[gas].loc["ssd"]
        emissions[gas].loc["dnk"] = emissions[gas].loc["dnk"] + emissions[gas].loc["grl"]
        # > drop contracted countries
        drop_list = ["srb (kosovo)","ssd","gum","grl"]  
        emissions[gas] = emissions[gas].drop(drop_list)
        # > generate warning for countries that have (near) zero-valued data arrays
        country_names = emissions[gas].index
        novalue_countries = [name for name in country_names if emissions[gas].loc[name] < epsilon]
        if novalue_countries:
            print(f"    > zero emissions of [{gas}] encountered in countries: {novalue_countries}")
    
    # TODO: for now just set the historic "n2o" emissions to `NO_DATA`, later read N2O emissions?
    n2o_index = emissions[ceds_gases[0]].index
    emissions["n2o"] = pd.Series(np.full(n2o_index.shape, NO_DATA), index=n2o_index)
    return emissions


def gas_emission_units() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gas emisison units and conversion."""
    # Gases
    gases = ["bc","ch4","co","co2","n2o","nh3","nox","oc","so2","voc"]
    
    # Units
    energy_units = {"bc": "Gg BC/yr", "ch4": "Tg CH4/yr","co": "Tg C/yr",
                    "co2": "Pg C/yr","n2o": "Tg N/yr","nh3": "","nox": "Tg N/yr",
                    "oc": "Gg OC/yr","so2": "Tg S/yr","voc": "Tg NMVOC/yr"}
    land_units = {"bc": "Tg BC/yr", "ch4": "Tg CH4/yr","co": "Tg C/yr",
                  "co2": "Pg C/yr","n2o": "Tg N/yr","nh3": "Tg N/yr",
                  "nox": "Tg N/yr", "oc": "Tg OC/yr","so2": "Tg S/yr",
                  "voc": "Tg NMVOC/yr"}
    image_units = {"bc": "Tg BC/yr", "ch4": "Tg CH4/yr","co": "Tg CO/yr",
                   "co2": "Tg CO2/yr","n2o": "Tg N2O/yr","nh3": "Tg NH3/yr",
                   "nox": "Tg NO2/yr", "oc": "Tg OC/yr","so2": "Tg SO2/yr",
                   "voc": "Tg NMVOC/yr"}

    # Unit converversion
    # > the conversion variables specify the conversion from the energy/industry
    #   and land sector to a set of general IMAGE units
    # > for the conversion from ["oc"] to ["pom"] (particulate organic matter) a 
    #   conversion factor is used of 1.3 (Van Dingenen et al., 2018)
    # Are the co2 conversions making sense? No, from Pg to Tg should be 10**3
    energy_conversion = {"bc": 10**(-3), "ch4": 1.0,"co": 28.0/12.0,"co2": 44.0/12.0 * 10**(3),
                         "n2o": 44.0/14.0,"nh3": 1.0,"nox": 46.0/14.0,"oc": 1.3 * 10**(-3),
                         "so2": 32.0/16.0,"voc": 1.0}
    land_conversion = {"bc": 1.0, "ch4": 1.0,"co": 28.0/12.0,"co2": (44.0/12.0) * 10**(3),
                       "n2o": 44.0/14.0,"nh3": 17.0/14.0,"nox": 46.0/14.0,"oc": 1.3,
                       "so2": 32.0/16.0,"voc": 1.0}
    Tg_to_kg = {gas: 10**9 for gas in gases}

    # Organize in DataFrames
    scope_index = pd.Index(["energy","industry","land","aviation","image"], name="scope")
    units = pd.DataFrame(
        [energy_units, energy_units, land_units, energy_units, image_units],
        index=scope_index
        )
    conversion = pd.DataFrame(
        [energy_conversion, energy_conversion, land_conversion, energy_conversion, Tg_to_kg],
        index=scope_index
        )

    conversion.columns.names = ["gases"]
    units.columns.names = ["gases"]
    
    return units, conversion


def read_image_emissions(data_library: DataLibrary, time: TimeLine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read all IMAGE emissions and map data to TimeLine. Return total emissions per species."""
    gases_by_sector = {
        "energy": ["bc","ch4","co","co2","n2o","nox","oc","so2","voc"],
        "industry": ["ch4","co","co2","n2o","nox","so2","voc"],
        "land": ["bc","ch4","co","co2","n2o","nh3","nox","oc","so2","voc"],
        "aviation": ["bc","nox","voc"],
        "shipping": ["bc", "co", "nox", "oc", "so2", "voc"]
    }

    # `sectors` excludes shipping, as we currently do not read separate data for that 
    sectors = ["energy","industry","land", "aviation"]
    bunker_sectors = ["aviation", "shipping"]
    NREGIONS = 26
    
    region_indices = [i for i in range(NREGIONS)]
    # world_regions = {"energy": 27, "industry": 27, "land": 26, "bunkers": None}
    # region_idx = {
    #     sector: region_indices + [world_index] if world_index else region_indices
    #     for sector, world_index in world_regions.items()
    #     }
    # TODO: for now do not add world region, see if that gets us into trouble elsewhere
    slices = {"energy": np.index_exp[region_indices,:,-1,:],
              "industry": np.index_exp[region_indices,-1,:],
              "land": np.index_exp[-1,region_indices,:],
              "aviation": np.index_exp[region_indices,:,:]}

    # Prepare `pd.DataFrame`s
    index_elements = []
    for sector, gases in gases_by_sector.items():
        # TODO: ugly other way of separating bunkers and 'regular' emissions?
        if sector not in bunker_sectors:
            index_elements.extend(list(itertools.product([sector], gases, region_indices)))
    index_names = ["sector", "gas", "region"]
    emissions_multiindex = pd.MultiIndex.from_tuples(index_elements, names=index_names)

    data = pd.DataFrame(np.nan, index=emissions_multiindex, columns=time.line)

    bunkers_index = [(bunker, gas) for bunker in bunker_sectors for gas in gases_by_sector[bunker]]
    bunkers_index = pd.MultiIndex.from_tuples(bunkers_index)

    bunkers = pd.DataFrame(np.nan, index=bunkers_index, columns=time.line)

    # TODO: for now only use conversion data, perhaps add units?
    conversion = gas_emission_units()[1]

    # Read data
    bunkers_idx = 8
    sector_total_idx = 9
    for sector in sectors:
        if sector == "shipping": # if I only want land data downscaled   or sector == 'industry'
            # for now derive shipping from total bunkers instead of reading directly
            continue

        for gas in gases_by_sector[sector]:
            datapath = data_library.get_datapath(gas, sector)
            raw_data = read_mym(datapath)

            # Move first dimension to the last
            # - Pym has a different dimension order compared to the original read_mym.
            raw_data.data = np.moveaxis(raw_data.data, 0, -1)

            mapped_data = map_time(time, raw_data.time, raw_data.data[slices[sector]])
            mapped_data = mapped_data * conversion.loc[sector,gas]
            if sector == "energy":
                if gas in gases_by_sector["aviation"]:
                    bunkers_by_region = mapped_data[:, bunkers_idx, :]
                    mapped_data = mapped_data[:, sector_total_idx, :] - bunkers_by_region
                    bunkers.loc["shipping", gas] = bunkers_by_region.sum(axis=0)
                else:
                    mapped_data = mapped_data[:, sector_total_idx, :]
            elif sector == "aviation":
                bunkers.loc[sector, gas] = mapped_data.sum(axis=(0,1))
                bunkers.loc["shipping", gas] -= bunkers.loc[sector, gas]
            # Used to be elif        
            if sector not in bunker_sectors:
                data.loc[sector,gas,:] = mapped_data
    
    # > return the regional totals per gas, sum over the `sectors` (excluding bunkers)
    region_totals = data.groupby(["gas","region"]).sum()
    # Save the unaggregated and aggregated image data for inspectation
    data.to_csv('../data/IMAGE/data_image_input.csv')
    # aggregated_image_df = region_totals.loc[(['bc', 'oc'], slice(None)), 2010]
    # aggregated_image_df.to_csv('../data/IMAGE/data_aggregated_image_input.csv')
    region_totals.to_csv('../data/IMAGE/data_aggregated_image_input.csv')
    bunkers.to_csv('../data/IMAGE/bunkers_image_input.csv')

    # As a test use fake region_totals data based on fasst_excel 2010 data
    # region_totals = pd.read_csv('../data/IMAGE/FAKE_data_aggregated_image_input.csv', index_col=['gas', 'region'])
    region_totals.columns = region_totals.columns.astype(int)
    return region_totals, bunkers

# ............................................................................ #
def format_output(region_data: List[Region], bunkers: pd.DataFrame, gases: List[str]) -> pd.DataFrame:
    """
    Construct an emissions output DataFrame, such that it contains the required emissions for 
    all gases, for all regions and bunkers, in the required units [kg/yr].
    """
    # Concatenate region and bunker data into single DataFrame. 
    data = []
    for region in region_data:
        region_index = pd.MultiIndex.from_product(
            [[region.name], region.emissions.index], names=["region", region.emissions.index.name]
            )
        region_emissions = region.emissions
        region_emissions.index = region_index
        data.append(region_emissions)
    
    data.append(bunkers)  
    data = pd.concat(data)
    # > add total by summing regional and bunker emissions, while ignoring `NO_DATA` entries
    data[data == NO_DATA] = np.nan
    # Whats going on here? Should probably grouped by gas/level 1
    total = pd.concat([data.groupby(level=1).sum()], keys=["total"], names=["region"]) #data.groupby("gas") 

    data = pd.concat([total, data])

    # > add CO2, N2O and PM2.5 emissions as having `NO_DATA` data here
    # TODO: move elsewhere?
    region_names = data.index.levels[0].tolist()
    all_gas_names = gases + ["co2", "n2o","pm25"]

    all_gas_index = pd.MultiIndex.from_product(
        [region_names, sorted(all_gas_names)], names=["region","gas"]
        )
    data = data.reindex(all_gas_index, fill_value=NO_DATA)
    
    # TODO: perhaps move this conversion elsewhere?
    # > convert to fasst units, excluding `NO_DATA` entries
    image_to_fasst_conversion = gas_emission_units()[1].loc["image",:]
    view = data != NO_DATA
    data[view] = data[view].mul(image_to_fasst_conversion, axis=0, level=1)
    # >> fill pm2.5 values
    pm25_filename = "TOTAL_COUNTRY_BASELINE_CLE_2005_INCL_BB.TXT"
    pm25_filepath = HOME / "data" / "input_fasst_core" / "emissions" / "cntry_emiss_tables"
    pm25 = read_fasst_data(pm25_filepath / pm25_filename).rename_axis("region", axis=0)

    # >> taking max(column, 0.0) to filter out potential `NO_DATA` entries
    # TODO: check how to improve pm2.5 representation
    pm25 = pm25["pm25"] - np.maximum(pm25["bc"], 0.0) - np.maximum(pm25["om"], 0.0)

    # >> rename index and differently named index values to the ones used in [data]
    pm25 = pm25.rename(index={'*total*': "total"})
    pm25 = pd.concat([pm25], keys=["pm25"], names=["gas"]).swaplevel("gas", "region")
    pm25 = pd.concat([pm25] * time.line.size, axis=1, keys=time.line)

    data = data.rename(index={"shipping": "ship", "aviation": "air"}, level=0)

    # > ["bc","oc"] emissions are added to "pm25" and later subtracted again in fasst_core MATLAB.
    # > Other pm25 is currently not added (priv. comm. R. van Dingenen).
    idx = pd.IndexSlice
    # data.to_csv('pm25_data_test.csv')
    bc_oc = (data.xs("bc", level="gas") + data.xs("oc", level="gas"))
    bc_oc['gas'] = 'pm25'
    
    # Setting the 'gas' column as part of the index
    bc_oc.set_index('gas', append=True, inplace=True)
    data.loc[idx[:,"pm25"],:] = bc_oc
    # data.loc[idx[:,"pm25"],:] = (pm25.sort_index(level=0) + data.xs("bc", level="gas") + data.xs("oc", level="gas"))

    # TODO: for now add name to the time axis (columns) here, move elsewhere?
    data.columns.names = ["time"]
    return data
# ............................................................................ #

def map_health_output(fasst_health_file, project: str, scenario: str) -> pd.DataFrame:
    """
    Map the fasst region level health impact output to image region level, 
    using the mapping provided by fasst. 
    """
    # > get fasst to image regions mapping, so not using the country mapping of both models
    fasst_to_image = define_fasst_to_image_mapping()
    # TODO: improve filepath handling?
    filepath = HOME.parent / 'scenlib' / project / scenario / 'FASST_output' / fasst_health_file
    #filepath = HOME.parent / 'scenlib' / project / scenario / 'output' / fasst_health_file

   #change above path to match how my files are structured 
    data = pd.read_csv(filepath, delimiter=r"\s+", skiprows=1)
    
    # > make sure to use lower case by default
    data.columns = data.columns.str.lower()
    data = data.set_index(["fasst_region"])
    data.index = data.index.str.lower()

    # > generate `image_data` by adding the fasst regions that make up the image regions
    years = list(set(data["year"]))
    image_index = pd.MultiIndex.from_product(
        [fasst_to_image.keys(), years], names=["regions","years"]
        )
    image_data = pd.DataFrame(index=image_index, columns=data.columns.drop("year"))
    image_data["scenario"] = scenario

    # > make sure only the selected columns and rows are added
    data_cols = data.columns.drop(["year","scenario"])
    data_popw_cols = ['popw_so4', 
                      'popw_no3', 
                      'popw_nh4', 
                      'popw_bc', 
                      'popw_pom', 
                      'popw_pm2.5', 
                      'popw_dust', 
                      'popw_ss',
                      'popw_pmtot', 
                      'popw_m6m',]
    data_population_cols = ['pop_cnt',
                            'bu_pm_mort_med', 
                            'bu_pm_mort_lo',
                            'bu_pm_mort_up', 
                            'an_pm_mort', 
                            'o3_mort_med', 
                            'o3_mort_lo',
                            'o3_mort_up']
    for year in years:
        yearly_data = data[data["year"] == year]
        for region in fasst_to_image:
            # TODO: use groupby?
            # Calculate weighted sum for population weighted data
            yearly_region_data = yearly_data.loc[fasst_to_image[region], data_cols]
            image_data.loc[(region, year), data_popw_cols] = (
                yearly_region_data[data_popw_cols].mul(yearly_region_data['pop_cnt'], axis=0).sum(axis=0) 
                / yearly_region_data['pop_cnt'].sum()
            )
            # Sum the population and mortalities data
            image_data.loc[(region, year), data_population_cols] = yearly_region_data.sum(axis=0)

    return image_data
# ............................................................................ #

def print_health_output(data, project, scenario):
    """
    Output image regional health data to a text file.
    """
    filepath = OUTPUT_FOLDER / project / scenario / "image_health_impacts.dat"
    data.to_csv(filepath)

# ............................................................................ #
def print_output(data, project: str, scenario: str, output_year) -> str:
    """
    Print [data] to [output_file]. The formatting is such that it can be read
    by the main fasst function. Returns the name of the [output_file] for
    further use.
    """
    # > use [output_year] to slice [data] columns to enable printing a matrix,
    #   with "region" rows and "gases" columns for a single [output_year]
    data = data.unstack(level="gas")
    # TODO: reorder columns, because fasst core is still dependent on the exact order of gases.
    data = data[output_year]
    data = data.reindex(["bc","ch4","co2","co","n2o","nh3","nox","oc","so2","voc","pm25"], axis=1)
    # > output float emission values in exp notation ("%.2e")
    # TODO: find out why new python formatting, e.g. "{:.2e}" does not work -> use callable?
    scenario_output_folder = OUTPUT_FOLDER / project / scenario
    output_filename = f"emissions_{output_year}.txt"
    data.to_csv(scenario_output_folder / output_filename, sep="\t", float_format="%.2e")

    return output_filename
# ............................................................................ #

def run_fasst_core(project, scenario, year, emissions_file):
    """This runs the fasst core with the generated emissions file."""
    # Currently this just runs the MATLAB executable
    emissions_folder = OUTPUT_FOLDER / project / scenario
    netcdf_file = f"fasst_concentrations_{year}.nc"
    fasst_path = HOME / "bin"
    fasst = fasst_path / "fasst.exe"
    outcome = subprocess.run([fasst, netcdf_file, emissions_folder, emissions_file],
                             cwd=fasst_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(outcome.stdout.decode())
    if not outcome.stderr.decode():
        print('    > successfully calculated fasst concentration fields ...')
    else:
        print('    > an error occurred while calculating the fasst concentration fields:')
        print(outcome.stderr.decode())
# ............................................................................ #

def run_fasst_health(project: str, scenario: str, years: List[int], h2o_flag: int):
    """
    This runs the fasst health module with the generated concentrations file.
    """
    # Currently this just runs the MATLAB executable
    # TODO: check input for matlab: output_file, scenario, project, years
    netcdf_file = "fasst_health_impacts.nc"
    fasst_path = HOME / "bin"
    fasst = fasst_path / "health_impacts.exe"
    years_string = ["["] + [str(year) for year in years] + ["]"]
    years_string = " ".join(years_string)
    outcome = subprocess.run([fasst, netcdf_file, project, scenario, years_string, str(h2o_flag)],
                             cwd=fasst_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(outcome.stdout.decode())
    if not outcome.stderr.decode():
        print('    > successfully calculated fasst health impact data ...')
    else:
        print('    > an error occurred while calculating fasst health impacts:')
        print(outcome.stderr.decode())
# ............................................................................ #

def run_fasst(time: TimeLine, project: str, scenario_root: str, scenario: str, data_library: DataLibrary):
    """
    Run fasst based on IMAGE [scenario] output.

    This function prepares the IMAGE framework data for use in the FASST model
    The following steps are taken in this data reduction process:
    1. read in excel data files that contain (i) the IMAGE country-region mapping,
       and (ii) the FASST country region mapping.
    2. Compare FASST/IMAGE country-region mappings to ensure a correct mapping 
       of countries from the one region set to the other.
    3. Enable down- and upscaling of IMAGE regional emission data.
    4. Perform simple downscaling based on country population and GDP levels.
    5. Generate FASST regional emission output in corresponding data format.
    6. Run the FASST core module to generate concentration output.
    """
# ............................................................................ #

    print('[0] > Welcome to FASST, initializing ...')
    # > 'fasst_settings.json' among others contain the years for which output is generated
    with open(HOME / "fasst_settings.json") as config:
        settings = json.load(config)
        output_years = settings["years"]
        pm25_h2o_flag = settings["pm_h2o_flag"]

    # Create the output directory
    output_path = OUTPUT_FOLDER / project / scenario
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('[0] > Output directory created:',output_path)
    print('[0] > done.')

    # Read regional emissions
    print('[1] > Reading regional and bunker emissions ...')
    emissions, bunkers = read_image_emissions(data_library, time)
    print('[1] > done.')
    
    # Read country gdp and population data
    gdp_filepath = data_library.get_datapath("gdp", "socio_economic")
    pop_filepath = data_library.get_datapath("pop", "socio_economic")
    
    countries_gdp, countries_pop = read_gdp_pop(scenario_root, pop_filepath, gdp_filepath, time)
    
    # Read country historic emissions
    historic_gases = ["bc","ch4","co","co2","n2o","nh3","nox","oc","so2","voc"]
    historic_emissions = read_historic_emissions(historic_gases, time, data_library)
    print('[2] > done.')

    # Read region-country mappings
    print('[2] > Reading mappings, population, gdp and historic emissions ...')
    fasst_region_definition, image_region_definition = read_mappings(data_library)

    # Regenerate image and fasst mapping, accounting for data availability
    # > generate set of countries that do not have (all) data available [lcd]
    print('[3] > Checking compatibility of datasets ...')
    # > extract country names from the gdp data for now as follows   
    country_names = countries_gdp.index
    image_countries = set(image_region_definition.index)
    fasst_countries = set(fasst_region_definition.index)
    historic_emissions_countries = set(historic_emissions["ch4"].index)
    countries_sets = [image_countries, fasst_countries, historic_emissions_countries, country_names]
    # least common denominator
    lcd = set.difference(set.union(*countries_sets), set.intersection(*countries_sets))
    
    # > exclude countries that lack socio-economic and/or emission data
    drop_fasst = [c for c in lcd if c in fasst_countries]
    drop_image = [c for c in lcd if c in image_countries]
    fasst_region_definition = fasst_region_definition.drop(drop_fasst)
    image_region_definition = image_region_definition.drop(drop_image)
    print('[3] > done.')

    # First generate [Country]s, and then generate [Region]s, as the regions can be dependent on
    # the regions, e.g. region gdp and pop as the sum of the countries
    print('[4] > Generating regions and countries ...')
    countries = generate_countries(country_names, countries_gdp, countries_pop, historic_emissions)

    fasst_region_names, image_region_names = retrieve_region_names()
    image_regions = generate_image_regions(
        image_region_definition, image_region_names, emissions, countries, time
        )
    print('[4] > done.')

    # Downscale emissions
    print('[5] > Downscaling emissions to country level and upscaling to FASST region level ...')
    future_gases = ["bc","ch4","co","nh3","nox","oc","so2","voc"]
    # Why does future gases not contain co2 and n2o?
    test_land_gases = ["bc","ch4","co","co2","n2o","nh3","nox","oc","so2","voc"]
    country_emissions = region_downscaling(
        image_regions, countries, future_gases, time, convergence_downscaling
        )
    fasst_regions = generate_fasst_regions(
        fasst_region_definition, fasst_region_names, country_emissions, countries, time
        )
    print('[5] > done.')

    # Generate output and run fasst core and fasst health
    print('[6] > Formatting and printing output ...')
    output = format_output(fasst_regions, bunkers, future_gases)

    for year in output_years:
        output_file = print_output(output, project, scenario, year)
        print(f"[6] > {year} done.")
        print("[7] > Running fasst concentration calculations and storing netcdf output ...")
        run_fasst_core(project, scenario, year, output_file)
        print(f"[7] > {year} done.")

    print('[7.1] > Calculate population exposed')
    calc_exposure(HOME, OUTPUT_FOLDER, project, scenario, output_years)

    print("[8] > Running fasst health impact calculations and storing netcdf and csv output ...")   
    run_fasst_health(project, scenario, output_years, pm25_h2o_flag)
    image_health_data = map_health_output("fasst_health_impacts.dat", project, scenario)
    print_health_output(image_health_data, project, scenario)
    print("[8] > done.")

# ............................................................................ #
def parse_arguments():
    """Process input arguments"""
    ssps = ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]
    parser = argparse.ArgumentParser(
        description="Run the FASST air pollution and health modules.",
        epilog="You can run the default scenario by calling fasst with: ssps ssp2 ssp2_450."
        )

    parser.add_argument("project", help="name of the project folder")
    parser.add_argument(
        "scenario_root", choices=ssps, type=str.lower, help="ssp base scenario"
        )
    parser.add_argument("scenario", help="full scenario name")
    parser.add_argument("input_config", type=Path, default=Path("fasst_input.json"), nargs="?")
    parser.add_argument(
        "--use-scenario-lib", choices=["local-lib", "1click-lib", "cobra-lib"],
        default="1click-lib", type=str.lower, help="define the scenario-lib you are using"
        )
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # > set time line and retrieve passed arguments, initialize `data_library` for I/O
    time = TimeLine()
    args = parse_arguments()
    data_library = DataLibrary(
        args.project, args.scenario, args.use_scenario_lib, args.input_config
        )

    # > run the main fasst model
    run_fasst(time, args.project, args.scenario_root, args.scenario, data_library)