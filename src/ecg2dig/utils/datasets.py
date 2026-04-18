# =================
# Adapted from https://github.com/broadinstitute/ml4h
# =================
import os
import sys
import numpy as np
import h5py
import numcodecs
import torch
import random
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torchvision import transforms
from typing import Callable, List, Union, Optional, Tuple, Dict, Any
import pandas as pd
from ml4ht.data.defines import LoadingOption, SampleID, Tensor
from ml4ht.data.data_description import DataDescription
from ml4h.defines import PARTNERS_DATETIME_FORMAT, ECG_REST_AMP_LEADS
from ml4ht.data.util.date_selector import DATE_OPTION_KEY

def safe_pearsonr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return 0.0, 1.0  # fallback: no correlation
    return pearsonr(x[mask], y[mask])

def is_valid_ecg(ecg: torch.Tensor, abs_threshold: float = 10.0) -> bool:
    """
    Returns False if the ECG tensor contains NaN, Inf, or
    any value whose absolute magnitude exceeds abs_threshold.

    Args:
        ecg:       Tensor of shape [C, T] or [T, C].
        abs_threshold: maximum plausible absolute voltage (in mV).
                       Anything above this is flagged invalid.

    Returns:
        True if all values are finite AND within ±abs_threshold.
    """
    # Check for NaN or Inf
    if not torch.isfinite(ecg).all():
        return False

    return True

def is_bound_ecg(ecg: torch.Tensor, abs_threshold: float = 40.0) -> bool:
    """
    Returns False if the ECG tensor contains a value whose absolute 
    magnitude exceeds abs_threshold.

    Args:
        ecg: Tensor of shape [C, T] or [T, C].
        abs_threshold: maximum 

    Returns:
        True if all values are finite AND within ±abs_threshold.
    """
    
    if ecg.abs().max() > abs_threshold:
        print(f'ecg_max={ecg.abs().max()}', file=sys.stderr)
        return False
    
    return True
# this is used to load data from our hd5 storage
def decompress_data(data_compressed: np.array, dtype: str) -> np.array:
    """Decompresses a compressed byte array. If the primitive type of the data
    to decompress is a string, calls decode using the zstd codec. If the
    primitive type of the data to decompress is not a string (e.g. int or
    float), the buffer is interpreted using the passed dtype."""
    codec = numcodecs.zstd.Zstd()
    data_decompressed = codec.decode(data_compressed)
    if dtype == "str":
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data

class ECGDataDescription(DataDescription):
    """
    -----------------------------------------------------
    Adapted to handle input from both MGH and BWH @eleni
    Before: f'{ROOT_DIR}ecgs/ecg_{hospital.lower()}_hd5s/', #folder with MRN.h5 files
    Now (hospital agnostic): f'{ROOT_DIR}ecgs/'
    Also, new transforms are applied. Before:
    transforms=[millivolt_ecg, standardize_by_sample_ecg]  
    # these will be applied in order
    -----------------------------------------------------
    Original: ml4h ECGDataDescription
    https://github.com/broadinstitute/ml4h/blob/master/ml4h/data_descriptions.py 
    -----------------------------------------------------
    Reads ECGs in our hd5 format stored on ccds s3.
    If s3 information is provided, if an hd5 is not available locally,
    it's downloaded from s3.
    :param local_hd5_folder: Where to check for locally stored hd5s.
        Also where hd5s will be downloaded to from s3.
    :param name: Name of the output of this DataDescription.
    :param ecg_len: length in samples to interpolate all leads of ECG to.
    :param transforms: transformations including augmentations to apply to ECG.
    :param s3_bucket_name: s3 bucket to get hd5s from. E.g. 2017P001650
    :param s3_bucket_path: list of folders in bucket to pull hd5s from.
        E.g. ['ecg_bwh_hd5s', 'ecg_mgh_hd5s']
    :param hd5_path_to_ecg: key in hd5 of the ECG leads
    :param leads: mapping from lead name in hd5 -> channel in output array

    EXAMPLE CALLING
    ecg_dd = ECGDataDescription(
    '../ecgs/ecg_mgh_hd5s/', 
    name=ecg_tmap.input_name(), 
    ecg_len=5000,  # all ECGs will be linearly interpolated to be this length
    transforms=[standardize_by_sample_ecg],  # these will be applied in order
    # data will be automatically localized from s3
    """
    
    S3_PATH_OPTION = "s3_path"
    TEXT_DIAGNOSES = {
        "sb": ["sinus brady"],
        "st": ["sinus tachy"],
        "af": [
            "afib",
            "atrial fib",
            "afibrillation",
            "atrialfibrillation",
        ],
        "rbbb": [
            "right bbb",
            "rbbb",
            "right bundle branch block",
        ],
        "lbbb": [
            "left bbb",
            "lbbb",
            "left bundle branch block",
        ],
        "avb": [
            "1st degree atrioventricular block",
            "1st degree av block",
            "first degree av block",
            "first degree atrioventricular block",
        ],
        "lvh": [
            "biventricular hypertrophy",
            "biventriclar hypertrophy",
            "leftventricular hypertrophy",
            "combined ventricular hypertrophy",
            "left ventricular hypertr",
        ],
        "ischemia": [
            "diffuse st segment elevation",
            "consistent with lateral ischemia",
            "subendocardial ischemia",
            "apical subendocardial ischemia",
            "inferior subendocardial ischemia",
            "anterolateral ischemia",
            "antero-apical ischemia",
            "consider anterior and lateral ischemia",
            "st segment depression",
            "minor st segment depression",
            "st segment depression in leads v4-v6",
            "anterolateral st segment depression",
            "infero- st segment depression",
            "st depression",
            "suggest anterior ischemia",
            "st segment depression is more marked in leads",
            "possible anterior wall ischemia",
            "consistent with ischemia",
            "diffuse scooped st segment depression",
            "anterolateral subendocardial ischemia",
            "diffuse st segment depression",
            "st segment elevation consistent with acute injury",
            "inferior st segment elevation and q waves",
            "st segment depression in anterolateral leads",
            "widespread st segment depression",
            "consider anterior ischemia",
            "suggesting anterior ischemia",
            "consistent with subendocardial ischemia",
            "marked st segment depression in leads",
            "inferior st segment depression",
            "st segment elevation in leads",
            "st segment elevation",
            "st segment depressions more marked",
            "anterior st segment depression",
            "apical st depression",
            "septal ischemia",
            "st segment depression in leads",
            "suggests anterolateral ischemia",
            "st elevation",
            "diffuse elevation of st segments",
            "marked st segment depression",
            "anterior infarct or transmural ischemia",
            "inferoapical st segment depression",
            "lateral ischemia",
            "nonspecific st segment depression",
            "anterior subendocardial ischemia",
        ],
    }
    
    ## For the ECGMaskedEncoder
    META_DATA_FIELDS = [ 
        "atrialrate_md",
        "ventricularrate_md",
        "gender",
        "patientage",
        "sitename",
        "locationname",
        "qrsduration_md",
        "printerval_md",
        "qtinterval_md",
        "poffset_md",
        "qrscount_md",
        "qtcorrected_md",
        "ponset_md",
        "paxis_md",
        "raxis_md",
        "taxis_md",
        "toffset_md",
        "voltagelength",
        # RestingECGMeasurements
        "ventricularrate_md",
        "qonset_md",
        "qoffset_md",
        "poffset_md",
    ]


    def __init__(
        self,
        local_hd5_folder: str,  # '../ecgs/', was '../ecgs/ecg_mgh_hd5s/'
        name: str, # name=ecg_tmap.input_name(), from ecg_tmap deinition 
        ecg_len: int, # ecg_len=5000, all ECGs will be linearly interpolated to be this length
        #transforms: List[Callable[[Tensor, Loadin], Tensor]] = None,  
        transforms: Optional[Callable[[Tensor], Tensor]] = None,
        s3_bucket_name: str = None,
        s3_bucket_path: Union[str, List[str]] = None,
        date_format: str = PARTNERS_DATETIME_FORMAT,
        hd5_path_to_ecg: str = "partners_ecg_rest",
        leads: Dict[str, int] = ECG_REST_AMP_LEADS,
    ):

        self.local_hd5_folder = local_hd5_folder
        self.date_format = date_format
        self.hd5_path_to_ecg = hd5_path_to_ecg
        self._name = name
        self.transforms = transforms or []
        self.ecg_len = ecg_len
        self.leads = leads
        # s3
        self.local_only = s3_bucket_name is None
        self.s3_bucket_name = s3_bucket_name
        if s3_bucket_name is not None and s3_bucket_name is None:
            raise ValueError(
                f"S3 bucket {s3_bucket_name} provied, but no s3_buckets given.",
            )
        self.s3_bucket_paths = self._prep_s3_bucket_paths(s3_bucket_path)

    def _prep_s3_bucket_paths(
        self,
        s3_bucket_paths: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Prepares user input s3 paths for internal use"""
        if s3_bucket_paths is None:
            return
        elif isinstance(s3_bucket_paths, list):
            paths = s3_bucket_paths
        elif isinstance(s3_bucket_paths, str):
            paths = [s3_bucket_paths]
        else:
            raise TypeError(
                f"Cannot use s3_bucket_path input of type {type(s3_bucket_paths)}.",
            )
        for path in paths:
            os.makedirs(self._local_path(path), exist_ok=True)
        return paths

    def _local_path(self, s3_bucket_path: str) -> str:
        """The folder downloaded hd5s from s3_bucket_path end up in"""
        if self.local_only:
            return self.local_hd5_folder
        return os.path.join(self.local_hd5_folder, s3_bucket_path)

    def download_if_s3(self, sample_id, s3_bucket_path: str):
        if self.s3_bucket_name is not None:
            s3_path = os.path.join(s3_bucket_path, f"{sample_id}.hd5")
            download_s3_if_not_exists(
                self.s3_bucket_name,
                s3_path,
                self._local_path(s3_bucket_path),
            )

    def s3_sample_ids_and_s3_paths(self) -> Tuple[int, str]:
        if self.local_only is None:
            raise ValueError("No s3 bucket provided.")
        bucket = get_s3_bucket(self.s3_bucket_name)
        for s3_bucket_path in self.s3_bucket_paths:
            paths = bucket.objects.filter(Prefix=s3_bucket_path).all()
            for path in paths:
                path = path.key
                if not path.endswith(".hd5"):
                    continue
                yield sample_id_from_path(path), s3_bucket_path

    def local_sample_ids(self) -> int:
        yield from map(
            sample_id_from_path,
            glob.glob(os.path.join(self.local_hd5_folder, "*.hd5")),
        )
       
    def _loading_options(
        self,
        MRN_path: str,  # <-- 'ecg_mgh_hd5s/100073'
        local_hd5_folder: str, # <-- '../ecgs/' or ROOT_DIR + 'ecgs/'
    ) -> List[LoadingOption]: 
        """
        Lists all the dates of a person's ECGs (person is in MRN_path)
        """
        
        #hd5_path = os.path.join(local_hd5_folder, f"{sample_id}.hd5")
        hd5_path = os.path.join(local_hd5_folder, f"{MRN_path}.hd5")
        
        # e.g. hd5_path: /data/cvrepo/ecgs/ecg_bwh_hd5s/18473.hd5
        # print(f'Inside _loading options hd5_path: {hd5_path}')
        # dates are in hierarchical structure:
        
        with h5py.File(hd5_path, "r") as hd5:
            dates = list(
                hd5[self.hd5_path_to_ecg],  # hd5_path_to_ecg="partners_ecg_rest",
            )  # list all of the dates of saved ECGs
            sites = [
                decompress_data(
                  data_compressed=hd5[f'{self.hd5_path_to_ecg}/{date}/sitename'][()], 
                    dtype='str'
                )
                for date in dates
            ]
        return [
            {
                'SITE': site,
                DATE_OPTION_KEY: pd.to_datetime(date).to_pydatetime(),
                self.S3_PATH_OPTION: os.path.basename(local_hd5_folder),

            }
            for date, site in zip(dates, sites)
        ]

    def get_loading_options(self, MRN_path) -> List[LoadingOption]:
        '''
        Basically picks up all the ECG dates in a specific MRN
        Effectively it loads all ECGS of a person
        Output
        ------
        a list of dates for ecgs
        '''
        options = []
        if self.local_only:
            return self._loading_options(MRN_path, self.local_hd5_folder)
        
        for s3_bucket_path in self.s3_bucket_paths:
            try:  # try to find sample id in all s3 folders
                self.download_if_s3(MRN_path, s3_bucket_path)
            except ValueError:
                continue
            local_folder = self._local_path(s3_bucket_path)
            options += self._loading_options(MRN_path, local_folder)
            
        return options

    def get_raw_data(self, 
                     MRN_path:str, 
                     loading_option: LoadingOption,
                     ) -> Tensor:
        '''
        Decompresses and returns the ECG Tensor ()
        - Use the date provided in the dataframe inside the 
        loading option to get the ecg waveform from the hd5 file. 
        With the date we navigate to the specific ECG waveform
        as a dictionary {lead: signal}. 
        - Signal is decompressed once retrieved
        - Signal is converted to PyTorch Tensor and standardized
        with mean and std from the training set
        Most of the code from @ml4h
        Args:
        
        Returns:
            ecg Tensor
        
        '''
        s3_path = loading_option[self.S3_PATH_OPTION]
        self.download_if_s3(MRN_path, s3_path)

        hd5_path = os.path.join(self._local_path(s3_path), f"{MRN_path}.hd5")
        
        with h5py.File(hd5_path, "r") as hd5:
            date_str = loading_option[DATE_OPTION_KEY].strftime(
                self.date_format,
            )
            compressed_leads = {
                    lead: hd5[self.hd5_path_to_ecg][date_str][lead] for lead in self.leads
            }
            ecg = np.zeros((self.ecg_len, len(self.leads)), dtype=np.float32)
            for lead in compressed_leads:
                voltages = decompress_data(
                    compressed_leads[lead][()],
                    dtype=compressed_leads[lead].attrs["dtype"],
                )
                if voltages.shape[0] != self.ecg_len:
                    voltages = np.interp(
                        np.linspace(0, 1, self.ecg_len),
                        np.linspace(0, 1, voltages.shape[0]),
                        voltages,
                    )
                ecg[:, self.leads[lead]] = voltages
                
            # Only ToTensor here, rest of the transforms in __getitem__
            # Get the ToTensorTransform
            if self.transforms:
                ecg = self.transforms.transforms[0](ecg)
                #print(f'Inside get_raw_data: type ECG: {ecg.type()}')
                
            return ecg

    def get_ecg(self,
                     MRN_path:str, 
                     loading_option: LoadingOption,
                     ) -> Tensor:
        '''
        Decompresses and returns the ECG Tensor ()
        basically for plotting
        Args:
        
        Returns:
            ecg Tensor
        
        '''
        s3_path = loading_option[self.S3_PATH_OPTION]
        self.download_if_s3(MRN_path, s3_path)

        hd5_path = os.path.join(self._local_path(s3_path), f"{MRN_path}.hd5")
        
        with h5py.File(hd5_path, "r") as hd5:
            date_str = loading_option[DATE_OPTION_KEY].strftime(
                self.date_format,
            )
            compressed_leads = {
                    lead: hd5[self.hd5_path_to_ecg][date_str][lead] for lead in self.leads
            }
            ecg = np.zeros((self.ecg_len, len(self.leads)), dtype=np.float32)
            for lead in compressed_leads:
                voltages = decompress_data(
                    compressed_leads[lead][()],
                    dtype=compressed_leads[lead].attrs["dtype"],
                )
                if voltages.shape[0] != self.ecg_len:
                    voltages = np.interp(
                        np.linspace(0, 1, self.ecg_len),
                        np.linspace(0, 1, voltages.shape[0]),
                        voltages,
                    )
                ecg[:, self.leads[lead]] = voltages
                
            if self.transforms:
                ecg = self.transforms(ecg)
            
            #ecg = torch.from_numpy(ecg) # no need
            # Compatible with ml4h transform def
            # for transform in self.transforms:
            #     ecg = transform(ecg, loading_option)
                
            return ecg

        
    @property
    def name(self):
        return self._name

    def process_read(self, read: str) -> Dict[str, bool]:
        # TODO: speed this up, does a for loop over read for every kw
        return {
            name: any(kw in read for kw in kws)
            for name, kws in self.TEXT_DIAGNOSES.items()
        }

    def get_meta_data(self, hd5: h5py.Dataset) -> Dict[str, Any]:
        """hd5 should be keyed to desired ECG date"""
        out = {}
        for key in self.META_DATA_FIELDS:
            val = np.nan
            if key in hd5:
                dset = hd5[key]
                val = decompress_data(dset[()], dset.attrs["dtype"])
            out[key] = val
        return out

    def get_summary_data(self, MRN_path, loading_option):
        """
        This function lets us pair down the 5000 x 12 data into something manageable
        for a DataFrame built by data exploration functions
        """
        # ECG meta data
        s3_path = loading_option[self.S3_PATH_OPTION]
        self.download_if_s3(MRN_path, s3_path)

        hd5_path = os.path.join(self._local_path(s3_path), f"{MRN_path}.hd5")
        with h5py.File(hd5_path, "r") as hd5:
            date_str = loading_option[DATE_OPTION_KEY].strftime(
                self.date_format,
            )
            # get meta data
            out = self.get_meta_data(hd5[self.hd5_path_to_ecg][date_str])
            if "read_md_clean" in hd5[self.hd5_path_to_ecg][date_str]:
                read = hd5[self.hd5_path_to_ecg][date_str]["read_md_clean"]
                read = decompress_data(read[()], read.attrs["dtype"]).lower()
                out["read"] = "md"

            elif "read_pc_clean" in hd5[self.hd5_path_to_ecg][date_str]:
                read = hd5[self.hd5_path_to_ecg][date_str]["read_pc_clean"]
                read = decompress_data(read[()], read.attrs["dtype"]).lower()
                out["read"] = "pc"
            else:
                read = ""
                out["read"] = "none"
        
        # Raw ECG information
        ecg = self.get_raw_data(MRN_path, loading_option)
        print(f'ECG from get_raw is: {type(ecg)}')
        out["max_absolute_amp"] = np.abs(ecg).max()
        #out["num_zeros"] = np.count_nonzero(ecg == 0)
        out["s3_path"] = loading_option[self.S3_PATH_OPTION]
        
        for lead_name, lead_idx in ECG_REST_AMP_LEADS.items():
            out[f"{lead_name}_zeros"] = np.count_nonzero(ecg[..., lead_idx] == 0)
            out = {**out, **self.process_read(read)}
        return out

def _extract_1d_tensor(sample, key):
    tensor = sample[key]
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor.view(-1)


class ECGDrugDataset(Dataset):
    """
    Inputs:
    - dataframe: the input dataframe, e.g., train_df
    - ecg_dataset: the object that can fetch ECG data from disc
    - meta_columns: other features besides ecg, dig_value 
     ['hr']
    - options_cache: Here we introduce caching for the ECG loading options via a dictionary 
    to store the results of self.ecg_dd.get_loading_options(MRN_path) keyed by the MRN. 
    Lookup in self.ecg_dd.get_loading_options is expensive since it's an I/O process
    If the same MRN is accessed again, we can directly retrieve its options 
    from the cache without recomputing.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame,       
                 ecg_dataset: ECGDataDescription, # object that can fetch ECG data
                 meta_columns: Optional[List[str]] = None, 
                 seed: int = 42,
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 ecg_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 ecg_augment: Optional[Callable[[Tensor], Tensor]] = None,
                 ecg_augment_prob:float = 0.0,
                 use_raw_digoxin: bool = False,
                 imbalance: bool = False
                ):
        
        super().__init__()
        self.df = dataframe.copy()
        self.ecg_dd = ecg_dataset
        self.meta_columns = meta_columns if meta_columns else []
        self.transform = transform
        self.ecg_transform = ecg_transform
        self.ecg_augment = ecg_augment
        self.ecg_augment_prob = ecg_augment_prob
        self.use_raw_digoxin = use_raw_digoxin
        # Initialize the cache dictionary.
        self.options_cache = {}
        self.verbose = False
        self.imbalance = imbalance
        self.intervals_exist = False
        missing_cols = [col for col in self.meta_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
        if 'hr' in self.meta_columns:
            self.intervals_exist = True
            
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int):
        r""" 
        HERE WE CHOOSE THE DATETIME TO ECG
        Takes an index corresponding to a row in our dataframe and 
        returns the contents in the order expected by the training loop:
        - ecg_tensor
            - retrieves using the ecg_datetime from the ecg file of
            the designated hospital,
            - transforms as indicated in the ECGDataDescription
                - .view(-1) reshapes this scalar tensor into a 1D tensor 
                with a single element, as that is what our time models wants.
        - dig_tensor: for regression, transforms as indicated by the self.transform
        - female_tensor: categorical variable (0=male, 1=female)
        - age_tensor: transforms as indicated by the self.transform
        - MRN_path: string e.g. 'ecg_bwh_hd5s/72389' (bundle in meta)
        - row_index: save the dataframe row index for optional later 
          processing (bundle in meta)
        - *meta = 'MRN_path', int(row_index), OPTIONAL 'qrs', 'hr', 'qt', 'qtc'        
        """
        
        if index >= len(self.df):
            raise IndexError(f"Index {index} out of bounds for \
            dataset with length {len(self.df)}")
        
        row = self.df.iloc[index]
        row_index = index  
        
        MRN_path = row['MRN_path']
        ecg_dig_date = row['ecg_datetime'] 
    
        # Categorical variables - binary or 3-way
        high_dig_tensor = torch.tensor(row['high_digoxin'], dtype=torch.long)
        female_tensor = torch.tensor(row['female'], dtype=torch.long)

        # Numerical
        dig_level_tensor = torch.tensor(row['dig_level'], dtype=torch.float32)
        age_tensor = torch.tensor(row['age_at_ecg_days'], dtype=torch.float32)
     
        # Numericals to be standardized, Sample dict has a structure instantiated 
        # at the dataloaders.py using stats from train
        sample = {
            'dig_level': dig_level_tensor,
            'age_at_ecg_days': age_tensor,
        }

        
        # Add extra numerical interval variables if they exist
        if self.intervals_exist:
            hr_tensor = torch.tensor(row['hr'], dtype=torch.float32)
            # Add to standardize dict 
            sample['hr'] = hr_tensor
        
        if self.transform:
            sample = self.transform(sample.copy())
            inv_sample = self.transform.transforms[0].inverse(sample.copy())
        
        dig_level_tensor = _extract_1d_tensor(sample, 'dig_level')
        age_tensor = _extract_1d_tensor(sample, 'age_at_ecg_days')
        hr_tensor = _extract_1d_tensor(sample, 'hr') if self.intervals_exist else None
        
        # ------
        # Load ECG data, apply augmentations, etc.
        # -----
        
        try: 
            if MRN_path in self.options_cache:
                ecg_dts = self.options_cache[MRN_path]
            else:
                ecg_dts = self.ecg_dd.get_loading_options(MRN_path)
                self.options_cache[MRN_path] = ecg_dts

            # opt[datetime] is of kind '2015-11-11 14:45:06'
            # 1. Exact datetime match
            option = next((opt for opt in ecg_dts if opt['datetime'] == ecg_dig_date), None)

            # 2. Fallback: match within same hour and a half
            if option is None:
                target_dt = ecg_dig_date
                if isinstance(target_dt, str):
                    target_dt = datetime.strptime(target_dt, "%Y-%m-%dT%H:%M:%S")
                
                def within_one_hour(dt_str_or_dt, 
                                    target_dt, threshold=timedelta(hours=1.5)):
                    dt = dt_str_or_dt
                    if isinstance(dt, str):
                        dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
                    return abs(dt - target_dt) <= threshold
                
                # Fallback match if exact datetime not found
                option = next((opt for opt in ecg_dts if within_one_hour(opt['datetime'], ecg_dig_date)),
                    None
                )

            # 3. ECG does not exist inside hour - return None
            if option is None:
                msg = f" - No ECG data found for MRN_path: {MRN_path} within 1.5h from ecg_datetime: {ecg_dig_date}"
                print(msg, file=sys.stderr)
                return None

            ecg_tensor = self.ecg_dd.get_raw_data(MRN_path, option)
            
            # 4. Double check for corrupted ECGs
            if not is_valid_ecg(ecg_tensor, abs_threshold=10.0):
                # bad or corrupt ECG, skip it
                print(f'Found corrupted ECG:{MRN_path}-{ecg_dig_date}')
                return None

            # 5. ECG Found!: Apply ECG augmentations (optional)
            if random.random() < self.ecg_augment_prob:
                if self.ecg_augment:
                    #print('Doing ecg augment!')
                    ecg_tensor = self.ecg_augment(ecg_tensor.unsqueeze(0)).squeeze(0)

            # 6. Apply regular transforms
            #print("Before transform ECG range:", ecg_tensor.min(), ecg_tensor.max())

            if self.ecg_transform:
                ecg_tensor = self.ecg_transform(ecg_tensor)             
                if not is_bound_ecg(ecg_tensor):
                    print(f'[NOTE] LARGE VALUES for MRN_path: {MRN_path} - {ecg_dig_date}', file=sys.stderr)

        # If the whole file is missing throw exception
        except Exception as e:
            msg = f"Exception loading MRN_path: {MRN_path} at ecg_datetime: {ecg_dig_date} → {str(e)}"
            print(msg)
            return None

        # Process additional meta-data for post-processing reasons
        # exclude intervals which are taking place in training and are
        # being transformed
        # (not participating in training)
        # FUTURE VERSIONS: remove this feature, if we have the index we can 
        # recover other features
        meta_list = []
        for col in self.meta_columns:
            if col in ['hr']:
                continue
            value = row[col]
            meta_tensor = torch.tensor(value, dtype=torch.float32).view(-1)
            meta_list.append(meta_tensor)
        
        # Attach MRN_path and row_index to meta-data.
        # these might prove useful in post-processing
        
        meta_tuple = (MRN_path, ) + (row_index, ) + tuple(meta_list)
        
        item = (ecg_tensor, dig_level_tensor, high_dig_tensor, female_tensor, age_tensor) 
        if self.intervals_exist:
            item = item + tuple([hr_tensor, qt_tensor])
            
        # Data returned by ECGDrugDataset: 
        # if meta_columns = []
        # (ecg_tensor, dig_level_tensor, high_dig_tensor, 
        # female_tensor, age_tensor, MRN_path, row_index)

        return item + meta_tuple
    
    
